"""MXFP8 Linear and streamed-checkpoint adapter.

The native path checks weight metadata on every forward. Debug mode adds a
bounded scan of weight bytes and one scalar synchronization. The default output
tile covers full N when its FP32 epilogue budget fits.
"""
import hashlib
import json
import re

import torch
from torch import nn
from fasth3_kernels.mxfp8 import (
    FORMAT_VERSION, PACKED_SCALE_LAYOUT, ROW_MAJOR_SCALE_LAYOUT, MXFP8Tensor,
    quantize_mxfp8, quantize_mxfp8_cuda, quantize_mxfp8_cute,
)
from .storage import _TYPES, _unique

MANIFEST = 'mxfp8_manifest.json'


def replace_mxfp8_linears(model, store, *, debug_checks=False):
    """Verify the export metadata and install meta modules without reading payloads."""
    path = (store.directory / MANIFEST).resolve(strict=True)
    if path.parent != store.directory or path.stat().st_size > 8 << 20:
        raise ValueError('Unsafe or oversized MXFP8 manifest')
    manifest = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique)
    if (not isinstance(manifest, dict) or manifest.get('schema') != 'fasth3-mxfp8-export-v1' or
            manifest.get('format') != FORMAT_VERSION or manifest.get('complete') is not True or
            manifest.get('block_size') != 32 or manifest.get('zero_scale_byte') != 127 or
            manifest.get('exponent_implementation') != 'frexp_exact_ceil_v1'):
        raise ValueError('Unsupported or incomplete MXFP8 manifest')
    scale_layout = manifest.get('scale_layout', ROW_MAJOR_SCALE_LAYOUT)
    if scale_layout not in (ROW_MAJOR_SCALE_LAYOUT, PACKED_SCALE_LAYOUT):
        raise ValueError('Unsupported MXFP8 scale layout')
    packed_scales = scale_layout == PACKED_SCALE_LAYOUT
    config = store.directory / 'config.json'
    if (config.resolve().parent != store.directory or config.stat().st_size > 1 << 20 or
            hashlib.sha256(config.read_bytes()).hexdigest() != manifest.get('source_config_sha256')):
        raise ValueError('MXFP8 config digest mismatch')
    shards = manifest['shard_sha256']
    if (set(shards) != set(store.weight_map.values()) or
            any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in shards.values())):
        raise ValueError('MXFP8 shard hash manifest mismatch')
    tensors = manifest['tensors']
    names = {entry['key'] for entry in tensors}
    if len(tensors) != len(names) or names != set(store.weight_map):
        raise ValueError('MXFP8 manifest tensor coverage mismatch')
    for entry in tensors:
        key = entry['key']
        info = store.info(key)
        if (entry['shard'] != store.weight_map[key] or tuple(entry['shape']) != info.shape or
                _TYPES.get(entry['dtype']) != info.dtype or
                entry['nbytes'] != info.nbytes):
            raise ValueError(f'MXFP8 manifest tensor metadata mismatch: {key}')
    if sum(entry['nbytes'] for entry in tensors) != manifest['payload_bytes']:
        raise ValueError('MXFP8 manifest payload size mismatch')
    linears = manifest['linears']
    scale_keys = {key for key in names if key.endswith('.weight_scale')}
    if not linears or len(linears) != manifest['quantized_linears'] or scale_keys != {
            f'{name}.weight_scale' for name in linears}:
        raise ValueError('MXFP8 manifest linear coverage mismatch')
    for name, spec in linears.items():
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, leaf)
        if not isinstance(original, nn.Linear):
            raise ValueError(f'MXFP8 target must be a Linear: {name}')
        n, k = original.out_features, original.in_features
        padded = (k + 31) // 32 * 32
        weight, scale = name + '.weight', name + '.weight_scale'
        if (spec['weight_key'], spec['scale_key'], spec['out_features'], spec['original_k'], spec['padded_k']) != (
                weight, scale, n, k, padded):
            raise ValueError(f'MXFP8 logical dimensions mismatch: {name}')
        if packed_scales and (n % 32 or k % 128):
            raise ValueError(f'Invalid packed MXFP8 dimensions: {name}')
        scale_shape = (n // 32, k // 128, 32, 4) if packed_scales else (n, padded // 32)
        if (tuple(store.shape(weight)) != (n, padded) or store.dtype(weight) != torch.float8_e4m3fn or
                tuple(store.shape(scale)) != scale_shape or store.dtype(scale) != torch.uint8):
            raise ValueError(f'MXFP8 weight or scale header mismatch: {name}')
        bias = original.bias is not None
        bias_dtype = store.dtype(name + '.bias') if bias else torch.bfloat16
        setattr(parent, leaf, MXFP8Linear(k, n, bias=bias, device='meta', bias_dtype=bias_dtype,
                                        backend='cute-optimized', chunk_tokens=512, debug_checks=debug_checks,
                                        packed_scales=packed_scales))


def _native_validated_gemm(a,b,*,tiled,optimized=None,bias=None,out=None,packed_b=False,tma=False,
                           tma_store=False):
    # Private launcher for format-checked module buffers and quantized activations.
    from fasth3_kernels._cute_mxfp8 import launch
    return launch(a,b,tiled=tiled,optimized=optimized,bias=bias,out=out,packed_b=packed_b,
                  producer_warps=int(tma),tma=tma,tma_store=tma_store)


def _sm120_tile(m):
    if m < 64:
        return (32, 64, 2)
    if 128 < m < 192:
        return (64, 128, 2, 2, 4)
    return (128, 128, 2, 2, 4)


def _sm120_tma_tile(m, n, k):
    """Tiles measured on the full and final token chunks of the 768p DiT."""
    if (n, k) == (7168, 5376):
        if m == 512:
            return (128, 128, 2, 2, 4)
        if m == 136:
            return (64, 128, 2, 2, 4)
    if (n, k) == (28672, 5376):
        if m == 512:
            return (128, 128, 2, 2, 2)
        if m == 136:
            return (64, 128, 3, 2, 4)
    if (n, k) == (5376, 7168):
        if m == 512:
            return (128, 128, 2, 2, 4)
        if m == 136:
            return (128, 128, 2, 2, 2)
    return None


def _oracle_validated_gemm(a,b):
    """Explicit FP32 reference, never invoked on a native failure."""
    def decode(q):
        sf=torch.ldexp(torch.ones_like(q.scales,dtype=torch.float64),q.scales.int()-127)
        expanded=q.values.double().reshape(q.values.shape[0],-1,32)*sf[...,None]
        return expanded.reshape(q.values.shape)[:,:q.logical_k].float()
    return decode(a)@decode(b).T


class MXFP8Linear(nn.Module):
    def __init__(self,in_features,out_features,bias=True,*,device=None,
                 backend='cute',chunk_tokens=128,chunk_out_features=None,
                 bias_dtype=torch.bfloat16,max_working_bytes=256<<20,
                 debug_checks=False,packed_scales=False):
        super().__init__()
        for name,value in [('in_features',in_features),('out_features',out_features),
                           ('chunk_tokens',chunk_tokens),('max_working_bytes',max_working_bytes)]:
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if chunk_out_features is not None and (isinstance(chunk_out_features,bool)
                or not isinstance(chunk_out_features,int) or chunk_out_features<1):
            raise ValueError('chunk_out_features must be None or a positive integer')
        if backend not in ('cute','cute-tiled','cute-tiled-fused','cute-optimized','oracle'):
            raise ValueError('unknown MXFP8 Linear backend')
        if packed_scales and (backend!='cute-optimized' or out_features%32 or in_features%128):
            raise ValueError('packed MXFP8 scales require optimized GEMM, N32 and K128 alignment')
        if bias_dtype not in (torch.float32,torch.bfloat16,torch.float16):
            raise ValueError('unsupported bias dtype')
        self.in_features,self.out_features=in_features,out_features
        self.padded_k=(in_features+31)//32*32
        self.backend,self.chunk_tokens=backend,chunk_tokens
        self.debug_checks=debug_checks
        self.tma_store = False
        self.packed_scales=packed_scales
        self.max_working_bytes=max_working_bytes
        m,k=chunk_tokens,self.padded_k
        self.eager_chunk_tokens = min(m, max_working_bytes//(64*k)) if backend=='cute-optimized' else m
        if self.eager_chunk_tokens<1 or 64*self.eager_chunk_tokens*k>max_working_bytes:
            raise ValueError('configured chunks exceed conservative working-byte budget')
        # Native B is already FP8: no fictional FP64 weight-quantization budget.
        # Activation codes persist during GEMM; include 2*M*K beside epilogue.
        limit=(max_working_bytes-2*m*k)//(8*m)
        if backend=='oracle':
            # Only explicit reference actually dequantizes weight rows in FP64.
            limit=min(limit,max_working_bytes//(64*k)-m)
        if limit<1: raise ValueError('configured chunks exceed conservative working-byte budget')
        self.chunk_out_features=min(out_features,chunk_out_features or out_features,limit)
        if packed_scales:
            self.chunk_out_features=self.chunk_out_features//32*32
            if self.chunk_out_features<32:
                raise ValueError('packed MXFP8 requires a 32-row weight tile')
        self.validation_rows=min(out_features,128,max(1,max_working_bytes//(8*k)))
        if self.workspace_upper_bound()>max_working_bytes:
            raise ValueError('configured chunks exceed conservative working-byte budget')
        self.register_buffer('weight',torch.empty((out_features,k),dtype=torch.float8_e4m3fn,device=device))
        self.scale_shape=(out_features//32,k//128,32,4) if packed_scales else (out_features,k//32)
        self.register_buffer('weight_scale',torch.empty(self.scale_shape,dtype=torch.uint8,device=device))
        self.bias=nn.Parameter(torch.empty(out_features,dtype=bias_dtype,device=device),requires_grad=False) if bias else None

    def workspace_upper_bound(self):
        """Sequential-phase bound; persistent weights/input/final output excluded.

        Does not account for CUDA context/JIT/allocator caching. Noncontiguous
        input reshape can copy in original dtype (not full-M FP64).
        """
        m,n,k=self.chunk_tokens,self.chunk_out_features,self.padded_k
        # The fused BF16 kernel writes FP8/scales directly. Allow for the
        # contiguous BF16 copy and debug finite-scan temporaries.
        quantization = 64*self.eager_chunk_tokens*k
        if self.backend=='cute-optimized':
            quantization = max(quantization,12*m*k)
        phases=[quantization,8*m*n+2*m*k,8*self.validation_rows*k]
        if self.backend=='oracle': phases.append(64*(m+n)*k)
        return max(phases)

    def _validate_weight_once(self,x):
        if self.weight.device!=x.device or self.weight_scale.device!=x.device or (self.bias is not None and self.bias.device!=x.device):
            raise ValueError('input and all streamed buffers must share device')
        if self.weight.dtype!=torch.float8_e4m3fn or self.weight_scale.dtype!=torch.uint8:
            raise ValueError('invalid exported weight/scale dtype')
        if self.weight.shape!=(self.out_features,self.padded_k) or self.weight_scale.shape!=self.scale_shape:
            raise ValueError('invalid exported weight/scale shape')
        if not self.weight.is_contiguous() or not self.weight_scale.is_contiguous():
            raise ValueError('weight and scale must be contiguous')
        if self.bias is not None and (self.bias.shape!=(self.out_features,) or self.bias.dtype not in (torch.float32,torch.bfloat16,torch.float16)):
            raise ValueError('invalid bias metadata')
        if not self.debug_checks and self.backend!='oracle':
            return
        # Raw E4M3 FN NaNs are exactly 0x7f and 0xff. Bitwise byte checks
        # avoid materializing the full weight as FP32 or FP64. Accumulate device
        # flags across row chunks, synchronize once only AFTER all reductions.
        invalid=torch.zeros(3,dtype=torch.bool,device=self.weight.device)
        scale_bytes=self.weight_scale.view(self.out_features,self.padded_k//32)
        for first in range(0,self.out_features,self.validation_rows):
            last=min(first+self.validation_rows,self.out_features)
            raw=self.weight[first:last].view(torch.uint8)
            invalid[0].logical_or_(((raw&127)==127).any())
            invalid[1].logical_or_((scale_bytes[first:last]==255).any())
            if self.in_features<self.padded_k:
                invalid[2].logical_or_(((raw[:,self.in_features:]&127)!=0).any())
        if bool(invalid.any()):
            errors=[name for name,bad in zip(('NaN E4M3 weight','reserved E8M0 scale','nonzero K padding'),invalid.tolist()) if bad]
            raise ValueError('invalid MXFP8 buffers: '+', '.join(errors))

    def forward(self,x,*,swiglu=False,packed=None,output=None,columns=None):
        input_width = self.in_features * (2 if swiglu else 1)
        if x.ndim<1 or x.shape[-1]!=input_width:
            raise ValueError('input K differs from logical original K')
        if x.dtype not in (torch.float32,torch.bfloat16,torch.float16):
            raise ValueError('input must be FP32/BF16/FP16')
        if self.backend!='oracle' and x.device.type!='cuda':
            raise RuntimeError('native MXFP8 Linear requires CUDA; CPU oracle is explicit')
        self._validate_weight_once(x)
        flat=x.reshape(-1,input_width)
        first_col, end_col = (0, self.out_features) if columns is None else columns
        if columns is not None and (self.backend != 'cute-optimized' or
                                    not 0 <= first_col < end_col <= self.out_features or
                                    (first_col % 32 or end_col % 32)):
            raise ValueError('Partial MXFP8 projections require aligned native output columns')
        width = end_col - first_col
        output_shape = (flat.shape[0], width)
        if output is not None and (output.shape != output_shape or output.dtype != x.dtype or
                                   output.device != x.device or not output.is_contiguous()):
            raise ValueError('MXFP8 output buffer must match the flat output')
        result = x.new_empty(output_shape) if output is None else output
        optimized=self.backend=='cute-optimized'
        fused_quantizer=self.backend in ('cute-tiled-fused','cute-optimized') and x.dtype==torch.bfloat16
        quantizer=quantize_mxfp8_cute if fused_quantizer else quantize_mxfp8_cuda
        chunk_tokens=self.chunk_tokens if fused_quantizer else self.eager_chunk_tokens
        if swiglu and (not fused_quantizer or packed is not None):
            raise ValueError('Fused SwiGLU requires the BF16 CuTe quantizer')
        if packed is not None and (not fused_quantizer or len(flat)>chunk_tokens or
                                   packed.logical_k!=self.in_features or len(packed.values)!=len(flat)):
            raise ValueError('Shared MXFP8 activation must match one BF16 token chunk')
        for begin in range(0,flat.shape[0],chunk_tokens):
            end=min(begin+chunk_tokens,flat.shape[0])
            chunk=flat[begin:end]
            if packed is not None:
                activation = packed
            elif swiglu and not self.debug_checks:
                from fasth3_kernels._cute_mxfp8 import launch_quantizer
                values, scales = launch_quantizer(chunk, swiglu=True)
                activation = MXFP8Tensor(values, scales, self.in_features)
            elif swiglu:
                value, gate = chunk.chunk(2, dim=-1)
                activation = quantizer(value * torch.nn.functional.silu(gate), check_finite=True)
            elif self.backend=='oracle':
                activation=quantize_mxfp8(chunk)
            else:
                activation=quantizer(chunk,check_finite=self.debug_checks)
            for first in range(first_col,end_col,self.chunk_out_features):
                last=min(first+self.chunk_out_features,end_col)
                if self.packed_scales:
                    scales=self.weight_scale[first//32:last//32]
                else:
                    scales=self.weight_scale[first:last]
                weight=MXFP8Tensor(self.weight[first:last],scales,self.in_features)
                tma_tile = None
                if optimized and self.packed_scales and fused_quantizer:
                    config_width = self.out_features if columns is not None else last-first
                    tma_tile = _sm120_tma_tile(end-begin,config_width,self.padded_k)
                config=(tma_tile or _sm120_tile(end-begin)) if optimized else None
                bias = None if self.bias is None else self.bias[first:last]
                if config and last-first==width:
                    tma_store = (self.tma_store and tma_tile is not None and result.dtype == torch.bfloat16
                                 and (end-begin) % config[0] == 0 and width % config[1] == 0)
                    _native_validated_gemm(activation,weight,tiled=True,optimized=config,
                                           bias=bias,out=result[begin:end],packed_b=self.packed_scales,
                                           tma=tma_tile is not None,tma_store=tma_store)
                    continue
                if self.backend=='oracle':
                    accum=_oracle_validated_gemm(activation,weight)
                else:
                    accum=_native_validated_gemm(activation,weight,tiled=self.backend!='cute',optimized=config,
                                                  packed_b=self.packed_scales,tma=tma_tile is not None)
                if bias is not None: accum.add_(bias)
                result[begin:end,first-first_col:last-first_col]=accum.to(x.dtype)
                del accum
            del activation
        return result.reshape(*x.shape[:-1],width)
