"""Bounded CPU conversion of original H3 BF16 DiT weights to packed MXFP8."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import torch

from fasth3_kernels.mxfp8 import FORMAT_VERSION, PACKED_SCALE_LAYOUT, quantize_mxfp8
from .storage import PersistentTensorStore, INDEX
from .mxfp8_linear import MANIFEST
_TYPES={torch.float32:'F32',torch.float64:'F64',torch.float16:'F16',torch.bfloat16:'BF16',
        torch.float8_e4m3fn:'F8_E4M3',torch.float8_e5m2:'F8_E5M2',torch.uint8:'U8',
        torch.int64:'I64',torch.int32:'I32',torch.int16:'I16',torch.int8:'I8',torch.bool:'BOOL'}


def _positive(value,name):
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _json_bytes(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')


def _config(directory):
    root=Path(directory).resolve(strict=True)
    path=(root/'config.json').resolve(strict=True)
    if path.parent!=root: raise ValueError('unsafe source config path')
    if path.stat().st_size>1<<20: raise ValueError('oversized source config')
    raw=path.read_bytes(); json.loads(raw)
    return raw


def _model_spec(config, profile):
    from .transformer import build_meta_transformer

    model = build_meta_transformer(config, profile=profile)
    suffixes = ('attn.to_q', 'attn.to_k', 'attn.to_v', 'attn.to_out.0', 'ff.net.0.proj', 'ff.net.2')
    selected = set()
    for index, block in enumerate(model.transformer_blocks):
        for suffix in suffixes:
            if not isinstance(block.get_submodule(suffix), torch.nn.Linear):
                raise ValueError(f'Expected a linear module at transformer_blocks.{index}.{suffix}')
            selected.add(f'transformer_blocks.{index}.{suffix}.weight')
    return {name: tuple(value.shape) for name, value in model.state_dict().items()}, selected


def plan_export(source, *, profile='omni-pdd8', memory_bytes=256 << 20, row_chunk=128):
    """Validate checkpoint headers against the Meta model without reading payloads."""
    _positive(memory_bytes, 'memory_bytes')
    _positive(row_chunk, 'row_chunk')
    config = _config(source)
    model_shapes, selected = _model_spec(json.loads(config), profile)
    if not selected or not selected <= model_shapes.keys():
        raise ValueError('Invalid model linear selection')
    with PersistentTensorStore(source, backend='readonly_mmap') as src:
        if src.weight_map.keys() != model_shapes.keys():
            missing = sorted(model_shapes.keys() - src.weight_map.keys())
            extra = sorted(src.weight_map.keys() - model_shapes.keys())
            raise ValueError(f'Checkpoint keys mismatch: missing={missing[:8]}, extra={extra[:8]}')
        entries, linears, max_quant_work = [], {}, 0
        for key in sorted(src.weight_map):
            info = src.info(key)
            if info.shape != model_shapes[key]:
                raise ValueError(f'Checkpoint shape mismatch: {key}')
            if key not in selected:
                if info.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    raise ValueError(f'Checkpoint passthrough is already quantized: {key}')
                entries.append(dict(key=key, source_key=key, shape=list(info.shape), dtype=_TYPES[info.dtype],
                                    nbytes=info.nbytes, role='passthrough'))
                continue
            if len(info.shape) != 2 or info.dtype != torch.bfloat16:
                raise ValueError(f'Quantization requires original BF16 matrix: {key}')
            n, k = info.shape
            if min(n, k) < 1 or n % 32 or k % 128:
                raise ValueError(f'tile32 requires N divisible by 32 and K by 128: {key}')
            per_row = 66 * k + k // 32
            rows = min(n, row_chunk, memory_bytes // per_row) // 32 * 32
            if rows < 1:
                raise ValueError(f'One 32-row scale tile exceeds memory budget: {key}')
            max_quant_work = max(max_quant_work, rows * per_row)
            scale_key = key + '_scale'
            entries.extend([
                dict(key=key, source_key=key, shape=[n, k], dtype='F8_E4M3', nbytes=n * k, role='weight'),
                dict(key=scale_key, source_key=key, shape=[n // 32, k // 128, 32, 4],
                     dtype='U8', nbytes=n * (k // 32), role='scale'),
            ])
            linears[key[:-len('.weight')]] = dict(original_k=k, padded_k=k, out_features=n, row_chunk=rows,
                                                 weight_key=key, scale_key=scale_key)
        for index, entry in enumerate(entries, 1):
            entry['shard'] = f'tensor-{index:05d}.safetensors'
        result = dict(
            schema='fasth3-mxfp8-export-v1', format=FORMAT_VERSION, complete=False, profile=profile,
            source=str(Path(source).resolve()),
            source_index_sha256=hashlib.sha256((Path(source) / INDEX).read_bytes()).hexdigest(),
            source_config_sha256=hashlib.sha256(config).hexdigest(),
            source_shards={name: dict(size=(Path(source) / name).stat().st_size,
                                      mtime_ns=(Path(source) / name).stat().st_mtime_ns)
                           for name in sorted(set(src.weight_map.values()))},
            memory_budget_scope='owned CPU conversion temporaries; excludes OS page cache and mappings',
            quantized_linears=len(linears), scale_layout=PACKED_SCALE_LAYOUT,
            scale_dtype='UInt8 unsigned E8M0 raw bytes; 255 reserved', block_size=32,
            exponent_implementation='frexp_exact_ceil_v1', zero_scale_byte=127,
            memory_bytes=memory_bytes, max_quantization_working_bytes=max_quant_work,
            row_chunk_limit=row_chunk, linears=linears, tensors=entries, source_stats=src.stats_snapshot(),
            payload_bytes=sum(entry['nbytes'] for entry in entries),
        )
        overhead = 8 * len(_json_bytes(result)) + len(config) + (1 << 20) + 8192 * len(entries)
        result['estimated_disk_bytes_upper_bound'] = result['payload_bytes'] + overhead
        return result


def _header(entry):
    header={entry['key']:dict(dtype=entry['dtype'],shape=entry['shape'],data_offsets=[0,entry['nbytes']]),
            '__metadata__':{'format':'pt','mxfp8_format':FORMAT_VERSION}}
    raw=json.dumps(header,separators=(',',':')).encode('utf-8')
    raw+=b' '*((-len(raw))%8)
    return struct.pack('<Q',len(raw))+raw


class _TensorWriter:
    def __init__(self,path,entry):
        self.file=path.open('xb'); self.expected=entry['nbytes']; self.written=0; self.digest=hashlib.sha256()
        self._write(_header(entry))
    def _write(self,data):
        count=self.file.write(data)
        if count!=len(data): raise IOError('short output write')
        self.digest.update(data)
    def add(self,tensor):
        raw=memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast('B')
        if self.written+len(raw)>self.expected: raise ValueError('excess output payload')
        self._write(raw); self.written+=len(raw)
    def close(self):
        self.file.close()
        if self.written!=self.expected: raise IOError('incomplete output payload')
        return self.digest.hexdigest()


def export_mxfp8(source, output, *, profile='omni-pdd8', memory_bytes=256<<20, row_chunk=128,
                 reserve_disk_bytes=1<<30):
    """Explicit payload export; caller must serialize SSD use with model runs."""
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('refuse existing output directory')
    if output.is_relative_to(Path(source).resolve()):
        raise ValueError('output must not modify source directory')
    plan=plan_export(source, profile=profile, memory_bytes=memory_bytes, row_chunk=row_chunk)
    ancestor=output.parent
    while not ancestor.exists(): ancestor=ancestor.parent
    if isinstance(reserve_disk_bytes,bool) or not isinstance(reserve_disk_bytes,int) or reserve_disk_bytes<0:
        raise ValueError('reserve_disk_bytes must be a nonnegative integer')
    if shutil.disk_usage(ancestor).free < plan['estimated_disk_bytes_upper_bound']+reserve_disk_bytes:
        raise OSError('insufficient free disk including reserve')
    if hashlib.sha256((Path(source)/INDEX).read_bytes()).hexdigest()!=plan['source_index_sha256']:
        raise RuntimeError('source index changed after planning')
    for shard,identity in plan['source_shards'].items():
        stat=(Path(source)/shard).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(identity['size'],identity['mtime_ns']):
            raise RuntimeError('source shard changed after planning')
    output.mkdir(parents=True,exist_ok=False)
    config=_config(source)
    if hashlib.sha256(config).hexdigest()!=plan['source_config_sha256']: raise RuntimeError('source config changed')
    with (output/'config.json').open('xb') as f: f.write(config)
    entries={e['key']:e for e in plan['tensors']}; hashes={}; weight_map={}
    with PersistentTensorStore(source,backend='readonly_mmap') as src:
        for key in sorted(src.weight_map):
            entry=entries[key]; writer=_TensorWriter(output/entry['shard'],entry)
            scale_writer=None
            try:
                if entry['role']=='weight':
                    spec=plan['linears'][key[:-len('.weight')]]; n,k=src.shape(key)
                    scale_entry=entries[key+'_scale']; scale_writer=_TensorWriter(output/scale_entry['shard'],scale_entry)
                    for first in range(0,n,spec['row_chunk']):
                        count=min(spec['row_chunk'],n-first)
                        raw=torch.empty(count*k*2,dtype=torch.uint8)
                        got=src.read_into(key,raw,byte_offset=first*k*2)
                        if type(got)!=int or got!=raw.numel(): raise IOError('short source row read')
                        q=quantize_mxfp8(raw.view(torch.bfloat16).reshape(count,k))
                        scales = q.scales.view(count//32,32,k//128,4).permute(0,2,1,3)
                        writer.add(q.values); scale_writer.add(scales)
                        del scales
                        del q,raw
                    hashes[scale_entry['shard']]=scale_writer.close(); scale_writer=None
                    weight_map[scale_entry['key']]=scale_entry['shard']
                else:
                    size=src.info(key).nbytes; chunk=min(max(1,memory_bytes//2),8<<20)
                    for offset in range(0,size,chunk):
                        raw=torch.empty(min(chunk,size-offset),dtype=torch.uint8)
                        got=src.read_into(key,raw,byte_offset=offset)
                        if type(got)!=int or got!=raw.numel(): raise IOError('short source byte read')
                        writer.add(raw)
                hashes[entry['shard']]=writer.close(); writer=None
                weight_map[key]=entry['shard']
            finally:
                if writer is not None: writer.file.close()
                if scale_writer is not None: scale_writer.file.close()
        stats=src.stats_snapshot()
    if set(weight_map)!=set(entries): raise RuntimeError('incomplete output coverage')
    complete={**plan,'complete':True,'shard_sha256':hashes,'execution_source_stats':stats,
              'publication':'complete manifest precedes final index; missing index means incomplete export'}
    # Index is the final publication marker: no normal store opens a partial
    # export before ALL payloads and the complete manifest are available.
    with (output/MANIFEST).open('xb') as f: f.write(_json_bytes(complete))
    with (output/INDEX).open('xb') as f:
        f.write(_json_bytes({'metadata':{'total_size':plan['payload_bytes'],'mxfp8_format':FORMAT_VERSION},'weight_map':weight_map}))
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--profile', choices=('omni-pdd8', 't2va-v2'), default='omni-pdd8')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--memory-mib', type=int, default=256)
    parser.add_argument('--row-chunk', type=int, default=128)
    parser.add_argument('--reserve-disk-mib', type=int, default=1024)
    args = parser.parse_args(argv)
    options = dict(profile=args.profile, memory_bytes=args.memory_mib << 20, row_chunk=args.row_chunk)
    if args.execute:
        if args.output is None:
            parser.error('--execute requires --output')
        result = export_mxfp8(args.source, args.output, reserve_disk_bytes=args.reserve_disk_mib << 20, **options)
    else:
        result = plan_export(args.source, **options)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
