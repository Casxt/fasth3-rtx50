"""Synthetic-only export/Linear tests. No real model payload or CUDA access."""
import hashlib
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from fasth3_kernels.mxfp8 import quantize_mxfp8,dequantize_mxfp8,MXFP8Tensor
from fasth3_pipeline.mxfp8_linear import MXFP8Linear,replace_mxfp8_linears
from fasth3_pipeline.storage import PersistentTensorStore,INDEX
from fasth3_pipeline.weight_conversion import plan_export,export_mxfp8,MANIFEST
from fasth3_pipeline import weight_conversion


def _checkpoint(folder,tensors):
    folder.mkdir()
    save_file(tensors,str(folder/'part.safetensors'))
    (folder/INDEX).write_text(json.dumps({'weight_map':{k:'part.safetensors' for k in tensors}}),encoding='utf-8')
    (folder/'config.json').write_text('{"original_config": true, "attention_head_dim": 128}\n',encoding='utf-8')


def _pair(tmp_path, monkeypatch):
    torch.manual_seed(19)
    tensors={'a.weight':torch.randn(32,128).bfloat16(),'a.bias':torch.randn(32),
             'b.weight':torch.randn(32,128).bfloat16(),'gates.weight':torch.randn(2,2).bfloat16(),
             'pdd.weight':torch.randn(2,2),'running':torch.tensor(1.),'empty':torch.empty(0,2)}
    source=tmp_path/'source'
    _checkpoint(source,tensors)
    monkeypatch.setattr(weight_conversion, '_model_spec',
                        lambda config, profile: ({k: tuple(v.shape) for k, v in tensors.items()},
                                                 {'a.weight', 'b.weight'}))
    return source,tensors


def test_dryrun_headers_only_and_exact_coverage(tmp_path,monkeypatch):
    source,tensors=_pair(tmp_path,monkeypatch)
    def forbidden(*args,**kwargs): raise AssertionError('payload requested during dryrun')
    monkeypatch.setattr(PersistentTensorStore,'tensor',forbidden)
    monkeypatch.setattr(PersistentTensorStore,'read_into',forbidden)
    plan=plan_export(source,memory_bytes=1<<20,row_chunk=32)
    assert not plan['complete'] and plan['quantized_linears']==2
    assert set(e['key'] for e in plan['tensors'])==set(tensors)|{'a.weight_scale','b.weight_scale'}
    assert plan['linears']['a']['original_k']==128 and plan['linears']['a']['padded_k']==128
    assert plan['max_quantization_working_bytes']<=1<<20
    assert plan['source_stats']['shard_opens']==0 and plan['source_stats']['tensor_requests']==0


def test_export_exact_streamed_buffers_hashes_and_store_meta_validation(tmp_path,monkeypatch):
    source,tensors=_pair(tmp_path,monkeypatch); output=tmp_path/'mx'
    before=hashlib.sha256((source/'part.safetensors').read_bytes()).hexdigest()
    manifest=export_mxfp8(source,output,memory_bytes=1<<20,row_chunk=32,reserve_disk_bytes=0)
    assert manifest['complete'] and (output/INDEX).exists() and (output/MANIFEST).exists()
    assert (output/'config.json').read_bytes()==(source/'config.json').read_bytes()
    assert hashlib.sha256((source/'part.safetensors').read_bytes()).hexdigest()==before
    with PersistentTensorStore(output,backend='readonly_mmap') as store:
        for prefix in ('a','b'):
            q=quantize_mxfp8(tensors[prefix+'.weight'])
            assert torch.equal(store.tensor(prefix+'.weight').view(torch.uint8),q.values.view(torch.uint8))
            scales=store.tensor(prefix+'.weight_scale').permute(0,2,1,3).reshape_as(q.scales)
            assert torch.equal(scales,q.scales)
        for key in ('a.bias','gates.weight','pdd.weight','running','empty'):
            assert torch.equal(store.tensor(key),tensors[key])
        model=torch.nn.Module()
        model.a=torch.nn.Linear(128,32,device='meta')
        model.b=torch.nn.Linear(128,32,bias=False,device='meta')
        model.gates=torch.nn.Linear(2,2,bias=False,device='meta',dtype=torch.bfloat16)
        model.pdd=torch.nn.Linear(2,2,bias=False,device='meta',dtype=torch.float32)
        model.register_buffer('running',torch.empty((),device='meta'))
        model.register_buffer('empty',torch.empty(0,2,device='meta'))
        replace_mxfp8_linears(model,store,debug_checks=True)
        assert isinstance(model.a,MXFP8Linear) and model.a.bias.dtype == torch.float32
        assert isinstance(model.b,MXFP8Linear) and model.b.bias is None
        assert model.a.backend == model.b.backend == 'cute-optimized'
        assert model.a.debug_checks and model.b.debug_checks
        store.validate_keys(model)
        assert store.dtype('a.weight_scale')==torch.uint8
    for shard,expected in manifest['shard_sha256'].items():
        assert hashlib.sha256((output/shard).read_bytes()).hexdigest()==expected
    assert sum(p.stat().st_size for p in output.iterdir())<=manifest['estimated_disk_bytes_upper_bound']


def test_runtime_refuses_tampered_export_metadata(tmp_path,monkeypatch):
    source,_=_pair(tmp_path,monkeypatch); output=tmp_path/'mx'
    manifest=export_mxfp8(source,output,memory_bytes=1<<20,reserve_disk_bytes=0)
    manifest['tensors'][0]['dtype']='I64'
    (output/MANIFEST).write_text(json.dumps(manifest),encoding='utf-8')
    with PersistentTensorStore(output,backend='file_readinto') as store:
        with pytest.raises(ValueError,match='metadata mismatch'):
            replace_mxfp8_linears(torch.nn.Module(),store)


def test_export_packed_scales_across_row_chunks(tmp_path,monkeypatch):
    source,output=(tmp_path/name for name in ('source','packed'))
    weight=torch.randn(96,256).bfloat16()
    _checkpoint(source,{'a.weight':weight})
    monkeypatch.setattr(weight_conversion, '_model_spec', lambda config, profile: ({'a.weight': (96, 256)}, {'a.weight'}))
    export_mxfp8(source,output,row_chunk=64,reserve_disk_bytes=0)
    q=quantize_mxfp8(weight)
    with PersistentTensorStore(output,backend='file_readinto') as store:
        model=torch.nn.Module()
        model.a=torch.nn.Linear(256,96,bias=False,device='meta')
        replace_mxfp8_linears(model,store)
        store.validate_keys(model)
        assert model.a.packed_scales
        assert torch.equal(store.tensor('a.weight').view(torch.uint8),q.values.view(torch.uint8))
        scales=store.tensor('a.weight_scale').permute(0,2,1,3).reshape_as(q.scales)
        assert torch.equal(scales,q.scales)


def test_failure_never_publishes_manifest_or_index(tmp_path,monkeypatch):
    source,_=_pair(tmp_path,monkeypatch); output=tmp_path/'failed'
    monkeypatch.setattr('fasth3_pipeline.weight_conversion.quantize_mxfp8',lambda x: (_ for _ in ()).throw(RuntimeError('injected')))
    with pytest.raises(RuntimeError,match='injected'):
        export_mxfp8(source,output,reserve_disk_bytes=0)
    assert not (output/INDEX).exists() and not (output/MANIFEST).exists()
    with pytest.raises(FileExistsError): export_mxfp8(source,output)


def test_export_rejects_disk_existing_unsafe_and_coverage(tmp_path,monkeypatch):
    source,_=_pair(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError): export_mxfp8(source,source)
    with pytest.raises(ValueError,match='must not modify'):
        export_mxfp8(source,source/'new')
    with pytest.raises(ValueError,match='One 32-row scale tile'):
        plan_export(source,memory_bytes=1)
    class Disk: free=1
    monkeypatch.setattr('fasth3_pipeline.weight_conversion.shutil.disk_usage',lambda p:Disk())
    with pytest.raises(OSError,match='free disk'):
        export_mxfp8(source,tmp_path/'no-space')
    assert not (tmp_path/'no-space').exists()
    index=json.loads((source/INDEX).read_text()); del index['weight_map']['gates.weight']
    # Rewrite source index without a real tensor too: planning must reject exact
    # key coverage before any payload materialization or missing-key dropping.
    (source/INDEX).write_text(json.dumps(index))
    with pytest.raises(ValueError,match='keys'): plan_export(source)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('bias',[False,True])
def test_linear_explicit_oracle_epilogue_chunks_and_empty(dtype,bias,monkeypatch):
    torch.manual_seed(23)
    layer=MXFP8Linear(33,5,bias=bias,bias_dtype=torch.float32,backend='oracle',chunk_tokens=2,chunk_out_features=3)
    q=quantize_mxfp8(torch.randn(5,33).bfloat16())
    layer.weight=q.values; layer.weight_scale=q.scales
    if bias: layer.bias.data.copy_(torch.randn(5))
    x=torch.randn(2,3,33).to(dtype)
    # Expected activation quantization chunk-independent, FP32 accumulation,
    # FP32 bias, then a SINGLE cast to input dtype.
    qa=quantize_mxfp8(x.reshape(-1,33)); reference=dequantize_mxfp8(qa)@dequantize_mxfp8(q).T
    if bias: reference.add_(layer.bias)
    expected=reference.to(dtype).reshape(2,3,5)
    calls=[]
    import fasth3_pipeline.mxfp8_linear as module
    original=module.quantize_mxfp8
    def bounded(value):
        calls.append(value.shape[0]); assert value.shape[0]<=2
        return original(value)
    monkeypatch.setattr(module,'quantize_mxfp8',bounded)
    torch.testing.assert_close(layer(x),expected,atol=1e-5 if dtype==torch.float32 else 0,rtol=1e-6 if dtype==torch.float32 else 0)
    assert calls==[2,2,2]
    assert layer(torch.empty(0,33,dtype=dtype)).shape==(0,5)


def test_linear_native_default_full_n_workspace_and_launch_count_meta_only():
    layer=MXFP8Linear(5376,28672,device='meta')
    assert layer.chunk_tokens==128 and layer.chunk_out_features==28672
    token_tiles=(38536+127)//128
    assert token_tiles==302
    assert token_tiles*((28672+layer.chunk_out_features-1)//layer.chunk_out_features)==302
    assert token_tiles*((28672+127)//128)==67648  # previous wrapper count
    assert layer.workspace_upper_bound()==64*128*5376
    assert MXFP8Linear(14336,28672,device='meta').workspace_upper_bound()==64*128*14336
    optimized=MXFP8Linear(14336,5376,device='meta',backend='cute-optimized',chunk_tokens=512)
    assert optimized.eager_chunk_tokens<optimized.chunk_tokens
    assert optimized.workspace_upper_bound()<=256<<20
    # Budgeted epilogue tiling still applies, WITHOUT fake FP64 weight bytes.
    limited=MXFP8Linear(32,10000,device='meta',max_working_bytes=1<<20)
    assert limited.chunk_out_features<10000
    assert limited.workspace_upper_bound()<=1<<20


def test_linear_validates_weights_once_and_full_n_gemm_once_per_token_chunk(monkeypatch):
    import fasth3_pipeline.mxfp8_linear as module
    layer=MXFP8Linear(33,5,bias=False,backend='oracle',chunk_tokens=2)
    q=quantize_mxfp8(torch.randn(5,33)); layer.weight=q.values; layer.weight_scale=q.scales
    checks=[]; gemms=[]
    original_check=layer._validate_weight_once; original_gemm=module._oracle_validated_gemm
    def check(value): checks.append(True); return original_check(value)
    def gemm(a,b): gemms.append((a.values.shape[0],b.values.shape[0])); return original_gemm(a,b)
    monkeypatch.setattr(layer,'_validate_weight_once',check)
    monkeypatch.setattr(module,'_oracle_validated_gemm',gemm)
    # Would fail if the native-like trusted path accidentally calls the public
    # validation wrapper again per tile (no weakening external kernel APIs).
    monkeypatch.setattr(MXFP8Tensor,'validate',lambda self: (_ for _ in ()).throw(AssertionError('duplicate wrapper validation')))
    layer(torch.randn(7,33))
    assert checks==[True] and gemms==[(2,5),(2,5),(2,5),(1,5)]


@pytest.mark.parametrize('corruption',['positive_nan','negative_nan','scale','padding'])
def test_linear_revalidates_mutated_streamed_buffers_each_forward(corruption):
    layer=MXFP8Linear(33,5,bias=False,chunk_tokens=2)
    q=quantize_mxfp8(torch.randn(5,33)); layer.weight=q.values; layer.weight_scale=q.scales
    layer.validation_rows=2  # corruption in the final ragged validation chunk
    x=torch.ones(1,33)
    layer.debug_checks=True
    layer._validate_weight_once(x)
    if corruption=='scale': layer.weight_scale[-1,-1]=255
    else:
        raw=layer.weight.view(torch.uint8)
        raw[-1,34 if corruption=='padding' else 0]={'positive_nan':127,'negative_nan':255,'padding':1}[corruption]
    with pytest.raises(ValueError,match='invalid MXFP8 buffers'): layer._validate_weight_once(x)
    layer.debug_checks=False
    layer._validate_weight_once(x)


@pytest.mark.parametrize('bad',['shape','dtype','noncontiguous'])
def test_linear_strict_weight_metadata(bad):
    layer=MXFP8Linear(33,5,bias=False,backend='oracle')
    q=quantize_mxfp8(torch.ones(5,33)); layer.weight=q.values; layer.weight_scale=q.scales
    if bad=='shape': layer.weight_scale=torch.empty(5,1,dtype=torch.uint8)
    if bad=='dtype': layer.weight_scale=torch.empty(5,2,dtype=torch.float32)
    if bad=='noncontiguous': layer.weight_scale=torch.ones(2,5,dtype=torch.uint8).T
    with pytest.raises(ValueError): layer(torch.ones(1,33))


def test_linear_meta_native_fail_closed_and_budget():
    layer=MXFP8Linear(33,5,device='meta')
    assert set(layer.state_dict())=={'weight','weight_scale','bias'}
    assert layer.weight.shape==(5,64) and layer.weight_scale.shape==(5,2)
    with pytest.raises(RuntimeError,match='requires CUDA'): layer(torch.ones(2,33))
    with pytest.raises(ValueError,match='working-byte'): MXFP8Linear(14336,5376,max_working_bytes=1)
    with pytest.raises(ValueError,match='backend'): MXFP8Linear(32,8,backend='bf16')
    with pytest.raises(ValueError): MXFP8Linear(32,8,chunk_tokens=0)
