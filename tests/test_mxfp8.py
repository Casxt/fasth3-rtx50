"""CPU MXFP8 format contracts; native compilation is Lead-run via probe tool."""
import ast
import os
from pathlib import Path
import pytest
import torch
from fasth3_kernels.mxfp8 import (
    MXFP8Tensor, decode_e8m0, quantize_mxfp8, dequantize_mxfp8,
    quantize_mxfp8_scalar_oracle, quantize_mxfp8_cuda, mxfp8_gemm,
)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('k',[1,31,32,33,65])
def test_scalar_codebook_exact(dtype,k):
    torch.manual_seed(k)
    x=torch.randn(2,k).to(dtype)
    actual=quantize_mxfp8(x); expected=quantize_mxfp8_scalar_oracle(x)
    assert torch.equal(actual.scales,expected.scales)
    assert torch.equal(actual.values.view(torch.uint8),expected.values.view(torch.uint8))
    actual.validate()
    assert dequantize_mxfp8(actual).shape==x.shape


def test_zero_outlier_and_power_boundaries():
    x=torch.zeros(4,65)
    x[1,0]=448; x[2,0]=449; x[3,0]=torch.finfo(torch.float32).max
    q=quantize_mxfp8(x)
    assert q.scales[0].tolist()==[127,127,127]
    assert q.scales[1,0]==127 and q.scales[2,0]==128
    assert q.scales[3,0] <= 254
    # A maximal FP32 value rounds to decoded 2**128: finite MX value, outside
    # FP32 range. Preserve that fact rather than silently saturating decode.
    assert torch.isfinite(dequantize_mxfp8(q,dtype=torch.float64)).all()
    assert dequantize_mxfp8(q,dtype=torch.float64)[3,0] == 2.**128
    assert torch.isinf(dequantize_mxfp8(q)[3,0])
    assert torch.equal(q.values.float()[:,65:],torch.zeros(4,31))


def test_smallest_subnormal_and_signed_zero():
    x=torch.tensor([[0.,-0.,2.**-149,-2.**-149]+[0.]*28])
    actual=quantize_mxfp8(x); expected=quantize_mxfp8_scalar_oracle(x)
    assert torch.equal(actual.scales,expected.scales)
    assert torch.equal(actual.values.view(torch.uint8),expected.values.view(torch.uint8))
    assert actual.scales.item()==0


def test_nearest_even_ties_with_fixed_unit_scale():
    x=torch.zeros(1,32)
    x[0,:7]=torch.tensor([448.,1.0625,1.1875,.5/512.,1.5/512.,-1.0625,-1.1875])
    q=quantize_mxfp8(x)
    assert q.scales.item()==127
    assert q.values.float()[0,:7].tolist()==[448.,1.,1.25,0.,2./512.,-1.,-1.25]
    assert torch.equal(q.values.view(torch.uint8),quantize_mxfp8_scalar_oracle(x).values.view(torch.uint8))


def test_e8m0_extremes_and_reserved():
    assert decode_e8m0(torch.tensor([0,127,254],dtype=torch.uint8)).tolist()==[2.**-127,1.,2.**127]
    with pytest.raises(ValueError,match='E8M0'):
        decode_e8m0(torch.tensor([255],dtype=torch.uint8))


@pytest.mark.parametrize('k',[5376,14336])
def test_h3_k_and_noncontiguous(k):
    torch.manual_seed(k)
    x=torch.randn(k,3).T
    assert not x.is_contiguous()
    q=quantize_mxfp8(x)
    q.validate()
    assert q.scales.shape==(3,k//32)
    error=(dequantize_mxfp8(q)-x).norm()/x.norm()
    assert error < .04


@pytest.mark.parametrize('bad',[float('inf'),float('nan')])
def test_nonfinite_input_rejected(bad):
    with pytest.raises(ValueError,match='nonfinite'):
        quantize_mxfp8(torch.tensor([[bad]]))


def test_bad_geometry_dtype_scale_and_padding_rejected():
    q=quantize_mxfp8(torch.randn(2,33))
    with pytest.raises(ValueError,match='reserved'):
        MXFP8Tensor(q.values,torch.full_like(q.scales,255),33).validate()
    broken=q.values.float(); broken[:,34]=1
    with pytest.raises(ValueError,match='padding'):
        MXFP8Tensor(broken.to(q.values.dtype),q.scales,33).validate()
    with pytest.raises(ValueError): quantize_mxfp8(torch.empty(0,32))
    with pytest.raises(ValueError): quantize_mxfp8(q.values)
    with pytest.raises(ValueError): MXFP8Tensor(q.values,q.scales,1).validate()


def test_explicit_oracle_and_no_silent_native_fallback():
    a=quantize_mxfp8(torch.randn(17,65)); b=quantize_mxfp8(torch.randn(9,65))
    expected=dequantize_mxfp8(a)@dequantize_mxfp8(b).T
    torch.testing.assert_close(mxfp8_gemm(a,b,backend='oracle'),expected,atol=0,rtol=0)
    with pytest.raises(ValueError,match='CUDA'): mxfp8_gemm(a,b)
    with pytest.raises(ValueError,match='backend'): mxfp8_gemm(a,b,backend='bf16')


def _power_boundary_cases(dtype):
    powers = [-25,-16,-9,-8,-1,0,6] if dtype==torch.float16 else [-132,-120,-9,-8,-1,0,1,64,118]
    values=[]; expected=[]
    for exponent in powers:
        exact=torch.tensor(448.*2.**exponent,dtype=dtype)
        previous=torch.nextafter(exact,torch.tensor(0.,dtype=dtype))
        following=torch.nextafter(exact,torch.tensor(float('inf'),dtype=dtype))
        values.extend([previous,exact,following])
        expected.extend([max(0,min(254,exponent+127)),max(0,min(254,exponent+127)),
                         max(0,min(254,exponent+128))])
    host=torch.zeros(len(values),33,dtype=dtype)
    for row,value in enumerate(values): host[row,0]=value if row%2 else -value
    return host,torch.tensor(expected,dtype=torch.uint8)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_power_of_two_448_boundaries_and_neighbors_cpu(dtype):
    host,expected=_power_boundary_cases(dtype)
    q=quantize_mxfp8(host); scalar=quantize_mxfp8_scalar_oracle(host)
    assert torch.equal(q.scales[:,0],expected)
    assert torch.equal(q.scales[:,1],torch.full_like(expected,127))
    assert torch.equal(q.scales,scalar.scales)
    assert torch.equal(q.values.view(torch.uint8),scalar.values.view(torch.uint8))


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA')!='1',reason='Lead-coordinated GPU only')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_power_of_two_448_boundaries_and_neighbors_cuda(dtype):
    from fasth3_platform import configure_gpu_budget,check_gpu_headroom
    budget=configure_gpu_budget(max_gib=10.,reserve_mib=256); check_gpu_headroom(budget)
    host,expected=_power_boundary_cases(dtype)
    cpu=quantize_mxfp8(host); q=quantize_mxfp8_cuda(host.cuda())
    assert torch.equal(q.scales[:,0].cpu(),expected)
    assert torch.equal(q.scales.cpu(),cpu.scales)
    assert torch.equal(q.values.view(torch.uint8).cpu(),cpu.values.view(torch.uint8))


def test_explicit_cuda_quantizer_rejects_cpu_without_initializing_cuda():
    with pytest.raises(ValueError,match='CUDA-resident'):
        quantize_mxfp8_cuda(torch.ones(2,33))


def test_resident_quantizer_has_no_bulk_host_transfer_or_scalar_reads():
    path=Path(__file__).resolve().parents[1]/'fasth3_kernels/mxfp8.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    resident=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                  and n.name=='_quantize_mxfp8_resident')
    calls=[n for n in ast.walk(resident) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in
                   ('cpu','numpy','item','tolist','to_global') for n in calls)
    assert not any(ast.unparse(n.func) in ('bool','decode_e8m0','torch.log2','torch.ceil') for n in calls)
    assert any(ast.unparse(n.func)=='torch.frexp' for n in calls)
    wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                 and n.name=='quantize_mxfp8_cuda')
    assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='bool' for n in ast.walk(wrapper))==1


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA')!='1',reason='Lead-coordinated GPU only')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('k',[33,65,5376,14336])
def test_cuda_resident_quantization_codes_scales_and_stream(dtype,k):
    from fasth3_platform import configure_gpu_budget,check_gpu_headroom
    budget=configure_gpu_budget(max_gib=10.,reserve_mib=256); check_gpu_headroom(budget)
    torch.manual_seed(91+k)
    # Noncontiguous input exercises device-side pad/materialization. First row
    # all-zero, second has a power-boundary outlier, third ordinary varied data.
    host=torch.randn(k,3,dtype=dtype).T
    host[0]=0; host[1,0]=449
    qcpu=quantize_mxfp8(host)
    resident=host.T.contiguous().cuda().T
    assert not resident.is_contiguous()
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qgpu=quantize_mxfp8_cuda(resident)
    stream.synchronize()
    assert qgpu.values.device==resident.device and qgpu.scales.device==resident.device
    assert torch.equal(qgpu.scales.cpu(),qcpu.scales)
    assert torch.equal(qgpu.values.view(torch.uint8).cpu(),qcpu.values.view(torch.uint8))
    # Independently enumerated E4M3 codebook for bounded prefix only.
    scalar=quantize_mxfp8_scalar_oracle(host[:,:min(k,65)])
    prefix=quantize_mxfp8_cuda(resident[:,:min(k,65)])
    assert torch.equal(prefix.scales.cpu(),scalar.scales)
    assert torch.equal(prefix.values.view(torch.uint8).cpu(),scalar.values.view(torch.uint8))


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA')!='1',reason='Lead-coordinated GPU only')
def test_cuda_quantizer_rejects_nonfinite():
    from fasth3_platform import configure_gpu_budget,check_gpu_headroom
    budget=configure_gpu_budget(max_gib=10.,reserve_mib=256); check_gpu_headroom(budget)
    for bad in (float('inf'),float('-inf'),float('nan')):
        with pytest.raises(ValueError,match='nonfinite'):
            quantize_mxfp8_cuda(torch.tensor([[bad]],device='cuda'))


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA')!='1',reason='Lead-coordinated GPU only')
def test_tiled_native_and_fused_activation():
    from fasth3_kernels._cute_mxfp8 import launch
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
    from fasth3_platform import BudgetGuard

    with BudgetGuard(), torch.no_grad():
        generator = torch.Generator().manual_seed(73)
        host = torch.randn(35, 161, generator=generator, dtype=torch.bfloat16)
        # Tiny subnormals, exact ceil threshold, signed zero and K padding.
        host[0, :32] = 0
        host[0, :4] = torch.tensor([2.**-133, -(2.**-133), 0., -0.])
        host[1, :32] = 1.75
        host[1, 32:64] = 1.765625
        resident = host.T.contiguous().cuda().T
        expected = quantize_mxfp8(host)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            a = quantize_mxfp8_cute(resident)
            b = quantize_mxfp8(torch.randn(41, 161, generator=generator).cuda())
            result = mxfp8_gemm(a, b, backend='cute-tiled')
            scalar = mxfp8_gemm(a, b)
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.equal(a.scales.cpu(), expected.scales)
        assert torch.equal(a.values.view(torch.uint8).cpu(), expected.values.view(torch.uint8))
        torch.testing.assert_close(result, scalar, rtol=0, atol=0)
        bias = torch.randn(result.shape[1], device=result.device)
        out = torch.empty_like(result, dtype=torch.bfloat16)
        returned = launch(a, b, optimized=(128, 128, 2, 2, 4), bias=bias, out=out)
        assert returned is out
        torch.testing.assert_close(out, (scalar + bias).bfloat16(), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA')!='1',reason='Lead-coordinated GPU only')
def test_packed_scales_match_row_major_with_output_chunking():
    from fasth3_pipeline.mxfp8_linear import MXFP8Linear
    from fasth3_platform import BudgetGuard

    with BudgetGuard(), torch.no_grad():
        q=quantize_mxfp8(torch.randn(96,256,device='cuda',dtype=torch.bfloat16))
        x=torch.randn(129,256,device='cuda',dtype=torch.bfloat16)
        normal=MXFP8Linear(256,96,bias=False,device='cuda',backend='cute-optimized',chunk_tokens=512)
        normal.weight,normal.weight_scale=q.values,q.scales
        packed=MXFP8Linear(256,96,bias=False,device='cuda',backend='cute-optimized',chunk_tokens=512,
                          chunk_out_features=64,packed_scales=True)
        packed.weight=q.values
        packed.weight_scale=q.scales.view(3,32,2,4).permute(0,2,1,3).contiguous()
        torch.testing.assert_close(normal(x),packed(x),rtol=0,atol=0)


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA')!='1',reason='Lead-coordinated GPU only')
@pytest.mark.parametrize('tma,producers,stages', [(False,2,2),(True,1,3)])
def test_mxfp8_warp_specialization_wraps_with_tails(tma,producers,stages):
    from fasth3_kernels._cute_mxfp8 import launch
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
    from fasth3_platform import BudgetGuard

    with BudgetGuard(), torch.no_grad():
        a=quantize_mxfp8_cute(torch.randn(137,640,device='cuda',dtype=torch.bfloat16))
        b=quantize_mxfp8_cute(torch.randn(160,640,device='cuda',dtype=torch.bfloat16))
        scales=b.scales.view(5,32,5,4).permute(0,2,1,3).contiguous()
        b=MXFP8Tensor(b.values,scales,640)
        reference=launch(a,b,optimized=(64,128,2,2,4),packed_b=True)
        actual=launch(a,b,optimized=(64,128,stages,2,4),packed_b=True,
                      producer_warps=producers,tma=tma)
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)


@pytest.mark.skipif(os.environ.get('FASTH3_TEST_CUDA') != '1', reason='Coordinated GPU only')
def test_tma_output_store_matches_direct_bf16_with_bias():
    from fasth3_kernels._cute_mxfp8 import launch
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
    from fasth3_platform import BudgetGuard

    with BudgetGuard(), torch.no_grad():
        a = quantize_mxfp8_cute(torch.randn(128, 256, device='cuda', dtype=torch.bfloat16))
        b = quantize_mxfp8_cute(torch.randn(256, 256, device='cuda', dtype=torch.bfloat16))
        b = MXFP8Tensor(b.values, b.scales.view(8, 32, 2, 4).permute(0, 2, 1, 3).contiguous(), 256)
        bias = torch.randn(256, device='cuda', dtype=torch.bfloat16)
        result = torch.empty((128, 256), device='cuda', dtype=torch.bfloat16)
        options = dict(optimized=(64, 128, 2, 2, 2), packed_b=True, tma=True, producer_warps=1, bias=bias)
        expected = launch(a, b, out=result, **options).clone()
        actual = launch(a, b, out=result, tma_store=True, **options)
        assert torch.equal(actual, expected)


def test_native_source_uses_sm120_warp_api_not_emulated_gemm():
    path=Path(__file__).resolve().parents[1]/'fasth3_kernels/_cute_mxfp8.py'
    source=path.read_text(encoding='utf-8'); tree=ast.parse(source)
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(any(x in module for x in ('flashinfer','triton','fastvideo')) for module in imports)
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'warp.MmaMXF8Op' in calls and 'cute.gemm' in calls
    assert not any('tcgen05' in call or 'scaled_mm' in call or call.endswith('linear') for call in calls)
    # Coordinate tensor layouts have tuple-valued ScaledBasis codomains.
    # The installed *_like API special-cases coordinate tensors into scalar
    # compact fragments while ca/cb/cc continue to map domain indices to rows.
    scalar=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_kernel')
    register_calls=[n for n in ast.walk(scalar) if isinstance(n,ast.Call)
                    and ast.unparse(n.func)=='cute.make_rmem_tensor_like']
    assert [ast.unparse(n.args[0]) for n in register_calls]==['ca','cb','cc']
    assert not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='cute.make_rmem_tensor'
                   and n.args and isinstance(n.args[0],ast.Attribute)
                   and ast.unparse(n.args[0]) in ('ca.layout','cb.layout','cc.layout')
                   for n in ast.walk(tree))
