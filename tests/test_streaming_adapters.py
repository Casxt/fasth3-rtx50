import copy
import pytest
import torch
from torch import nn
from fasth3_pipeline.streaming import CPUWeightStore, ManagedStream
from fasth3_pipeline.offload import BlockStreamedWeights


def test_cpu_vae_store_keeps_existing_storage_and_exact_partial_bytes():
    model=nn.Linear(3,2).bfloat16()
    store=CPUWeightStore(model)
    assert store.tensor('weight').data_ptr()==model.weight.data_ptr()
    raw=model.weight.detach().reshape(-1).view(torch.uint8)
    destination=torch.empty(5,dtype=torch.uint8)
    assert store.read_into('weight',destination,byte_offset=1)==5
    assert torch.equal(destination,raw[1:6])
    with pytest.raises(ValueError):store.read_into('weight',destination,byte_offset=raw.numel())
    with pytest.raises(ValueError):store.read_into('weight',destination.float())
    store.close()
    with pytest.raises(RuntimeError):store.tensor('weight')


def test_cpu_store_does_not_silently_materialize_noncontiguous_model():
    model=nn.Linear(3,3)
    model.weight=nn.Parameter(model.weight.T)
    with pytest.raises(ValueError,match='contiguous'):CPUWeightStore(model)
    with torch.device('meta'):model=nn.Linear(3,3)
    with pytest.raises(ValueError,match='CPU'):CPUWeightStore(model)


def test_managed_stream_closes_consumers_before_owned_source():
    order=[]
    class Stream:
        def close(self):order.append('consumer')
    class Store:
        def close(self):order.append('source')
    handle=ManagedStream(Stream(),Store(),'leaf')
    handle.close();handle.close()
    assert order==['consumer','source'] and handle.closed


def test_managed_stream_releases_source_even_on_consumer_failure():
    class Stream:
        def close(self):raise OSError('cleanup failure')
    class Store:
        def close(self):self.closed=True
    source=Store();handle=ManagedStream(Stream(),source,'block')
    with pytest.raises(OSError,match='cleanup failure'):handle.close()
    assert source.closed


def test_cpu_source_group_composition_matches_loaded_model_without_clone():
    model=nn.Sequential(nn.Linear(3,3),nn.SiLU(),nn.Linear(3,2)).double()
    reference=copy.deepcopy(model)
    source=CPUWeightStore(model)
    stream=BlockStreamedWeights(model,source,'cpu',groups=['0','2'],prefetch_depth=1)
    handle=ManagedStream(stream,source,'prefetch')
    for _ in range(3):
        x=torch.randn(2,3,dtype=torch.float64)
        with torch.no_grad():torch.testing.assert_close(model(x),reference(x),atol=0,rtol=0)
    handle.close()
    assert all(p.is_meta for p in model.parameters())
    stats=handle.stats_snapshot()
    assert stats['storage']['closed'] and stats['stream']['read_into_bytes']>0


@pytest.mark.parametrize('mode', ['cached', 'fixed'])
def test_runner_vae_context_uses_grouped_cpu_store_and_safe_device_hint(mode):
    from types import SimpleNamespace
    from diffusers.utils.accelerate_utils import apply_forward_hook
    from fasth3_pipeline.runner import _vae_execution
    from fasth3_pipeline.diagnostics import PipelineDiagnostics
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(nn.utils.weight_norm(nn.Linear(3,3))).double()
            self.gain=nn.Parameter(torch.tensor(2.,dtype=torch.float64))
        @apply_forward_hook
        def encode(self,x):return self.encoder(x)*self.gain
        def enable_tiling(self):self.tiling=True
    pipe=SimpleNamespace(vae=VAE(),audio_vae=VAE())
    x=torch.randn(2,3,dtype=torch.float64)
    expected=[pipe.vae.encode(x).detach(),pipe.audio_vae.encode(x).detach()]
    diag=PipelineDiagnostics()
    checks = []
    with _vae_execution(pipe,'cpu',checks.append,mode,diag,'test'):
        torch.testing.assert_close(pipe.vae.encode(x),expected[0],atol=0,rtol=0)
        torch.testing.assert_close(pipe.audio_vae.encode(x),expected[1],atol=0,rtol=0)
        assert pipe.vae._hf_hook.execution_device==torch.device('cpu')
    assert all(p.is_meta for p in pipe.vae.parameters())
    assert diag.streaming['test']['vae']['storage']['closed']
    if mode == 'fixed':
        assert not any(phase.startswith('block tensor:') for phase in checks)
        assert any(phase.startswith('block load:') for phase in checks)
        assert diag.streaming['test']['vae']['stream']['prefetch_loads'] == 0


def test_runner_resources_close_on_early_return_and_release_closed_handles():
    import gc
    import weakref
    from fasth3_pipeline.runner import _managed_run,_own_stream
    order=[]
    class Resource:
        def __init__(self,name):self.name=name
        def close(self):order.append(self.name)
    @_managed_run
    def operation():
        handle=_own_stream(ManagedStream(Resource('consumer'),Resource('source'),'block'))
        ref=weakref.ref(handle)
        handle.close()
        del handle
        gc.collect()
        assert ref() is None  # Run stack must not retain a closed DiT through decode.
        other=_own_stream(ManagedStream(Resource('early-consumer'),Resource('early-source'),'block'))
        return 'early'
    assert operation()=='early'
    assert order==['consumer','source','early-consumer','early-source']


def test_managed_active_close_does_not_destroy_live_backing_store():
    model=nn.Sequential(nn.Linear(3,3))
    source=CPUWeightStore(model)
    inner=BlockStreamedWeights(model,source,'cpu',groups=['0'])
    handle=ManagedStream(inner,source,'block')
    with handle.execution():
        with pytest.raises(RuntimeError):handle.close()
        assert not source.closed and not handle.closed and not inner.closed
        assert inner.handles
        assert torch.isfinite(model(torch.ones(1,3))).all()
    handle.close()
    assert handle.closed and source.closed and inner.closed and not inner.handles


def test_early_run_cleanup_records_stream_snapshot():
    from fasth3_pipeline.runner import _managed_run,_own_stream
    from fasth3_pipeline.diagnostics import PipelineDiagnostics
    diag=PipelineDiagnostics()
    @_managed_run
    def early():
        model=nn.Sequential(nn.Linear(2,2))
        source=CPUWeightStore(model)
        handle=_own_stream(ManagedStream(BlockStreamedWeights(model,source,'cpu',groups=['0']),source,'prefetch'),diag)
        assert handle.prime_first()
        return 'stopped-before-dit'
    assert early()=='stopped-before-dit'
    assert diag.streaming['dit']['closed']
    assert diag.streaming['dit']['storage']['closed']
    assert diag.streaming['dit']['stream']['resident_bytes']==0
