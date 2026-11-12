"""CPU simulation of real block execution contracts; no CUDA initialization."""
import copy
import os
import pytest
import torch
from torch import nn
from fasth3_pipeline.offload import BlockStreamedWeights, WeightGroup, dit_groups, vae_groups


class MemoryStore:
    def __init__(self, model):
        self.values = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.reads = []
        self.fail = None
        self.closed = False
    def validate_keys(self, model):
        assert set(model.state_dict()) == set(self.values)
        assert all(tuple(t.shape) == tuple(self.values[k].shape) for k, t in model.state_dict().items())
    def tensor(self, key):
        self.reads.append(key)
        if key == self.fail:
            raise OSError("simulated store failure")
        return self.values[key]
    def shape(self, key):
        return self.values[key].shape
    def dtype(self, key):
        return self.values[key].dtype


class ReadIntoStore(MemoryStore):
    def __init__(self, model):
        super().__init__(model)
        self.copies = []
    def tensor(self, key):
        pytest.fail("read_into store must not materialize a full CPU tensor")
    def read_into(self, key, destination, *, byte_offset=0):
        assert destination.device.type == "cpu" and destination.dtype == torch.uint8
        assert destination.is_contiguous()
        source = self.values[key].reshape(-1).view(torch.uint8)
        size = destination.numel()
        assert 0 <= byte_offset <= byte_offset+size <= source.numel()
        self.copies.append((key, byte_offset, size))
        if key == self.fail:
            raise OSError("simulated read_into failure")
        destination.copy_(source[byte_offset:byte_offset+size])
        return size


def all_meta(model):
    return all(v.is_meta for v in model.state_dict().values())


def test_resident_cache_reuses_a_fixed_hot_group_and_releases_at_end():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    reference = copy.deepcopy(model)
    store = MemoryStore(model)
    first_size = sum(t.nbytes for t in model[0].state_dict().values())
    stream = BlockStreamedWeights(model, store, 'cpu', groups=['0', '1'],
                                  resident_cache_bytes=first_size)
    x = torch.randn(2, 3)
    try:
        with stream.execution():
            for _ in range(3):
                actual = model[1](model[0](x))
                torch.testing.assert_close(actual, reference(x), atol=0, rtol=0)
            stats = stream.stats_snapshot()
            assert stats['cache_hits'] == 2
            assert stats['cache_peak_bytes'] == first_size
            assert stats['group_loads'] == 4
        assert all_meta(model)
        assert stream.stats_snapshot()['cached_bytes'] == 0
    finally:
        stream.close()


class Siblings(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3)
        self.b = nn.Linear(3, 3)
        self.c = nn.Linear(3, 3)
        self.fail = False
    def forward(self, x):
        x = self.a(x)
        # Reading sibling weight directly after a returns must remain valid.
        x = x + x @ self.b.weight.T + self.b.bias
        if self.fail:
            raise ValueError("model failure")
        return self.c(x)


@pytest.mark.parametrize("prefetch", [0, 1])
def test_sibling_group_eight_calls_exact_and_no_cuda(prefetch, monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: pytest.fail("CPU initialized CUDA"))
    model = Siblings().eval()
    reference = copy.deepcopy(model)
    store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("input", ("a", "b")), "c"], prefetch_depth=prefetch)
    x = torch.randn(2, 3)
    for _ in range(8):
        torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
        assert all_meta(model)
        assert stream.stats_snapshot()["resident_bytes"] == 0
    stats = stream.stats_snapshot()
    assert stats["runs"] == 8 and stats["group_loads"] == 16
    assert stats["pinned_peak_bytes"] == stats["bytes_h2d"] == 0
    assert stats["prefetch_hits"] == (8 if prefetch else 0)
    stream.close(); stream.close()
    assert not store.closed


@pytest.mark.parametrize("prefetch", [0, 1])
def test_read_into_priority_odd_cpu_chunks_eight_calls_and_separate_metrics(prefetch):
    model = Siblings(); reference = copy.deepcopy(model); store = ReadIntoStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"],
                                  prefetch_depth=prefetch, max_pinned_bytes=7)
    for _ in range(8):
        torch.testing.assert_close(model(torch.ones(2, 3)), reference(torch.ones(2, 3)), atol=0, rtol=0)
        assert all_meta(model)
    stats = stream.stats_snapshot()
    assert stats["read_into_bytes"] == stats["bytes_read"] > 0
    assert stats["read_into_calls"] == len(store.copies)
    assert stats["store_read_seconds"] == stats["pinned_copy_seconds"] == 0
    assert stats["read_into_seconds"] >= 0
    assert stats["pinned_peak_bytes"] == stats["bytes_h2d"] == 0
    for key, tensor in store.values.items():
        size = tensor.numel()*tensor.element_size()
        expected = [(key, i, min(7, size-i)) for i in range(0, size, 7)]
        assert [entry for entry in store.copies if entry[0] == key] == expected*8
    stream.close()


@pytest.mark.parametrize("backend", ["safe_open", "readonly_mmap"])
def test_persistent_store_read_into_integration(tmp_path, backend):
    import json
    from safetensors.torch import save_file
    from fasth3_pipeline.storage import PersistentTensorStore
    model = Siblings(); reference = copy.deepcopy(model)
    state = {key: value.detach().contiguous() for key, value in model.state_dict().items()}
    save_file(state, tmp_path / "weights.safetensors")
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors" for key in state}}), encoding="utf-8")
    store = PersistentTensorStore(tmp_path, backend=backend, max_open_shards=1)
    store.tensor = lambda key: pytest.fail("M2 may not call tensor() when read_into is available")
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"],
                                  prefetch_depth=1, max_pinned_bytes=7)
    try:
        for _ in range(8):
            torch.testing.assert_close(model(torch.ones(1, 3)), reference(torch.ones(1, 3)), atol=0, rtol=0)
            assert all_meta(model)
        assert store.stats_snapshot()["read_into_bytes"] == stream.stats_snapshot()["read_into_bytes"]
        assert store.stats_snapshot()["tensor_requests"] == 0
    finally:
        stream.close(); store.close()


@pytest.mark.parametrize("bad", [None, True, -1, 0, 1.0, 100000])
def test_read_into_short_or_invalid_count_aborts_and_unloads(bad):
    model = Siblings(); store = ReadIntoStore(model)
    original = store.read_into
    def read_into(key, destination, *, byte_offset=0):
        original(key, destination, byte_offset=byte_offset)
        return bad
    store.read_into = read_into
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"], max_pinned_bytes=7)
    with pytest.raises(IOError, match="exact integer byte count"):
        model(torch.ones(1, 3))
    assert all_meta(model) and not stream.running
    assert stream.stats_snapshot()["resident_bytes"] == 0
    stream.close()


def test_read_into_prefetch_exception_cancels_and_can_retry():
    model = Siblings(); store = ReadIntoStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"], prefetch_depth=1)
    store.fail = "c.bias"
    with pytest.raises(OSError, match="read_into failure"):
        model(torch.ones(1, 3))
    assert all_meta(model) and not stream.running
    store.fail = None
    assert torch.isfinite(model(torch.ones(1, 3))).all()
    stream.close()


def test_failure_unloads_partial_groups_and_next_call_recovers():
    model = Siblings()
    store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"], prefetch_depth=1)
    store.fail = "b.bias"
    with pytest.raises(OSError, match="store failure"):
        model(torch.ones(1, 3))
    assert all_meta(model) and stream.stats_snapshot()["resident_bytes"] == 0
    store.fail = None
    model.fail = True
    with pytest.raises(ValueError, match="model failure"):
        model(torch.ones(1, 3))
    assert all_meta(model) and not stream.running
    model.fail = False
    assert torch.isfinite(model(torch.ones(1, 3))).all()
    stream.close()


def test_prefetch_store_failure_cancels_current_and_next_groups():
    model = Siblings(); store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"], prefetch_depth=1)
    store.fail = "c.bias"
    with pytest.raises(OSError, match="store failure"):
        model(torch.ones(1, 3))
    assert all_meta(model) and stream.stats_snapshot()["resident_bytes"] == 0
    assert not stream.running
    stream.close()


@pytest.mark.parametrize("error_type", [MemoryError, torch.OutOfMemoryError, RuntimeError])
def test_prefetch_errors_propagate_unchanged_instead_of_becoming_skips(error_type):
    model = Siblings(); store = MemoryStore(model)
    original = store.tensor
    failure = error_type("injected transfer error")
    def tensor(key):
        if key == "c.bias":
            raise failure
        return original(key)
    store.tensor = tensor
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"], prefetch_depth=1)
    with pytest.raises(error_type) as caught:
        model(torch.ones(1, 3))
    assert caught.value is failure
    assert all_meta(model) and stream.stats_snapshot()["resident_bytes"] == 0
    assert not stream.running
    stream.close()


def test_autograd_rejected_before_weight_execution():
    model = Siblings(); store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"])
    with pytest.raises(RuntimeError, match="inference-only"):
        model(torch.ones(1, 3, requires_grad=True))
    assert all_meta(model) and not stream.running
    with torch.no_grad():
        assert torch.isfinite(model(torch.ones(1, 3, requires_grad=True))).all()
    stream.close()


def test_guard_failure_and_budget_bound_are_fail_closed():
    model = Siblings()
    store = MemoryStore(model)
    calls = []
    def guard(phase):
        calls.append(phase)
        if phase == "block tensor:b.bias":
            raise MemoryError("injected budget refusal")
    stream = BlockStreamedWeights(model, store, "cpu", guard, groups=[WeightGroup("all", ("",))])
    with pytest.raises(MemoryError, match="budget refusal"):
        model(torch.ones(1, 3))
    assert all_meta(model) and not stream.running
    stream.close()
    with pytest.raises(MemoryError, match="group exceeds"):
        BlockStreamedWeights(Siblings(), store, "cpu", groups=[WeightGroup("all", ("",))], max_resident_bytes=1)


@pytest.mark.parametrize("prefetch", [0, 1])
@pytest.mark.parametrize("store_class", [MemoryStore, ReadIntoStore])
def test_alternate_root_persistent_parameter_and_buffer_live_across_children(prefetch, store_class):
    class Alternate(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(2., dtype=torch.float64))
            self.register_buffer("offset", torch.tensor(.125, dtype=torch.float64))
            self.encoder = nn.Linear(3, 3).double()
            self.fail = False
        def encode(self, x):
            # The root persistent state is read before AND after child execution.
            y = self.encoder(x+self.offset)
            if self.fail:
                raise ValueError("alternate failure")
            return self.gain*y+self.offset
    model = Alternate(); reference = copy.deepcopy(model); store = store_class(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("root", ("",)), "encoder"], prefetch_depth=prefetch)
    x = torch.ones(2, 3, dtype=torch.float64)
    for _ in range(8):
        with stream.execution():
            torch.testing.assert_close(model.encode(x), reference.encode(x), atol=0, rtol=0)
            assert not model.gain.is_meta and not model.offset.is_meta
        assert all_meta(model) and stream.stats_snapshot()["resident_bytes"] == 0
    model.fail = True
    with pytest.raises(ValueError, match="alternate failure"):
        with stream.execution():
            model.encode(x)
    assert all_meta(model) and not stream.running
    store.fail = "gain"
    with pytest.raises(OSError, match="failure"):
        with stream.execution():
            pytest.fail("context body must not execute after failed root load")
    assert all_meta(model) and not stream.running
    stream.close()


@pytest.mark.parametrize("active,reserved,free,known,expected", [
    (9, 9, 16, True, False),   # Deferred frees count as active despite allocated=1.
    (2, 11, 16, True, False),  # Existing allocator reservations already over cap.
    (2, 9, 16, False, False),  # Unknown stats conservatively charge reservations.
    (2, 9, 16, True, True),
    (2, 9, 1, True, False),    # Driver free reserve cannot be spent as a cache.
])
def test_capacity_accounts_pending_allocator_blocks_without_cuda(monkeypatch, active, reserved, free, known, expected):
    from fasth3_pipeline.offload import GIB
    model = nn.Linear(1, 1)
    stream = BlockStreamedWeights(model, MemoryStore(model), "cpu", groups=[""])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free*GIB, 16*GIB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: GIB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: reserved*GIB)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {"active_bytes.all.current": active*GIB} if known else {})
    assert stream._cuda_capacity(2*GIB) is expected
    stream.close()


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2., dtype=torch.float64))
        self.a = nn.Linear(3, 3).double()
        self.b = nn.Linear(3, 3).double()
    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x*self.scale  # Parent state still needed AFTER both child groups.


def test_nested_parent_liveness_and_persistent_dtypes():
    model = Parent()
    reference = copy.deepcopy(model)
    store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("parent", ("",)), "a", "b"], prefetch_depth=1)
    assert model.scale.dtype == torch.float64
    for _ in range(2):
        x = torch.randn(2, 3, dtype=torch.float64)
        torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
        assert all_meta(model)
    stream.close()


def test_alternate_entry_context_nonpersistent_buffers_and_weightnorm():
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.utils.weight_norm(nn.Linear(3, 3)), nn.Linear(3, 3))
            self.register_buffer("constant", torch.ones(3), persistent=False)
        def encode(self, x):
            return self.encoder(x)+self.constant
    model = VAE()
    store = MemoryStore(model)
    x = torch.ones(1, 3)
    reference = model.encode(x).clone()
    stream = BlockStreamedWeights(model, store, "cpu", groups=vae_groups(model), prefetch_depth=1)
    with pytest.raises(RuntimeError, match="execution"):
        model.encode(x)
    for _ in range(2):
        with stream.execution():
            torch.testing.assert_close(model.encode(x), reference, atol=0, rtol=0)
        assert all_meta(model)
        assert model.encoder[0].weight.is_meta
    stream.close()


def test_direct_qkv_access_loaded_at_parent_boundary():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(3, 3)
        def forward(self, x):
            return nn.functional.linear(x, self.qkv.weight, self.qkv.bias)
    model = nn.Sequential(Attention(), nn.Linear(3, 3))
    reference = copy.deepcopy(model)
    store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=["0", "1"], prefetch_depth=1)
    torch.testing.assert_close(model(torch.ones(1, 3)), reference(torch.ones(1, 3)), atol=0, rtol=0)
    assert all_meta(model)
    stream.close()


def test_reentrant_forward_rejected_and_unloaded():
    class Reentrant(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.again = True
        def forward(self, x):
            if self.again:
                self.again = False
                return self(x)
            return self.linear(x)
    model = Reentrant()
    stream = BlockStreamedWeights(model, MemoryStore(model), "cpu", groups=["linear"])
    with pytest.raises(RuntimeError, match="Reentrant"):
        model(torch.ones(1, 3))
    assert all_meta(model) and not stream.running
    assert torch.isfinite(model(torch.ones(1, 3))).all()
    stream.close()


def test_real_tiny_diffusers_dit_dag_matches_eight_steps():
    from diffusers import MiniMaxH3Transformer3DModel
    from fasth3_pipeline.pdd import PDDProjection, PDDPlan
    from contextlib import ExitStack
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=8,
        hidden_size=16, num_layers=2, num_refiner_layers=1, ffn_dim=32,
        in_channels=2, audio_in_channels=2, patch_size=(1, 1, 1), text_dim=4,
        freq_dim=8, time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=1).eval()
    model.proj_out = PDDProjection(16, 2)
    model.audio_proj_out = PDDProjection(16, 2)
    reference = copy.deepcopy(model)
    store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=dit_groups(model), prefetch_depth=1)
    plan = PDDPlan.create()
    args = dict(hidden_states=torch.randn(1, 3, 2), audio_hidden_states=torch.randn(1, 2, 2),
                encoder_hidden_states=torch.randn(1, 2, 4), timestep=torch.tensor([.1]),
                timestep_indices=torch.zeros(7, dtype=torch.long), token_tags=torch.tensor([1,1,2,2,0,0,0]),
                position_ids=torch.randn(7, 3, dtype=torch.float64), video_indices=torch.tensor([4,5,6]),
                audio_indices=torch.tensor([2,3]), text_indices=torch.tensor([0,1]), return_dict=False)
    with torch.no_grad():
        for start, end in zip(plan.indices, plan.indices[1:]):
            with ExitStack() as stack:
                for target in (model, reference):
                    stack.enter_context(target.proj_out.fuse(start, end, plan.weights["video"]))
                    stack.enter_context(target.audio_proj_out.fuse(start, end, plan.weights["audio"]))
                actual, expected = model(**args), reference(**args)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert all_meta(model)
    assert stream.stats_snapshot()["runs"] == 8
    stream.close()


def test_real_tiny_video_vae_alternate_encode_decode_parent_groups():
    from diffusers import AutoencoderKLMiniMaxH3
    model = AutoencoderKLMiniMaxH3(latent_channels=2, block_out_channels=(4,),
        layers_per_block=1, spatial_downsample_factors=(2,), temporal_downsample_factors=(1,),
        norm_num_groups=1, decoder_num_layers=2, decoder_num_attention_heads=2,
        decoder_attention_head_dim=8, decoder_num_register_tokens=1,
        decoder_ffn_mult=2, clip_length=3, token_drop=0, latents_mean=(0., 0.), latents_std=(1., 1.)).eval()
    model.disable_tiling()
    reference = copy.deepcopy(model)
    store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=vae_groups(model), prefetch_depth=1)
    pixels, latent = torch.randn(1, 3, 3, 8, 8), torch.randn(1, 2, 3, 4, 4)
    with torch.no_grad():
        for _ in range(2):
            with stream.execution():
                actual = model.encode(pixels, return_dict=False)[0].mode()
            expected = reference.encode(pixels, return_dict=False)[0].mode()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert all_meta(model)
            with stream.execution():
                actual = model.decode(latent, return_dict=False)[0]
            expected = reference.decode(latent, return_dict=False)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert all_meta(model)
    stream.close()


def test_prefetch_skip_under_bound_and_wrong_prediction_stays_correct():
    model = Siblings(); reference = copy.deepcopy(model); store = MemoryStore(model)
    # Wrong order predicts c before a/b. Capacity fits only one sibling group.
    stream = BlockStreamedWeights(model, store, "cpu", groups=["c", WeightGroup("ab", ("a", "b"))],
                                  prefetch_depth=1, max_resident_bytes=96)
    torch.testing.assert_close(model(torch.ones(1, 3)), reference(torch.ones(1, 3)), atol=0, rtol=0)
    assert stream.stats_snapshot()["prefetch_skips"] > 0
    assert stream.stats_snapshot()["resident_peak_bytes"] <= 96
    stream.close()


def test_foreign_thread_is_rejected_without_corrupting_next_forward():
    from concurrent.futures import ThreadPoolExecutor
    model = Siblings(); store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("all", ("",))])
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="thread-affine"):
            pool.submit(model, torch.ones(1, 3)).result()
    assert all_meta(model) and not stream.running
    assert torch.isfinite(model(torch.ones(1, 3))).all()
    stream.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_read_into", [False, True])
def test_fake_dma_waits_before_reusing_pinned_slot_and_preserves_raw_bytes(monkeypatch, dtype, use_read_into):
    # Transfer engine contract test without a GPU or pinned allocator: CPU tensors
    # stand in for pin/device storage; event order, chunk bounds and bytes are real.
    model = nn.Linear(1, 1)
    store = ReadIntoStore(model) if use_read_into else MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[""], max_pinned_bytes=7)
    timeline = []
    class Event:
        def __init__(self, **kwargs):
            self.index = len(timeline)
        def record(self, unused):
            timeline.append(("record", self))
        def synchronize(self):
            timeline.append(("wait", self))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    stream.device = torch.device("cuda")
    stream.pin = torch.empty(7, dtype=torch.uint8)  # No pin allocator/CUDA initialization.
    source = torch.arange(12).reshape(3, 4).to(dtype)
    target = torch.empty_like(source)
    if use_read_into:
        store.values["payload"] = source
        stream._copy_from_store("payload", target)
        assert stream.stats["read_into_bytes"] == source.numel()*source.element_size()
    else:
        stream._copy(source, target)
    torch.testing.assert_close(target.float(), source.float(), atol=0, rtol=0)
    records = [event for kind, event in timeline if kind == "record"]
    waits = [event for kind, event in timeline if kind == "wait"]
    assert waits == records[1:-1:2]  # Every completed chunk is waited before next overwrite.
    assert stream.stats["bytes_h2d"] == source.numel()*source.element_size()
    assert stream.stats["h2d_chunks"] == (source.numel()*source.element_size()+6)//7
    # Reset simulated device before CPU cleanup; these fake events have no driver.
    stream.device = torch.device("cpu"); stream.pending_events.clear()
    stream.close()


@pytest.mark.skipif(os.environ.get("FASTH3_RUN_GPU_TESTS") != "1", reason="Lead-only GPU window")
@pytest.mark.parametrize("prefetch,store_class,readahead", [
    (0, MemoryStore, False), (1, MemoryStore, False),
    (0, ReadIntoStore, False), (1, ReadIntoStore, False), (0, ReadIntoStore, True),
])
def test_cuda_group_copy_stream_odd_byte_chunks_and_cancel(prefetch, store_class, readahead):
    from fasth3_platform import BudgetGuard
    with BudgetGuard(device="cuda:0", max_gib=10., reserve_mib=256) as guard:
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.Linear(16, 16)).bfloat16().eval()
        reference = copy.deepcopy(model).to("cuda:0")
        store = store_class(model)
        pin_bytes = 4 * 37 if readahead else 37
        stream = BlockStreamedWeights(model, store, "cuda:0", guard, groups=["0", "2", "3"],
                                      prefetch_depth=prefetch, max_pinned_bytes=pin_bytes,
                                      cpu_readahead=readahead)
        try:
            if readahead:
                def refuse_arena(phase):
                    assert phase == "block arena:0"
                    raise MemoryError("budget after arena allocation")

                stream.tensor_guard = refuse_arena
                assert not stream.prime_first()
                assert all_meta(model) and not store.copies
                assert stream.stats_snapshot()["resident_bytes"] == 0
                stream.tensor_guard = guard
            x = torch.randn(3, 16, device="cuda:0", dtype=torch.bfloat16)
            with torch.no_grad():
                for _ in range(8):
                    torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
                    assert all_meta(model)
            store.fail = "2.bias"
            with pytest.raises(OSError, match="failure"):
                model(x)
            assert all_meta(model) and not stream.running
            stats = stream.stats_snapshot()
            assert stats["resident_bytes"] == 0 and stats["pinned_peak_bytes"] == pin_bytes
            assert stats["bytes_h2d"] > 0 and stats["pending_cuda_timings"] == 0
        finally:
            stream.close()
        assert stream.stats_snapshot()["pinned_bytes"] == 0


@pytest.mark.parametrize("prefetch", [0, 1])
@pytest.mark.parametrize("store_class", [MemoryStore, ReadIntoStore])
@pytest.mark.parametrize("empty_setup", [False, True])
def test_prime_first_external_posthook_consumes_once_and_cleans_eight_forwards(prefetch, store_class, empty_setup):
    class Prepared(Siblings):
        def __init__(self):
            super().__init__(); self.setup = nn.Identity()
        def forward(self, x):
            return super().forward(self.setup(x))
    model = Prepared(); reference = copy.deepcopy(model); store = store_class(model)
    groups = (["setup"] if empty_setup else []) + [WeightGroup("input", ("a", "b")), "c"]
    stream = BlockStreamedWeights(model, store, "cpu", groups=groups, prefetch_depth=prefetch)
    foreground = nn.Linear(3, 3)
    foreground_reference = copy.deepcopy(foreground)
    def prime_hook(mod, args, output):
        stream.prime_first()  # Always return None: never replace foreground output.
    token = foreground.register_forward_hook(prime_hook)
    x = torch.ones(1, 3)
    torch.testing.assert_close(foreground(x), foreground_reference(x), atol=0, rtol=0)
    token.remove()
    assert stream.stats_snapshot()["loaded_groups"] == ["input"]
    assert stream.stats_snapshot()["primed_group"] == "input"
    assert not stream.running and not stream.in_forward
    assert stream.prime_first()  # Idempotent: no duplicate source reads.
    assert stream.stats_snapshot()["prime_loads"] == 1
    for step in range(8):
        if step:
            assert stream.prime_first()  # Reuse after the preceding transaction cleaned up.
        torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
        assert all_meta(model) and not stream.running
        assert stream.stats_snapshot()["resident_bytes"] == 0
    stats = stream.stats_snapshot()
    total = sum(t.numel()*t.element_size() for t in store.values.values())
    assert stats["bytes_read"] == total*8  # Prime did not add a ninth input read.
    assert stats["runs"] == stats["prime_hits"] == stats["prime_loads"] == 8
    stream.close()


def test_prime_cancel_close_and_reprime_before_consumption():
    model = Siblings(); store = ReadIntoStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"])
    assert stream.prime_first() and stream.cancel_prime()
    assert not stream.cancel_prime()
    assert all_meta(model) and stream.stats_snapshot()["pinned_bytes"] == 0
    assert stream.prime_first()
    stream.close(); stream.close()
    assert all_meta(model) and not stream.ready and not stream.loaded
    with pytest.raises(RuntimeError, match="closed"):
        stream.prime_first()


@pytest.mark.parametrize("phase", ["block prime start", "block load:ab", "block tensor:b.bias"])
def test_prime_budget_refusal_returns_false_and_leaves_foreground_intact(phase):
    model = Siblings(); reference = copy.deepcopy(model); store = ReadIntoStore(model)
    refusal = True
    def guard(actual):
        if refusal and actual == phase:
            raise MemoryError("budget refusal")
    stream = BlockStreamedWeights(model, store, "cpu", guard, groups=[WeightGroup("ab", ("a", "b")), "c"])
    assert stream.prime_first() is False
    assert all_meta(model) and not stream.running and stream.stats_snapshot()["resident_bytes"] == 0
    refusal = False
    torch.testing.assert_close(model(torch.ones(1, 3)), reference(torch.ones(1, 3)), atol=0, rtol=0)
    stream.close()


@pytest.mark.parametrize("error_type", [OSError, MemoryError, torch.OutOfMemoryError, RuntimeError])
def test_prime_actual_io_errors_propagate_unchanged_and_cleanup(error_type):
    model = Siblings(); store = ReadIntoStore(model)
    original = store.read_into
    failure = error_type("actual storage error")
    def read_into(key, destination, *, byte_offset=0):
        if key == "b.bias":
            raise failure
        return original(key, destination, byte_offset=byte_offset)
    store.read_into = read_into
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"])
    with pytest.raises(error_type) as caught:
        stream.prime_first()
    assert caught.value is failure
    assert all_meta(model) and not stream.running and stream.stats_snapshot()["resident_bytes"] == 0
    stream.close()


def test_primed_forward_exception_cleanup_and_illegal_reentrant_prime():
    model = Siblings(); store = MemoryStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"])
    assert stream.prime_first()
    model.fail = True
    with pytest.raises(ValueError, match="model failure"):
        model(torch.ones(1, 3))
    assert all_meta(model) and not stream.running
    with stream.execution():
        with pytest.raises(RuntimeError, match="idle"):
            stream.prime_first()
        with pytest.raises(RuntimeError, match="idle"):
            stream.cancel_prime()
        with pytest.raises(RuntimeError, match="idle"):
            stream.close()
    stream.close()


def test_prime_root_scalar_metadata_and_alternate_entry_liveness():
    model = Parent(); reference = copy.deepcopy(model); store = ReadIntoStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("metadata", ("",)), "a", "b"])
    assert stream.prime_first()
    assert stream.stats_snapshot()["primed_group"] == "metadata" and model.scale.dtype == torch.float64
    with stream.execution():
        torch.testing.assert_close(model(torch.ones(1, 3, dtype=torch.float64)), reference(torch.ones(1, 3, dtype=torch.float64)), atol=0, rtol=0)
    assert all_meta(model) and stream.stats_snapshot()["prime_hits"] == 1
    stream.close()


def test_prime_pinned_pool_guard_refusal_is_optional_without_pin_allocation(monkeypatch):
    model = Siblings(); store = ReadIntoStore(model)
    def guard(phase):
        if phase == "allocate bounded pinned pool":
            raise MemoryError("pinned budget refusal")
    stream = BlockStreamedWeights(model, store, "cpu", guard, groups=[WeightGroup("ab", ("a", "b")), "c"])
    # Exercise only the optional-pin guard, before any pin allocator/CUDA call.
    monkeypatch.setattr(stream, "_load", lambda *args, **kwargs: stream._ensure_pin())
    assert stream.prime_first() is False
    assert not stream.priming_attempt and not stream.running
    assert stream.pin is None and not store.copies and all_meta(model)
    stream.close()


def test_prime_simulated_gpu_capacity_refusal_does_not_read_or_initialize_cuda(monkeypatch):
    model = Siblings(); store = ReadIntoStore(model)
    stream = BlockStreamedWeights(model, store, "cpu", groups=[WeightGroup("ab", ("a", "b")), "c"])
    class FakeStream:
        def __init__(self, **kwargs):
            self.synchronized = False
        def synchronize(self):
            self.synchronized = True
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(stream, "_cuda_capacity", lambda size: False)
    stream.device = torch.device("cuda")
    assert stream.prime_first() is False
    assert stream.copy_stream.synchronized and not store.copies
    assert all_meta(model) and not stream.loaded and not stream.running
    stream.device = torch.device("cpu")
    stream.close()


def test_prime_empty_model_skips_and_does_not_start_transaction():
    model = nn.Identity()
    stream = BlockStreamedWeights(model, MemoryStore(model), "cpu", groups=[""])
    assert stream.prime_first() is False
    assert not stream.running and stream.stats_snapshot()["runs"] == 0
    stream.close()


def test_closed_context_unsafe_partition_and_pin_budget_rejected():
    model = Siblings(); store = MemoryStore(model)
    with pytest.raises(ValueError, match="No group"):
        BlockStreamedWeights(model, store, "cpu", groups=["a"])
    with pytest.raises(ValueError, match="Pinned pool"):
        BlockStreamedWeights(model, store, "cpu", groups=["a", "b", "c"], max_pinned_bytes=0)
    stream = BlockStreamedWeights(model, store, "cpu", groups=["a", "b", "c"])
    stream.close()
    with pytest.raises(RuntimeError, match="closed"):
        with stream.execution():
            pass


@pytest.fixture
def fake_cuda_arena(monkeypatch):
    """Run real read-ahead, packing and model hooks on CPU, recording CUDA order."""
    from contextlib import nullcontext
    timeline = []
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            timeline.append(('record', stream, self))
        def synchronize(self):
            timeline.append(('synchronize_event', self))
        def query(self):
            return True
        def elapsed_time(self, other):
            return 0.
    class Stream:
        def __init__(self, **kwargs):
            pass
        def wait_event(self, event):
            timeline.append(('wait', self, event))
        def synchronize(self):
            timeline.append(('synchronize_stream', self))
    compute = Stream()
    original_empty = torch.empty
    def empty(*args, **kwargs):
        if torch.device(kwargs.get('device') or 'cpu').type == 'cuda':
            kwargs['device'] = 'cpu'
            timeline.append(('allocate', args[0]))
        kwargs.pop('pin_memory', None)
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(torch, 'empty', empty)
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda tensor, stream: timeline.append(('use', stream)))
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: compute)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (16*1024**3, 24*1024**3))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 0)
    monkeypatch.setattr(torch.cuda, 'memory_stats', lambda device: {})
    # Any missing CUDA interception must fail instead of initializing a driver.
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU test initialized CUDA'))
    return timeline, compute


class MixedArenaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3).bfloat16())
        self.bias = nn.Parameter(torch.randn(3))
        self.register_buffer('gain', torch.tensor(0.5, dtype=torch.float64))
        self.fail = False
    def forward(self, x):
        result = nn.functional.linear(x, self.weight.float(), self.bias) * self.gain.float()
        if self.fail:
            raise ValueError('block failure')
        return result


def make_fixed_arena_stream(**kwargs):
    model = nn.Sequential(nn.Linear(3, 3), MixedArenaBlock(), MixedArenaBlock(), nn.Linear(3, 3))
    reference = copy.deepcopy(model)
    store = ReadIntoStore(model)
    stream = BlockStreamedWeights(model, store, 'cuda', groups=['0', '1', '2', '3'],
                                  fixed_arena_groups=['1', '2'], cpu_readahead=True,
                                  max_pinned_bytes=4*37, **kwargs)
    return model, reference, store, stream


def test_fixed_arena_lazy_reuse_across_forwards_and_copy_compute_handoff(fake_cuda_arena):
    timeline, compute = fake_cuda_arena
    model, reference, store, stream = make_fixed_arena_stream(max_resident_bytes=1000)
    observations = []
    def observe(module, args, output):
        group = next(str(i) for i, child in enumerate(model) if child is module)
        observations.append((group, stream.fixed_arena, stream.stats['resident_bytes']))
        if group in ('1', '2'):
            assert module.weight.dtype == torch.bfloat16
            assert module.bias.dtype == torch.float32 and module.gain.dtype == torch.float64
            assert module.weight.data_ptr() == stream.fixed_arena.data_ptr()
            assert module.bias.data_ptr() - module.weight.data_ptr() == 256
            assert module.gain.data_ptr() - module.weight.data_ptr() == 512
    handles = [child.register_forward_hook(observe) for child in model]
    x = torch.randn(2, 3)
    try:
        with stream.execution(), torch.no_grad():
            for _ in range(3):
                torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
            stats = stream.stats_snapshot()
            assert stats['fixed_arena_allocations'] == 1
            assert stats['fixed_arena_reuses'] == 5
            assert stats['fixed_arena_bytes'] == stream.fixed_arena_capacity == 520
            assert stats['resident_peak_bytes'] <= 1000
            # At the final ordinary group, unbound main backing is still held
            # and charged exactly once along with the ordinary packed group.
            assert stats['resident_bytes'] == 520 + stream.sizes[3]
        assert observations[0] == ('0', None, stream.sizes[0])
        arenas = [arena for group, arena, _ in observations if group in ('1', '2')]
        assert all(arena is arenas[0] for arena in arenas)
        assert all(resident == 520 for group, _, resident in observations if group in ('1', '2'))
        assert stream.fixed_arena is None and stream.fixed_arena_done is None
        assert all_meta(model) and stream.stats_snapshot()['resident_bytes'] == 0
        assert stream.stats_snapshot()['last_cuda_capacity']['required_bytes'] == stream.sizes[3]
        # Every overwrite is preceded by the compute-done event; every demand
        # then waits on its new copy-ready event before exposing weights to use.
        overwrite_waits = [(i, event) for i, (kind, *rest) in enumerate(timeline)
                           if kind == 'wait' for owner, event in [rest] if owner is stream.copy_stream]
        assert len(overwrite_waits) == 5
        for position, done in overwrite_waits:
            assert ('record', compute, done) in timeline[:position]
            following = timeline[position+1:]
            next_use = next(i for i, item in enumerate(following) if item == ('use', compute))
            ready_wait = [item for item in following[:next_use] if item[0] == 'wait' and item[1] is compute]
            assert ready_wait
            assert ('record', stream.copy_stream, ready_wait[-1][2]) in following[:next_use]
        assert any(item[0] == 'synchronize_event' for item in timeline)
    finally:
        for handle in handles:
            handle.remove()
        stream.close()


@pytest.mark.parametrize('failure', ['guard', 'read', 'block', 'start'])
def test_fixed_arena_failure_cleanup_and_retry(fake_cuda_arena, failure):
    refused = False
    def guard(phase):
        if refused and phase == ('block execution start' if failure == 'start' else 'block arena:2'):
            raise MemoryError('guard failure')
    model, reference, store, stream = make_fixed_arena_stream(guard=guard)
    if failure in ('guard', 'start'):
        refused = True
    elif failure == 'read':
        store.fail = '2.bias'
    else:
        model[1].fail = True
    expected = {'guard': MemoryError, 'start': MemoryError, 'read': OSError, 'block': ValueError}[failure]
    try:
        with pytest.raises(expected):
            model(torch.ones(2, 3))
        assert all_meta(model) and not stream.running and not stream.in_forward
        assert stream.fixed_arena is None and stream.fixed_arena_owner is None
        assert stream.stats_snapshot()['resident_bytes'] == 0
        assert not stream.loaded and not stream.ready
        assert stream.reader is None or not stream.reader.pending
        refused = False
        store.fail = None
        model[1].fail = False
        torch.testing.assert_close(model(torch.ones(2, 3)), reference(torch.ones(2, 3)), atol=0, rtol=0)
        assert all_meta(model) and stream.fixed_arena is not None
    finally:
        stream.close()


def test_fixed_arena_survives_successful_normal_forwards_until_close(fake_cuda_arena):
    model, reference, _, stream = make_fixed_arena_stream(max_resident_bytes=1000)
    x = torch.randn(2, 3)
    backing = None
    try:
        for _ in range(3):
            torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
            if backing is None:
                backing = stream.fixed_arena
            assert stream.fixed_arena is backing
            assert stream.fixed_arena_owner is None
            assert all_meta(model) and not stream.running
            assert stream.stats_snapshot()['resident_bytes'] == stream.fixed_arena_capacity
        assert stream.stats_snapshot()['fixed_arena_allocations'] == 1
        assert stream.stats_snapshot()['fixed_arena_reuses'] == 5
    finally:
        stream.close()
    assert stream.fixed_arena is None and stream.fixed_arena_done is None
    assert stream.stats_snapshot()['resident_bytes'] == stream.stats_snapshot()['fixed_arena_bytes'] == 0


def test_fixed_arena_retained_backing_is_dropped_after_next_root_guard_failure(fake_cuda_arena):
    refused = False
    def guard(phase):
        if refused and phase == 'block execution start':
            raise MemoryError('next request guard')
    model, _, _, stream = make_fixed_arena_stream(guard=guard)
    try:
        model(torch.ones(2, 3))
        assert stream.fixed_arena is not None
        refused = True
        with pytest.raises(MemoryError, match='next request guard'):
            model(torch.ones(2, 3))
        assert stream.fixed_arena is None and all_meta(model)
        assert stream.stats_snapshot()['resident_bytes'] == 0
        assert not stream.in_forward and not stream.running
    finally:
        stream.close()


@pytest.mark.parametrize('failure', ['copy_sync', 'harvest', 'reader_drain'])
def test_fixed_arena_cleanup_failure_drops_backing(fake_cuda_arena, monkeypatch, failure):
    model, _, _, stream = make_fixed_arena_stream()
    try:
        model(torch.ones(2, 3))
        assert stream.fixed_arena is not None
        target, method = {
            'copy_sync': (stream.copy_stream, 'synchronize'),
            'harvest': (stream, '_harvest'),
            'reader_drain': (stream.reader, 'drain'),
        }[failure]
        original = getattr(target, method)
        def fail(*args, **kwargs):
            raise OSError('cleanup failure')
        monkeypatch.setattr(target, method, fail)
        with pytest.raises(OSError, match='cleanup failure'):
            stream._end(keep_fixed_arena=True)
        assert stream.fixed_arena is None and stream.fixed_arena_done is None
        assert all_meta(model) and stream.stats['resident_bytes'] == 0
        assert not stream.running and not stream.loaded and not stream.ready
        monkeypatch.setattr(target, method, original)
    finally:
        stream.close()


def test_fixed_arena_cannot_overwrite_an_active_parent(fake_cuda_arena):
    model = Parent()
    stream = BlockStreamedWeights(model, ReadIntoStore(model), 'cuda',
                                  groups=[WeightGroup('parent', ('',)), 'a', 'b'],
                                  fixed_arena_groups=['parent', 'a'], cpu_readahead=True)
    try:
        with pytest.raises(RuntimeError, match='still belongs to an active group'):
            model(torch.ones(1, 3, dtype=torch.float64))
        assert all_meta(model) and stream.fixed_arena is None
        assert not stream.running and stream.stats_snapshot()['resident_bytes'] == 0
    finally:
        stream.close()


def test_failed_block_does_not_speculatively_load_next_group():
    model = nn.Sequential(MixedArenaBlock(), MixedArenaBlock())
    store = ReadIntoStore(model)
    stream = BlockStreamedWeights(model, store, 'cpu', groups=['0', '1'], prefetch_depth=1)
    model[0].fail = True
    try:
        with pytest.raises(ValueError, match='block failure'):
            model(torch.ones(2, 3))
        assert all(key.startswith('0.') for key, *_ in store.copies)
        assert all_meta(model)
    finally:
        stream.close()


@pytest.mark.parametrize('options,match', [
    ({'fixed_arena_groups': ['missing']}, 'Unknown fixed'),
    ({'fixed_arena_groups': ['0']}, 'Fixed arena requires'),
    ({'fixed_arena_groups': ['0'], 'cpu_readahead': True, 'resident_cache_bytes': 1}, 'Fixed arena requires'),
])
def test_fixed_arena_configuration_rejected_before_tensor_mutation(options, match):
    model = nn.Sequential(nn.Linear(1, 1))
    with pytest.raises(ValueError, match=match):
        BlockStreamedWeights(model, ReadIntoStore(model), 'cuda', groups=['0'], **options)
    assert not any(t.is_meta for t in model.state_dict().values())


@pytest.mark.skipif(os.environ.get('FASTH3_RUN_GPU_TESTS') != '1', reason='Lead-only GPU window')
def test_cuda_fixed_arena_waits_for_delayed_weight_reads_and_reuses_across_forwards():
    from fasth3_platform import BudgetGuard
    class DelayedLinear(nn.Linear):
        def forward(self, x):
            # The next block's DMA can otherwise overwrite this weight before
            # its GEMM even begins. record_stream alone cannot order that write.
            torch.cuda._sleep(20_000_000)
            return super().forward(x)
    with BudgetGuard(device='cuda:0', max_gib=10., reserve_mib=256) as guard:
        model = nn.Sequential(nn.Linear(16, 16), DelayedLinear(16, 16), DelayedLinear(16, 16)).bfloat16().eval()
        reference = copy.deepcopy(model).to('cuda:0')
        stream = BlockStreamedWeights(model, ReadIntoStore(model), 'cuda:0', guard,
                                      groups=['0', '1', '2'], fixed_arena_groups=['1', '2'],
                                      cpu_readahead=True, max_pinned_bytes=4096)
        x = torch.randn(3, 16, device='cuda:0', dtype=torch.bfloat16)
        pointer = None
        try:
            with torch.no_grad():
                for _ in range(3):
                    torch.testing.assert_close(model(x), reference(x), atol=0, rtol=0)
                    current = stream.fixed_arena.data_ptr()
                    pointer = current if pointer is None else pointer
                    assert current == pointer and all_meta(model)
            stats = stream.stats_snapshot()
            assert stats['fixed_arena_allocations'] == 1 and stats['fixed_arena_reuses'] == 5
            assert stats['resident_bytes'] == stats['fixed_arena_bytes'] == stream.fixed_arena_capacity
        finally:
            stream.close()
        assert stream.stats_snapshot()['resident_bytes'] == stream.stats_snapshot()['fixed_arena_bytes'] == 0
