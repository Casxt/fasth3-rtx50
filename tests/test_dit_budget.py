"""Whole-board DiT budget policy with CPU-only CUDA and NVML samples."""
from types import SimpleNamespace
import subprocess
import sys
from uuid import UUID

import pytest

import fasth3_platform as platform
from fasth3_platform import GIB, MIB, GPUBudget, dit_gpu_budget


def cuda_budget(monkeypatch, *, free=8*GIB, total=16*GIB, reserved=8*GIB,
                allocated=GIB, budget=10*GIB, fraction=None):
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda device: None, empty_cache=lambda: None,
        mem_get_info=lambda device: (free, total),
        memory_reserved=lambda device: reserved,
        memory_allocated=lambda device: allocated,
        set_per_process_memory_fraction=lambda value, device: calls.append((value, device)))
    if fraction is not None:
        cuda.get_per_process_memory_fraction = lambda device: fraction
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    guard = SimpleNamespace(budget=GPUBudget(1, budget, 256*MIB, free, total))
    return guard, calls


@pytest.mark.parametrize('board_used,board_total,budget,expected', [
    (12*GIB, 16*GIB, 10*GIB, 59*GIB//8),
    (8*GIB, 16*GIB, 9*GIB, 9*GIB),
    (8*GIB, 8*GIB, 10*GIB, 61*GIB//8),
])
def test_dit_budget_tightens_without_raising_existing_cap(
        monkeypatch, board_used, board_total, budget, expected):
    guard, calls = cuda_budget(monkeypatch, budget=budget)
    original = guard.budget
    samples = []
    def sample(device, receipt=None):
        samples.append(device)
        return board_used, board_total
    monkeypatch.setattr(platform, '_query_gpu_board_memory', sample)
    receipt = {}
    with dit_gpu_budget(guard, receipt):
        assert guard.budget.budget_bytes == expected
        assert guard.budget.reserve_bytes == 256*MIB
        assert calls == [(expected/(16*GIB), 1)]
        assert receipt['external_bytes'] == max(0, board_used-8*GIB)
        assert receipt['source'] == 'nvidia-smi'
        assert receipt['target_bytes'] == 23*GIB//2
    assert samples == [1]
    assert calls[-1] == (budget/(16*GIB), 1)
    assert guard.budget is original
    assert receipt['restored']


def test_dit_budget_uses_cuda_fallback_and_preserves_native_reserve(monkeypatch):
    guard, calls = cuda_budget(monkeypatch, free=256*MIB, reserved=8*GIB)
    monkeypatch.setattr(platform, '_query_gpu_board_memory', lambda device, receipt=None: None)
    with dit_gpu_budget(guard, receipt := {}):
        assert guard.budget.budget_bytes <= 8*GIB
        assert receipt['source'] == 'cuda.mem_get_info'
        assert receipt['external_bytes'] == 31*GIB//4
    assert len(calls) == 2


def test_dit_budget_preserves_stricter_actual_allocator_limit(monkeypatch):
    guard, calls = cuda_budget(monkeypatch, fraction=.5)
    monkeypatch.setattr(platform, '_query_gpu_board_memory', lambda device, receipt=None: (8*GIB, 16*GIB))
    with dit_gpu_budget(guard, {}):
        assert guard.budget.budget_bytes == 8*GIB
    assert calls == [(.5, 1), (.5, 1)]


def test_dit_budget_restores_after_phase_failure(monkeypatch):
    guard, calls = cuda_budget(monkeypatch)
    original = guard.budget
    monkeypatch.setattr(platform, '_query_gpu_board_memory', lambda device, receipt=None: (12*GIB, 16*GIB))
    with pytest.raises(RuntimeError, match='phase failed'):
        with dit_gpu_budget(guard, receipt := {}):
            raise RuntimeError('phase failed')
    assert guard.budget is original
    assert calls[-1] == (10/16, 1)
    assert receipt['restored']


def test_dit_budget_refuses_an_impossible_cap_without_mutating(monkeypatch):
    guard, calls = cuda_budget(monkeypatch, reserved=0)
    original = guard.budget
    monkeypatch.setattr(platform, '_query_gpu_board_memory', lambda device, receipt=None: (12*GIB, 16*GIB))
    with pytest.raises(MemoryError, match='whole-board'):
        with dit_gpu_budget(guard, receipt := {}):
            pytest.fail('unreachable')
    assert guard.budget is original
    assert not calls
    assert receipt['allocator_cap_bytes'] < 0


def test_dit_budget_restores_if_allocator_set_fails(monkeypatch):
    guard, calls = cuda_budget(monkeypatch)
    original = guard.budget
    monkeypatch.setattr(platform, '_query_gpu_board_memory', lambda device, receipt=None: (12*GIB, 16*GIB))
    def set_fraction(value, device):
        calls.append((value, device))
        if len(calls) == 1:
            raise RuntimeError('allocator failed')
    sys.modules['torch'].cuda.set_per_process_memory_fraction = set_fraction
    with pytest.raises(RuntimeError, match='allocator failed'):
        with dit_gpu_budget(guard, receipt := {}):
            pytest.fail('unreachable')
    assert guard.budget is original
    assert calls[-1] == (10/16, 1)
    assert receipt['restored']


@pytest.mark.parametrize('budget', [None, object()])
def test_dit_budget_accepts_cpu_and_mock_guards(monkeypatch, budget):
    monkeypatch.setitem(sys.modules, 'torch', None)
    guard = SimpleNamespace(budget=budget)
    with dit_gpu_budget(guard, receipt := {}):
        assert guard.budget is budget
    assert receipt == {}


def test_board_sample_matches_cuda_uuid_despite_visibility_remapping(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,0')
    cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(uuid='GPU-selected'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='0, GPU-other, 100, 16000\n1, GPU-selected, 12000, 16000\n')
    monkeypatch.setattr(subprocess, 'run', run)
    assert platform._query_gpu_board_memory(0) == (12000*MIB, 16000*MIB)
    assert len(calls) == 1
    assert calls[0][1]['timeout'] == 2


@pytest.mark.parametrize('cuda_uuid', [
    '12345678-1234-5678-9abc-def012345678',
    'GPU-12345678-1234-5678-9abc-def012345678',
    '12345678123456789abcdef012345678',
    'CUDAUUID(12345678-1234-5678-9abc-def012345678)',
    b'GPU-12345678-1234-5678-9abc-def012345678',
    UUID('12345678-1234-5678-9abc-def012345678'),
    UUID('12345678-1234-5678-9abc-def012345678').bytes,
])
def test_board_uuid_formats_use_nvml_sample(monkeypatch, cuda_uuid):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,0')
    cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(uuid=cuda_uuid))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=
        '0, GPU-12345678-1234-5678-9abc-def012345678, 11410, 16303\n'))
    assert platform._query_gpu_board_memory(0, receipt := {}) == (11410*MIB, 16303*MIB)
    assert receipt['board_query_selector'] == '12345678123456789abcdef012345678'
    assert 'board_query_error' not in receipt


def test_board_sample_records_mismatched_device_without_using_wrong_gpu(monkeypatch):
    cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(uuid='GPU-missing'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='0, GPU-other, 100, 16000\n'))
    assert platform._query_gpu_board_memory(0, receipt := {}) is None
    assert receipt['board_query_selector'] == 'missing'
    assert 'did not match' in receipt['board_query_error']


def test_board_sample_falls_back_to_visible_device_when_uuid_unavailable(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace()))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(stdout='0, GPU-a, 100, 16000\n2, GPU-b, 200, 16000\n'))
    assert platform._query_gpu_board_memory(0) == (200*MIB, 16000*MIB)


@pytest.mark.parametrize('failure', [FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 2)])
def test_board_sample_failures_allow_cuda_fallback(monkeypatch, failure):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace()))
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, 'run', run)
    assert platform._query_gpu_board_memory(0, receipt := {}) is None
    assert type(failure).__name__ in receipt['board_query_error']


@pytest.mark.parametrize('stop', ['install-failure', 'first-denoise', 'denoise', 'close-failure'])
def test_runner_restores_budget_after_dit_release(monkeypatch, tmp_path, stop):
    from contextlib import nullcontext
    import weakref
    import torch
    from diffusers.modular_pipelines.minimax_h3 import modular_blocks_minimax_h3 as official
    from fasth3_pipeline import encoder, long_video, runner
    from fasth3_pipeline.dmd2 import DMD2Plan
    from fasth3_pipeline.streaming import ManagedStream

    guard, caps = cuda_budget(monkeypatch, fraction=10/16)
    original = guard.budget
    cuda = sys.modules['torch'].cuda
    monkeypatch.setitem(sys.modules, 'torch', torch)
    for name, value in vars(cuda).items():
        monkeypatch.setattr(torch.cuda, name, value)
    monkeypatch.setattr(platform, '_query_gpu_board_memory', lambda device, receipt=None: (12*GIB, 16*GIB))
    released = []
    class Guard:
        budget = original
        def __call__(self, phase):
            if phase == 'DiT released':
                assert self.budget is original
    guard = Guard()
    class Resource:
        def __init__(self, name):
            self.name = name
        def close(self):
            released.append(self.name)
            if self.name == 'dit' and stop == 'close-failure':
                raise RuntimeError('close failed')
        def execution(self):
            return nullcontext()
    dit = ManagedStream(Resource('dit'), Resource('dit-source'), 'mock')
    qwen = ManagedStream(Resource('qwen'), Resource('qwen-source'), 'mock')
    core = SimpleNamespace(sub_blocks={'after_denoise': None})
    class Pipe:
        patch_size = (1, 2, 2)
        keyframe_noise_aug = 0
        def update_components(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    pipe = Pipe()
    blocks = SimpleNamespace(inputs=[], init_pipeline=lambda path: pipe,
                             sub_blocks={'before_encode': None, 'text_encoder': None,
                                         'denoise': SimpleNamespace(sub_blocks={'t2va': core})})
    monkeypatch.setattr(official, 'MiniMaxH3Blocks', lambda: blocks)
    monkeypatch.setattr(runner.DMD2Plan, 'from_checkpoint', lambda *args: DMD2Plan())
    monkeypatch.setattr(runner, 'configure_lightweight_components', lambda pipe: None)
    monkeypatch.setattr(runner, 'load_local_components', lambda *args: None)
    monkeypatch.setattr(runner, '_run_block', lambda block, pipe, state: state)
    monkeypatch.setattr(runner, '_clear_device', lambda device: released.append('clear'))
    monkeypatch.setattr(long_video, 'allow_released_maximum', lambda pipe: None)
    monkeypatch.setattr(encoder, 'load_text_encoder', lambda *args, **kwargs: (object(), qwen))
    predictions = []
    latent_storage = []
    reference_plan = DMD2Plan()
    expected = {}
    def model(**kwargs):
        # A completed step's predictions must not pin CUDA allocator segments
        # while the next forward computes its attention workspace.
        assert all(ref() is None for pair in predictions for ref in pair)
        states = (kwargs['hidden_states'], kwargs['audio_hidden_states'])
        storage = tuple(value.untyped_storage().data_ptr() for value in states)
        if latent_storage:
            assert storage == latent_storage[0]
        latent_storage.append(storage)
        for modality, value in zip(('video', 'audio'), states):
            torch.testing.assert_close(value, expected[modality], atol=0, rtol=0)
        step = len(predictions) + 1
        video = torch.full_like(states[0], .0625 * step, dtype=torch.bfloat16)
        audio = torch.full_like(states[1], -.03125 * step, dtype=torch.bfloat16)
        predictions.append((weakref.ref(video), weakref.ref(audio)))
        return video, audio
    monkeypatch.setattr(runner, 'load_transformer', lambda *args, **kwargs: (model, dit))
    def prepare(core, pipe, state, task):
        for key, value in dict(
                prompt_embeds=torch.zeros(1, 1, 3), latents=torch.full((1, 3), .5),
                audio_latents=torch.full((1, 3), -.25), token_tags=torch.zeros(3),
                position_ids=torch.zeros(3, 3), video_indices=torch.tensor([1]),
                audio_indices=torch.tensor([2]), text_indices=torch.tensor([0]),
                num_condition_video_rows=0, num_condition_audio_rows=0).items():
            state.set(key, value)
        expected.update(video=state.get('latents')[None].clone(),
                        audio=state.get('audio_latents')[None].clone())
        return state
    monkeypatch.setattr(runner, '_prepare_denoising', prepare)
    monkeypatch.setattr(runner, 'layout_from_state', lambda *args, **kwargs: object())
    def install(*args, **kwargs):
        assert guard.budget.budget_bytes == 59*GIB//8
        if stop == 'install-failure':
            raise RuntimeError('install failed')
        return {}
    monkeypatch.setattr(runner, 'install_vsa', install)
    checked_predictions = []
    def checkpoint(phase, tensors=None, **kwargs):
        if phase.startswith('step-'):
            video, audio = predictions[-1]
            assert tensors['video_prediction'] is video()
            assert tensors['audio_prediction'] is audio()
            assert video() is not None and audio() is not None
            step = len(checked_predictions)
            for modality, prediction in (('video', video), ('audio', audio)):
                expected[modality] = reference_plan.update(expected[modality], prediction(), modality, step)
                torch.testing.assert_close(tensors[modality], expected[modality], atol=0, rtol=0)
            assert tuple(tensors[name].untyped_storage().data_ptr() for name in ('video', 'audio')) == latent_storage[0]
            checked_predictions.append(phase)
        return phase == stop
    diagnostics = SimpleNamespace(streaming={}, debug_checks=False, forward_count=0,
        checkpoint=checkpoint,
        stopped=lambda phase: {'stopped_after': phase})
    kwargs = dict(task='t2va', base_model=tmp_path, transformer_path=tmp_path,
                  text_encoder_path=tmp_path, prompt='test', device='cpu', guard=guard,
                  diagnostics=diagnostics)
    if stop.endswith('failure'):
        with pytest.raises(RuntimeError, match='failed'):
            runner.run_h3(**kwargs)
    else:
        assert runner.run_h3(**kwargs) == {'stopped_after': stop}
    assert guard.budget is original
    assert caps[-1] == (10/16, 1)
    assert diagnostics.streaming['dit_budget']['restored']
    assert 'dit' in released and 'dit-source' in released
    assert pipe.transformer is None
    assert diagnostics.streaming['dit']['closed']
    assert released[-1] == 'clear'
    expected_forwards = 0 if stop == 'install-failure' else 1 if stop == 'first-denoise' else 8
    assert len(predictions) == len(checked_predictions) == expected_forwards
    assert len(latent_storage) == expected_forwards
    assert all(ref() is None for pair in predictions for ref in pair)
