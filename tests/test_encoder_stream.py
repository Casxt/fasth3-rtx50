"""Small CPU tests: no actual encoder weights or GPU allocations."""
import types
import unittest
import torch
from torch import nn
from fasth3_pipeline.encoder import EncoderStream, _capture_truncated_hidden_state


class DirectVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Embedding(3, 2, device='meta')

    def forward(self, x):
        # This access bypasses the child embedding's forward hooks.
        return x + self.pos_embed.weight[0]


class MemoryStore:
    def __init__(self, model):
        self.values = {k: torch.ones(v.shape, dtype=v.dtype) for k, v in model.state_dict().items()}
        self.weight_map = dict.fromkeys(self.values, 'synthetic')

    def validate_keys(self, model):
        assert set(model.state_dict()) == set(self.values)

    def tensor(self, key):
        return self.values[key].clone()


def tiny_model():
    outer = nn.Module()
    outer.model = nn.Module()
    outer.model.visual = DirectVisual()
    language = nn.Module()
    language.embed_tokens = nn.Embedding(4, 2, device='meta')
    language.norm = nn.LayerNorm(2, device='meta')
    language.layers = nn.ModuleList([nn.Linear(2, 2, device='meta')])
    outer.model.language_model = language
    return outer


class EncoderTests(unittest.TestCase):
    def test_visual_direct_parameter_access_and_release(self):
        model = tiny_model()
        phases = []
        stream = EncoderStream(model, MemoryStore(model), 'cpu', phases.append)
        self.assertTrue(model.model.visual.pos_embed.weight.is_meta)
        output = model.model.visual(torch.zeros(2))
        torch.testing.assert_close(output, torch.ones(2))
        self.assertTrue(model.model.visual.pos_embed.weight.is_meta)
        self.assertIn('encoder:load:model.visual', phases)
        stream.close()
        self.assertFalse(hasattr(model, '_hf_hook'))

    def test_language_group_is_released_after_exception(self):
        model = tiny_model()
        stream = EncoderStream(model, MemoryStore(model), 'cpu', lambda phase: None)
        with self.assertRaises(RuntimeError):
            model.model.language_model.layers[0](torch.zeros(3))
        self.assertTrue(model.model.language_model.layers[0].weight.is_meta)
        stream.close()

    def test_partial_load_failure_releases_already_loaded_parameters(self):
        model = tiny_model()
        store = MemoryStore(model)
        original = store.tensor
        def tensor(key):
            if key == 'model.language_model.layers.0.bias':
                raise ValueError('synthetic interrupted load')
            return original(key)
        store.tensor = tensor
        stream = EncoderStream(model, store, 'cpu', lambda phase: None)
        with self.assertRaisesRegex(ValueError, 'interrupted load'):
            model.model.language_model.layers[0](torch.zeros(2))
        self.assertTrue(model.model.language_model.layers[0].weight.is_meta)
        self.assertTrue(model.model.language_model.layers[0].bias.is_meta)
        stream.close()

    def test_failed_constructor_removes_partially_installed_hooks(self):
        model = tiny_model()
        model.model.visual.register_buffer('bad', torch.empty(1, device='meta'), persistent=False)
        with self.assertRaisesRegex(ValueError, 'Uninitialized nonpersistent'):
            EncoderStream(model, MemoryStore(model), 'cpu', lambda phase: None)
        for module in model.modules():
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._forward_hooks)
        self.assertTrue(model.model.visual.pos_embed.weight.is_meta)

    def test_truncation_returns_prenorm_not_normalized(self):
        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.norm = nn.LayerNorm(2)
                self.fail = False
            def forward(self, x, **kwargs):
                result = self.language_model.norm(x)
                if self.fail:
                    raise RuntimeError('synthetic forward failure')
                return types.SimpleNamespace(last_hidden_state=result)
        outer = nn.Module()
        outer.model = Core()
        _capture_truncated_hidden_state(outer, 50)
        x = torch.tensor([[2., 4.]])
        output = outer.model(x)
        self.assertEqual(len(output.hidden_states), 51)
        torch.testing.assert_close(output.hidden_states[50], x)
        self.assertFalse(torch.allclose(output.last_hidden_state, x))
        self.assertFalse(outer.model.language_model.norm._forward_pre_hooks)
        outer.model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'synthetic forward failure'):
            outer.model(x)
        self.assertFalse(outer.model.language_model.norm._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
