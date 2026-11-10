"""FastH3 V2's published eight-forward DMD2 contract."""
import json
from pathlib import Path

import torch


RUNGS = (999, 874, 749, 624, 500, 375, 250, 125)
SHIFTS = {'video': 10.0, 'audio': 3.0}


class DMD2Plan:
    def __init__(self, device='cpu'):
        from diffusers import MiniMaxH3Scheduler

        base = torch.tensor([rung / 1000 for rung in RUNGS] + [0.0], dtype=torch.float32)
        self.schedulers = {}
        self.sigmas = {}
        for modality, shift in SHIFTS.items():
            scheduler = MiniMaxH3Scheduler(shift=shift, shift_provided_sigmas=False)
            sigmas = shift * base / (1 + (shift - 1) * base)
            scheduler.set_timesteps(sigmas=sigmas, device=device)
            self.schedulers[modality] = scheduler
            self.sigmas[modality] = sigmas

    @classmethod
    def from_checkpoint(cls, base_model, device):
        root = Path(base_model)
        expected = {
            'schema_version': 'fasth3-inference-contract-v1',
            'task': 't2av', 'dmd_denoising_steps': list(RUNGS),
            'transformer_forwards': len(RUNGS), 'num_inference_steps': len(RUNGS) + 1,
            'video_scheduler_shift': SHIFTS['video'], 'audio_scheduler_shift': SHIFTS['audio'],
            'attention_backend': 'VIDEO_SPARSE_ATTN_H3', 'vsa_tile_size': 64, 'vsa_sparsity': 0.8,
            'guidance_scale': 1.0,
        }

        def read(name):
            path = root / name
            if not path.is_file() or path.stat().st_size > 1 << 20:
                raise ValueError(f'Missing or oversized FastH3 V2 metadata: {path}')
            contents = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(contents, dict):
                raise ValueError(f'FastH3 V2 metadata must be a JSON object: {path}')
            return contents

        contract = read('fastvideo_inference.json')
        for field, value in expected.items():
            if contract.get(field) != value:
                raise ValueError(f'FastH3 V2 inference contract mismatch: {field}')
        for modality, name in (('video', 'scheduler'), ('audio', 'audio_scheduler')):
            config = read(f'{name}/scheduler_config.json')
            if (config.get('_class_name') != 'MiniMaxH3Scheduler' or config.get('shift') != SHIFTS[modality]
                    or config.get('stochastic_sampling', False)):
                raise ValueError(f'FastH3 V2 {modality} scheduler differs from the published contract')
        return cls(device)

    def clean_time(self, modality, step):
        return float(1.0 - self.sigmas[modality][step])

    def scale_initial_noise(self, sample, modality, condition_rows=0):
        if modality not in self.schedulers or not 0 <= condition_rows <= sample.shape[1]:
            raise ValueError('Invalid FastH3 V2 modality or conditioning row count')
        # The upstream FastVideo DMD2 path starts from unscaled Gaussian noise.
        return sample

    def update(self, sample, prediction, modality, step, condition_rows=0):
        if not 0 <= condition_rows <= sample.shape[1] or sample.shape != prediction.shape:
            raise ValueError('Invalid packed FastH3 V2 prediction or conditioning rows')
        scheduler = self.schedulers[modality]
        if not condition_rows:
            return scheduler.step(prediction.float(), scheduler.timesteps[step], sample, return_dict=False)[0]
        # Endpoint rows precede the generated video and stay fixed across steps.
        target = slice(condition_rows, None)
        sample[:, target] = scheduler.step(prediction[:, target].float(), scheduler.timesteps[step],
                                           sample[:, target], return_dict=False)[0]
        return sample
