"""Export BF16 AdaLN outputs from local packed MXFP8 weights for the 8-step plan."""
import argparse
from itertools import product
import json
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from fasth3_pipeline.adaln import checkpoint_identity
from fasth3_pipeline.pdd import PDDPlan
from fasth3_pipeline.storage import PersistentTensorStore
from fasth3_platform import BudgetGuard


def schedule():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

    plan = PDDPlan.create()
    rows = set()
    for step, visual, audio in product(range(len(plan.indices) - 1), (0, 1), (0, 1)):
        video_time = plan.clean_time('video', step)
        times, _ = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            torch.arange(visual + 1), torch.arange(visual + 1, visual + audio + 2),
            visual, audio, 1, video_time, plan.clean_time('audio', step), max(video_time, .999), 1.)
        rows.add(tuple(times.tolist()))
    return sorted(rows)


def main(argv=None):
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists')
    config = json.loads((args.transformer / 'config.json').read_text(encoding='utf-8'))
    config.pop('pdd_steps', None)
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    times = schedule()
    data = {}
    with BudgetGuard() as guard, torch.no_grad(), PersistentTensorStore(args.transformer, backend='file_readinto') as store:
        time_weights = {key: store.tensor('time_embedder.' + key).cuda()
                        for key in model.time_embedder.state_dict()}
        model.time_embedder.load_state_dict(time_weights, assign=True)
        time_inputs = [F.silu(model.time_embedder(model.time_proj(torch.tensor(row, device='cuda', dtype=torch.float32))
                                                .to(time_weights['linear_1.weight'].dtype))).to(torch.bfloat16)
                       for row in times]
        for i, block in enumerate(model.transformer_blocks):
            guard(f'export AdaLN block {i}')
            prefix = f'transformer_blocks.{i}.adaln_proj.linear'
            weight, bias = (store.tensor(f'{prefix}.{key}').cuda() for key in ('weight', 'bias'))
            width = 6 * block.adaln_proj.hidden_size
            data[f'block-{i}'] = torch.cat([F.linear(x, weight, bias).view(-1, width).cpu()
                                          for x in time_inputs])
            del weight, bias
            if i % 10 == 0:
                print(f'Exported AdaLN block {i}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        temp = Path(directory) / 'adaln.safetensors'
        save_file(data, temp, metadata={'schema': 'fasth3-adaln-v1',
                                       'checkpoint': checkpoint_identity(args.transformer),
                                       'timesteps': json.dumps(times)})
        if args.output.exists():
            raise FileExistsError(args.output)
        temp.rename(args.output)


if __name__ == '__main__':
    main()
