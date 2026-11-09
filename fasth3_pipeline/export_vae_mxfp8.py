"""Export an offline MXFP8 overlay for the H3 video decoder FFNs."""
import argparse
from pathlib import Path
import tempfile

import torch
from diffusers import AutoencoderKLMiniMaxH3
from safetensors.torch import save_file

from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
from fasth3_pipeline.vae_mxfp8 import SCHEMA, ffn_linears, source_digest
from fasth3_platform import BudgetGuard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vae', type=Path, required=True, help='Local BF16 video VAE directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists')
    if not args.device.startswith('cuda'):
        parser.error('Export requires CUDA SM120')
    data, metadata = {}, {'schema': SCHEMA}
    with BudgetGuard(args.device) as guard, torch.no_grad():
        if torch.cuda.get_device_capability(args.device) != (12, 0):
            parser.error('Export requires CUDA SM120')
        vae = AutoencoderKLMiniMaxH3.from_pretrained(args.vae.resolve(strict=True), local_files_only=True,
                                                    torch_dtype=torch.bfloat16)
        for name, _, _, _, source in ffn_linears(vae):
            guard(f'export VAE MXFP8:{name}')
            n, k = source.weight.shape
            if n % 32 or k % 128:
                raise ValueError(f'VAE linear cannot use packed scales: {name}')
            metadata[name] = source_digest(source)
            weight = torch.empty((n, k), dtype=torch.float8_e4m3fn)
            scales = torch.empty((n // 32, k // 128, 32, 4), dtype=torch.uint8)
            for start in range(0, n, 1024):
                quantized = quantize_mxfp8_cute(source.weight[start:start + 1024].to(args.device))
                stop = start + len(quantized.values)
                weight[start:stop].copy_(quantized.values.cpu())
                packed = quantized.scales.reshape(-1, 32, k // 128, 4).permute(0, 2, 1, 3).contiguous()
                scales[start // 32:stop // 32].copy_(packed.cpu())
            data[name + '.weight'], data[name + '.weight_scale'] = weight, scales
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        path = Path(directory) / 'overlay.safetensors'
        save_file(data, path, metadata=metadata)
        if args.output.exists():
            raise FileExistsError(args.output)
        path.rename(args.output)
    print(f'Exported {len(data) // 2} video decoder FFN linears to {args.output.resolve()}')


if __name__ == '__main__':
    main()
