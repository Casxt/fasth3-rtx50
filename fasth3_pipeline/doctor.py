"""Check the local FastH3 Python and CUDA environment."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import shutil
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fasth3 doctor', description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Initialize CUDA and check for an SM120 GPU')
    args = parser.parse_args(argv)
    required = ('torch', 'torchvision', 'diffusers', 'transformers', 'nvidia-cutlass-dsl', 'cuda-python',
                'safetensors', 'huggingface-hub', 'accelerate', 'av', 'imageio', 'imageio-ffmpeg',
                'filelock', 'numpy', 'pillow', 'protobuf', 'psutil', 'sentencepiece')
    report = {'python': sys.version.split()[0], 'platform': platform.platform(),
              'curl': bool(shutil.which('curl.exe') or shutil.which('curl')), 'packages': {}, 'errors': []}
    if not report['curl']:
        report['errors'].append('Missing curl executable for fasth3 prepare')
    for name in (*required, 'open-h3-ir'):
        try:
            report['packages'][name] = version(name)
        except PackageNotFoundError:
            report['packages'][name] = None
            if name in required:
                report['errors'].append(f'Missing package: {name}')
    if args.gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError('PyTorch CUDA is unavailable')
            capability = torch.cuda.get_device_capability()
            free, total = torch.cuda.mem_get_info()
            report['gpu'] = {'name': torch.cuda.get_device_name(), 'capability': list(capability),
                             'cuda': torch.version.cuda, 'free_mib': free >> 20, 'total_mib': total >> 20}
            if capability != (12, 0):
                report['errors'].append('FastH3 requires a CUDA SM120 GPU')
        except (ImportError, RuntimeError) as error:
            report['errors'].append(str(error))
    print(json.dumps(report, indent=2))
    return int(bool(report['errors']))
