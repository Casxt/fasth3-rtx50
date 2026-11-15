#!/usr/bin/env python3
"""Install the Windows lock or Linux snapshot in an isolated Python environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
VERSION_CHECK = (
    'import platform,sys; '
    'sys.exit(0 if platform.python_version() == "3.12.13" else '
    '"Expected Python 3.12.13 in .venv-win. Move the old environment aside before retrying.")'
)
LINUX_CHECK = (
    'import sys; sys.exit(0 if sys.platform == "linux" and sys.version_info[:2] == (3,12) '
    'else "Linux with Python 3.12 is required")'
)


def default(env, name, value):
    if not env.get(name):
        env[name] = str(value)


def configuration(args):
    env = dict(os.environ)
    platform = args.platform or ('windows' if sys.platform == 'win32' else sys.platform)
    if platform not in ('windows', 'linux'):
        raise ValueError('Bootstrap supports Windows and Linux only')
    if not args.dry_run and platform != ('windows' if sys.platform == 'win32' else sys.platform):
        raise ValueError('--platform may select another OS only with --dry-run')
    if platform == 'windows':
        if args.work_dir or args.python or args.index_url:
            raise ValueError('--work-dir, --python and --index-url are Linux-only; Windows uses the repository lock and .venv-win')
        work = ROOT
        environment = ROOT / '.venv-win'
        python = environment / 'Scripts/python.exe'
        default(env, 'UV_CACHE_DIR', ROOT / '.uv-cache')
        default(env, 'UV_PYTHON_INSTALL_DIR', ROOT / '.uv-python')
        steps = []
        if not python.is_file():
            steps.append(('Create Windows Python 3.12.13 environment', ['uv', 'venv', '--python', '3.12.13', str(environment)]))
        steps += [
            ('Check Windows Python version', [str(python), '-c', VERSION_CHECK]),
            ('Sync Windows dependencies with hashes', ['uv', 'pip', 'sync', '--require-hashes', '--python', str(python), str(ROOT / 'requirements-win.lock')]),
            ('Install FastH3 source', ['uv', 'pip', 'install', '--python', str(python), '--no-deps', '--editable', str(ROOT)]),
            ('Check package compatibility', ['uv', 'pip', 'check', '--python', str(python)]),
        ]
    else:
        work = Path(args.work_dir or env.get('FASTH3_WORK_DIR') or ROOT).expanduser().resolve()
        environment = work / '.venv-linux'
        python = environment / 'bin/python'
        base_python = args.python or env.get('FASTH3_PYTHON') or sys.executable
        index = args.index_url or env.get('FASTH3_PYPI_INDEX') or 'https://pypi.org/simple'
        caches = {'UV_CACHE_DIR': 'cache/uv', 'PIP_CACHE_DIR': 'cache/pip',
                  'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache', 'HF_HOME': 'cache/huggingface',
                  'TORCH_HOME': 'cache/torch', 'CUDA_CACHE_PATH': 'cache/cuda',
                  'TRITON_CACHE_DIR': 'cache/triton', 'CUTE_DSL_CACHE_DIR': 'cache/cutlass',
                  'CUTE_DSL_DUMP_DIR': 'tmp/cute-dump'}
        for name, relative in caches.items():
            default(env, name, work / relative)
        sys.path.insert(0, str(ROOT))
        from fasth3_platform import effective_cpu_count

        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
            default(env, name, min(8, effective_cpu_count()))
        steps = [('Check Linux Python version', [base_python, '-c', LINUX_CHECK])]
        if not python.is_file():
            steps.append(('Create Linux environment', [base_python, '-m', 'venv', str(environment)]))
        uv = str(environment / 'bin/uv')
        steps += [
            ('Check Linux environment Python', [str(python), '-c', LINUX_CHECK]),
            ('Install uv 0.13.8', [str(python), '-m', 'pip', 'install', '--index-url', index, 'uv==0.13.8']),
            ('Install CUDA 12.8 stack', [uv, 'pip', 'install', '--python', str(python), '--no-deps', '--index-url', 'https://download.pytorch.org/whl/cu128', '-r', str(ROOT / 'requirements-linux-cu128.txt')]),
            ('Install Linux version snapshot', [uv, 'pip', 'install', '--python', str(python), '--index-url', index, '-r', str(ROOT / 'requirements-linux.txt')]),
            ('Install FastH3 source', [uv, 'pip', 'install', '--python', str(python), '--no-deps', '--index-url', index, '--editable', str(ROOT)]),
            ('Check package compatibility', [uv, 'pip', 'check', '--python', str(python)]),
        ]
    steps.append(('Check FastH3 environment', [str(python), '-m', 'fasth3_pipeline.cli', 'doctor']))
    return platform, work, environment, env, steps


def public_argument(value):
    """Avoid echoing authentication from an optional private index URL in a plan."""
    if '://' not in value:
        return value
    url = urlsplit(value)
    authority = url.netloc.rsplit('@', 1)[-1]
    return urlunsplit((url.scheme, authority, url.path, '<redacted>' if url.query else '', ''))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without installing or creating files')
    parser.add_argument('--platform', choices=('windows', 'linux'), help='Select an OS; cross-OS plans require --dry-run')
    parser.add_argument('--work-dir', help='Linux work directory (default: FASTH3_WORK_DIR or repository)')
    parser.add_argument('--python', help='Linux Python 3.12 executable (default: FASTH3_PYTHON or current Python)')
    parser.add_argument('--index-url', help='Linux PyPI index (default: FASTH3_PYPI_INDEX or public PyPI)')
    args = parser.parse_args(argv)
    if args.dry_run:
        sys.dont_write_bytecode = True
    try:
        platform, work, environment, env, steps = configuration(args)
        if args.dry_run:
            print(json.dumps({'platform': platform, 'environment': str(environment),
                              'steps': [{'label': label, 'argv': [public_argument(x) for x in command]}
                                        for label, command in steps]}, indent=2))
            return 0
        if platform == 'linux':
            (work / 'cache').mkdir(parents=True, exist_ok=True)
            Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
        for label, command in steps:
            print(label, flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        print(f'Environment ready: {environment}')
        return 0
    except subprocess.CalledProcessError as error:
        print(f'Bootstrap failed during {label} (exit {error.returncode})', file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f'Bootstrap failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
