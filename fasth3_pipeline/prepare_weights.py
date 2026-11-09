"""Download pinned upstream assets and prepare local H3 weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from tempfile import NamedTemporaryFile


MODEL_LICENSE = 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE'
_METADATA = {
    'LICENSE', 'LICENSE-Qwen3-VL', 'NOTICE', 'audio_scheduler/scheduler_config.json',
    'scheduler/scheduler_config.json', 'fastvideo_inference.json', 'modular_model_index.json',
    'checkpoint_content.json', 'checkpoint_metadata.json',
}
_TOKENIZER = {'merges.txt', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'}
_PROCESSOR = _TOKENIZER | {'chat_template.json', 'preprocessor_config.json', 'video_preprocessor_config.json'}


@dataclass(frozen=True)
class Source:
    folder: str
    repo: str
    revision: str
    count: int


OMNI = Source('bf16', 'FastVideo/FastVideo-FastH3-Omni-8-Step-V1',
              '53e0d26a1691aa473b6262d8196a8feb2c297df0', 20)
T2VA = Source('t2va-v2-bf16', 'FastVideo/FastVideo-FastH3-8-Step-V2',
              '3da2ddfe1954d9cda4c05b643dc0f26007a655c5', 25)
SHARED = Source('shared', 'MiniMaxAI/MiniMax-H3', '9bfb6693f2cf6de171db46d1aa586f67d773a1da', 19)
ENCODER = Source('encoder-nvfp4', 'FastVideo/FastH3-Omni-NVFP4-QAD',
                 'f01e6a23ac6066aa9eb622cf94b6259442e70ec2', 13)


@dataclass(frozen=True)
class Asset:
    path: str
    size: int
    digest: str
    algorithm: str


def _allow(source, path):
    p = PurePosixPath(path)
    if len(p.parts) != 2:
        return path in _METADATA if source in (OMNI, T2VA) else source == SHARED and path == 'LICENSE'
    component, name = p.parts
    if source in (OMNI, T2VA):
        if path in _METADATA:
            return True
        directory = 'transformer_ref' if source == OMNI else 'transformer'
        return component == directory and (name in {'config.json', 'diffusion_pytorch_model.safetensors.index.json'}
            or re.fullmatch(r'diffusion_pytorch_model-\d{5}-of-00014\.safetensors', name))
    if source == ENCODER:
        return component == 'text_encoder' and (name in _PROCESSOR | {'config.json', 'model.safetensors.index.json'}
            or re.fullmatch(r'model-\d{5}-of-00004\.safetensors', name))
    if component == 'tokenizer':
        return name in _TOKENIZER
    if component == 'processor':
        return name in _PROCESSOR
    if component == 'audio_vae':
        return name in {'config.json', 'diffusion_pytorch_model.safetensors'}
    return component == 'vae' and (name in {'config.json', 'diffusion_pytorch_model.safetensors.index.json'}
        or re.fullmatch(r'diffusion_pytorch_model-\d{5}-of-00003\.safetensors', name))


def _assets(api, source):
    info = api.model_info(source.repo, revision=source.revision, files_metadata=True)
    if info.sha != source.revision:
        raise ValueError(f'Revision mismatch for {source.repo}')
    result = []
    for file in info.siblings:
        if not _allow(source, file.rfilename):
            continue
        digest = file.lfs.sha256 if file.lfs else file.blob_id
        algorithm = 'sha256' if file.lfs else 'git-sha1'
        length = 64 if file.lfs else 40
        if (not isinstance(file.size, int) or file.size <= 0 or not isinstance(digest, str)
                or not re.fullmatch(f'[0-9a-f]{{{length}}}', digest)):
            raise ValueError(f'Incomplete upstream checksum metadata: {source.repo}/{file.rfilename}')
        result.append(Asset(file.rfilename, file.size, digest, algorithm))
    if len(result) != source.count or len({item.path for item in result}) != source.count:
        raise ValueError(f'Unexpected pinned file inventory for {source.repo}: {len(result)} files')
    return result


def _inside(root, path):
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes model root: {path}')
    return path


def _matches(path, asset):
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    with path.open('rb') as stream:
        if asset.algorithm == 'git-sha1':
            digest = hashlib.sha1(f'blob {asset.size}\0'.encode())
            while block := stream.read(8 << 20):
                digest.update(block)
            actual = digest.hexdigest()
        else:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    return actual == asset.digest


def _receipt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix='.tmp', delete=False) as file:
        file.write((json.dumps(payload, indent=2, sort_keys=True) + '\n').encode())
        temporary = Path(file.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _download(curl, root, source, asset, token):
    target = _inside(root, root / source.folder / asset.path)
    partial = _inside(root, root / '.downloads' / source.folder / source.revision / (asset.path + '.partial'))
    if target.exists():
        if _matches(target, asset):
            return target.stat().st_mtime_ns
        raise ValueError(f'Existing upstream asset failed verification: {target}; move it aside and retry')
    partial.parent.mkdir(parents=True, exist_ok=True)
    if not _matches(partial, asset):
        if partial.exists() and partial.stat().st_size >= asset.size:
            raise ValueError(f'Complete download failed verification: {partial}; move it aside and retry')
        # curl forwards Authorization only to the original host; never use --location-trusted.
        config = f'header = "Authorization: Bearer {token}"\n' if token else ''
        url = f'https://huggingface.co/{source.repo}/resolve/{source.revision}/{asset.path}'
        subprocess.run([curl, '-q', '--config', '-', '--location', '--fail', '--silent', '--show-error',
                        '--retry', '4', '--retry-delay', '3', '--connect-timeout', '15',
                        '--continue-at', '-', '--output', str(partial), url], input=config, text=True, check=True)
        if not _matches(partial, asset):
            raise ValueError(f'Download failed checksum verification: {asset.path}')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(target)
    return target.stat().st_mtime_ns


def _prepare_source(curl, root, source, assets, token):
    receipt = _inside(root, root / source.folder / '.verified-source.json')
    old = json.loads(receipt.read_text('utf-8')) if receipt.is_file() else {}
    recorded = old.get('files', {}) if old.get('source') == asdict(source) else {}
    pending, complete = [], {}
    for asset in assets:
        target = _inside(root, root / source.folder / asset.path)
        previous = recorded.get(asset.path)
        if target.is_file() and previous == {**asdict(asset), 'mtime_ns': target.stat().st_mtime_ns}:
            if target.stat().st_size == asset.size:
                complete[asset.path] = previous
                continue
        pending.append(asset)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for asset, mtime in zip(pending, pool.map(lambda a: _download(curl, root, source, a, token), pending)):
            complete[asset.path] = {**asdict(asset), 'mtime_ns': mtime}
            print(f'Verified {source.folder}/{asset.path}', flush=True)
    _receipt(receipt, {'source': asdict(source), 'files': complete})


def _verify_conversion(root, destination, manifest, *, written=False):
    from .storage import INDEX

    index = json.loads(_inside(root, destination / INDEX).read_text('utf-8'))['weight_map']
    hashes = manifest['shard_sha256']
    if set(index.values()) != hashes.keys():
        raise ValueError(f'Existing conversion shard coverage mismatch: {destination}')
    receipt = _inside(root, destination / '.verified-conversion.json')
    previous = json.loads(receipt.read_text('utf-8')) if receipt.is_file() else {}
    files = {}
    for name, checksum in hashes.items():
        if Path(name).name != name or not re.fullmatch(r'[0-9a-f]{64}', checksum):
            raise ValueError(f'Unsafe conversion shard name or digest: {name}')
        shard = _inside(root, destination / name)
        stat = shard.stat()
        stamp = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': checksum}
        if not written and stamp != previous.get(name):
            if not _matches(shard, Asset(name, stat.st_size, checksum, 'sha256')):
                raise ValueError(f'Converted shard failed checksum verification: {shard}')
        files[name] = stamp
    if previous != files:
        _receipt(receipt, files)


def _convert(root, source, raw_component, output_folder, profile):
    import torch
    from fasth3_kernels.mxfp8 import PACKED_SCALE_LAYOUT
    from .mxfp8_linear import MANIFEST
    from .storage import INDEX
    from .weight_conversion import export_mxfp8

    original = _inside(root, root / source.folder / raw_component)
    destination = _inside(root, root / output_folder / raw_component)
    if destination.exists():
        manifest_path = _inside(root, destination / MANIFEST)
        if not manifest_path.is_file() or not _inside(root, destination / INDEX).is_file():
            raise ValueError(f'Existing conversion is incomplete: {destination}')
        manifest = json.loads(manifest_path.read_text('utf-8'))
        for name, key in ((INDEX, 'source_index_sha256'), ('config.json', 'source_config_sha256')):
            if hashlib.sha256((original / name).read_bytes()).hexdigest() != manifest.get(key):
                raise ValueError(f'Existing conversion source mismatch: {destination}')
        if (not manifest.get('complete') or manifest.get('profile', profile) != profile
                or manifest.get('scale_layout') != PACKED_SCALE_LAYOUT):
            raise ValueError(f'Existing conversion profile mismatch: {destination}')
        _verify_conversion(root, destination, manifest)
        return destination
    staged = _inside(root, destination.with_name(destination.name + '.converting'))
    if staged.exists():
        raise ValueError(f'Incomplete staging exists: {staged}; inspect and move it aside before retrying')
    print(f'Converting {original.name} to packed MXFP8 ({profile})', flush=True)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(threads, 8))
        manifest = export_mxfp8(original, staged, profile=profile)
    finally:
        torch.set_num_threads(threads)
    _verify_conversion(root, staged, manifest, written=True)
    staged.rename(destination)
    return destination


def _overlays(root, omni):
    video_vae = _inside(root, root / SHARED.folder / 'vae-ffn-mxfp8.safetensors')
    if not video_vae.exists():
        from .export_vae_mxfp8 import main as export_vae

        export_vae(['--vae', str(root / SHARED.folder / 'vae'), '--output', str(video_vae)])
    if omni is not None:
        adaln = _inside(root, omni.parent / 'adaln-pdd8.safetensors')
        if not adaln.exists():
            from .export_adaln import main as export_adaln

            export_adaln(['--transformer', str(omni), '--output', str(adaln)])


def generation_assets(models_dir, task):
    """Resolve prepared model paths for the public generation commands."""
    root = Path(models_dir).resolve()
    if task not in ('t2va', 'f2va', 'fl2va', 'r2va'):
        raise ValueError(f'Unknown generation task: {task}')
    t2va = task == 't2va'
    return {
        'base_model': root / (T2VA.folder if t2va else OMNI.folder),
        'transformer_path': root / ('t2va-v2-mxfp8-packed/transformer' if t2va else 'mxfp8-packed/transformer_ref'),
        'text_encoder_path': root / ENCODER.folder / 'text_encoder',
        'shared_components': root / SHARED.folder,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fasth3 prepare', description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--task', choices=('t2va', 'f2va', 'fl2va', 'r2va', 'all'), default='all')
    parser.add_argument('--accept-model-license', action='store_true', help=f'Acknowledge {MODEL_LICENSE}')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--plan', action='store_true', help='Show required files and missing bytes without downloading')
    operation.add_argument('--download-only', action='store_true', help='Download verified assets without CUDA or conversion')
    args = parser.parse_args(argv)
    if not args.plan and not args.accept_model_license:
        parser.error(f'Read {MODEL_LICENSE} and pass --accept-model-license before download or conversion')
    from huggingface_hub import HfApi, get_token

    root = args.models_dir.resolve()
    sources = {'t2va': (T2VA,), 'all': (OMNI, T2VA)}.get(args.task, (OMNI,))
    api = HfApi()
    inventory = [(source, _assets(api, source)) for source in (*sources, SHARED, ENCODER)]
    missing = 0
    for source, assets in inventory:
        for asset in assets:
            path = _inside(root, root / source.folder / asset.path)
            if not path.is_file() or path.stat().st_size != asset.size or path.with_name(path.name + '.aria2').exists():
                missing += asset.size
    conversions = {
        OMNI: ('transformer_ref', 'mxfp8-packed', 'omni-pdd8'),
        T2VA: ('transformer', 't2va-v2-mxfp8-packed', 't2va-v2'),
    }
    convert_space = 0
    for source, assets in inventory:
        if source not in conversions:
            continue
        raw_component, output_folder, _ = conversions[source]
        if not _inside(root, root / output_folder / raw_component).exists():
            convert_space += sum(asset.size for asset in assets if asset.path.startswith(f'{raw_component}/'))
    if not _inside(root, root / SHARED.folder / 'vae-ffn-mxfp8.safetensors').exists():
        convert_space += 4 << 30
    if OMNI in sources and not _inside(root, root / 'mxfp8-packed' / 'adaln-pdd8.safetensors').exists():
        convert_space += 1 << 30
    if args.plan:
        for source, assets in inventory:
            print(f'{source.repo}@{source.revision} -> {root / source.folder} ({len(assets)} files)')
        print(f'Missing source downloads: {missing / (1 << 30):.2f} GiB; '
              f'conservative conversion allowance: {convert_space / (1 << 30):.2f} GiB.')
        return 0
    from filelock import FileLock

    curl = shutil.which('curl.exe') or shutil.which('curl')
    if not curl:
        parser.error('Install curl or add it to PATH')
    token = get_token()
    if token and not re.fullmatch(r'[A-Za-z0-9_\-]+', token):
        parser.error('Hugging Face token contains unexpected characters')
    root.mkdir(parents=True, exist_ok=True)
    required = missing + (0 if args.download_only else convert_space) + (1 << 30)
    if shutil.disk_usage(root).free < required:
        parser.error(f'Insufficient disk space for the selected operation and reserve ({required / (1 << 30):.2f} GiB)')
    with FileLock(str(root / '.prepare-weights.lock'), timeout=0):
        for source, assets in inventory:
            _prepare_source(curl, root, source, assets, token)
        if args.download_only:
            return 0
        packed = {source: _convert(root, source, *conversions[source]) for source in sources}
        _overlays(root, packed.get(OMNI))
        for path in packed.values():
            print(f'Prepared {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
