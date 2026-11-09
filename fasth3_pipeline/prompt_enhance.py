"""OpenH3-IR prompt compiler adapter for the four FastH3 generation tasks."""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import logging
from pathlib import Path
import math
import os
import re
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.parse import urlsplit


OPEN_H3_IR_VERSION = "0.4.1"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_TIMEOUT = 120.
TASKS = ("t2va", "f2va", "fl2va", "r2va")
_MAX_EDGE = 768
_FRAME_FRACTIONS = (0.1, 0.5, 0.9)
_compile_lock = Lock()


class PromptEnhancementError(RuntimeError):
    """A PE failure safe to report without provider payloads or credentials."""


@dataclass(frozen=True)
class EnhancedPrompt:
    text: str
    mode: str
    source: str
    prompt_tokens: int
    visual_samples: int
    findings: tuple[str, ...]


def _pin_open_h3_ir():
    try:
        version = metadata.version("open-h3-ir")
    except metadata.PackageNotFoundError:
        version = None
    if version != OPEN_H3_IR_VERSION:
        raise PromptEnhancementError(f"Prompt enhancement requires open-h3-ir=={OPEN_H3_IR_VERSION}")


def _api_base(base_url):
    try:
        url = urlsplit(base_url)
        host = url.hostname
    except ValueError:
        raise PromptEnhancementError("PE base URL is invalid") from None
    if (not host or url.username is not None or url.password is not None or url.query or url.fragment
            or not (url.scheme == "https" or url.scheme == "http" and host in {"127.0.0.1", "::1", "localhost"})):
        raise PromptEnhancementError("PE base URL must be HTTPS or loopback HTTP, without embedded credentials")
    return base_url.rstrip("/")


def _deepseek_backend(base_url, model, key, timeout):
    from h3ir.backend import Backend
    from h3ir.config import Config, LLMConfig

    class DeepSeekBackend(Backend):
        def _once(self, body):
            # OpenH3-IR's generic transport uses Qwen/vLLM's thinking toggle.
            # DeepSeek uses a top-level object and does not expose a seed field.
            payload = dict(body)
            thinking = "chat_template_kwargs" not in payload
            payload.pop("chat_template_kwargs", None)
            payload.pop("seed", None)
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
            return super()._once(payload)

        def json_call(self, messages, schema, *, thinking=None, **kwargs):
            # This endpoint validates JSON in the compiler. No server-guided
            # schema is sent alongside reasoning.
            return super().json_call(messages, schema, thinking=True if thinking is None else thinking, **kwargs)

    llm = LLMConfig(base_url=base_url, model=model, api_key=key, timeout_s=timeout,
                    default_thinking=True, guided_decoding=False)
    return DeepSeekBackend(Config(llm=llm))


def _save_sample(image, path):
    from PIL import Image

    frame = image if isinstance(image, Image.Image) else Image.fromarray(image)
    frame = frame.convert("RGB")
    original_size = frame.size
    frame.thumbnail((_MAX_EDGE, _MAX_EDGE), Image.Resampling.LANCZOS)
    frame.save(path, format="JPEG", quality=85)
    return original_size


def _id(slot, paths=()):
    digest = sha256(str(slot).encode("ascii"))
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _assets(task, references, reference_inputs, first_frame, last_frame, export_frames, directory):
    from h3ir.models import AssetKind, AssetRef, Role

    assets, video_samples, audio_ids = [], {}, []
    visual_count = 0

    def image_asset(image, role, slot):
        sample = directory / f"{slot}.jpg"
        px = _save_sample(image, sample)
        return AssetRef(AssetKind.IMAGE, role, _id(slot, (sample,)), path=str(sample), px=px,
                        role_stated=True)

    if task in ("f2va", "fl2va"):
        assets.append(image_asset(first_frame, Role.FRAME_ANCHOR_FIRST, 0))
        if task == "fl2va":
            assets.append(image_asset(last_frame, Role.FRAME_ANCHOR_LAST, 1))
        return assets, video_samples, audio_ids, len(assets)

    for index, (reference, specification) in enumerate(zip(references, reference_inputs, strict=True)):
        if reference.kind == "image":
            assets.append(image_asset(reference.image, Role.SUBJECT, index))
            visual_count += 1
        elif reference.kind == "video":
            fps = float(reference.fps)
            if not math.isfinite(fps) or fps <= 0 or not len(reference.frames):
                raise PromptEnhancementError("PE video reference has an invalid frame count or rate")
            count = min(len(reference.frames), math.ceil(export_frames / 24 * fps))
            indices = sorted({min(count - 1, int(count * fraction)) for fraction in _FRAME_FRACTIONS})
            frames = []
            for offset in indices:
                sample = directory / f"{index}-{offset}.jpg"
                px = _save_sample(reference.frames[offset], sample)
                frames.append(sample)
            asset_id = _id(index, frames)
            assets.append(AssetRef(AssetKind.VIDEO, Role.SUBJECT, asset_id, path=str(specification.path),
                                   px=px, seconds=count / fps, frames=round(count / fps * 24), role_stated=True))
            video_samples[asset_id] = [str(frame) for frame in frames]
            visual_count += len(frames)
            if reference.has_audio:
                audio_ids.append(_id(index + 100))
                assets.append(AssetRef(AssetKind.AUDIO, Role.SFX, audio_ids[-1],
                                       paired_video_sha256=asset_id,
                                       note="Original soundtrack; no transcription or listening performed.",
                                       role_stated=True))
        elif reference.kind == "audio":
            audio_ids.append(_id(index + 100))
            assets.append(AssetRef(AssetKind.AUDIO, Role.SFX, audio_ids[-1],
                                   note="User audio reference; no transcription or listening performed.",
                                   role_stated=True))
    return assets, video_samples, audio_ids, visual_count


@contextmanager
def _compiler_inputs(video_samples, export_frames):
    from h3ir import analyse, compile, plan
    from h3ir.grid import Target

    class ExportTimeline(Target):
        @property
        def effective_seconds(self):
            # Keep Target.frames aligned for OpenH3-IR's token budget, while
            # all prompt durations and cuts use the actually exported length.
            return self.nominal_seconds

    def sample_frames(path, asset_id, fractions=None):
        return video_samples[asset_id]

    with _compile_lock:
        old_target, old_sampler, old_validate = plan.Target, analyse.sample_frames, compile.validate
        logger = logging.getLogger("h3ir")
        old_handlers, old_propagation = logger.handlers, logger.propagate

        def validate(text, context):
            findings = old_validate(text, context)
            if context.duration_s == export_frames / 24:
                # T7 assumes its requested timeline is passed directly to the
                # VAE. FastH3 runs aligned frames and trims only the export.
                return [finding for finding in findings if finding.rule != "T7-illegal-duration"]
            return findings

        plan.Target, analyse.sample_frames, compile.validate = ExportTimeline, sample_frames, validate
        logger.handlers, logger.propagate = [logging.NullHandler()], False
        try:
            yield
        finally:
            logger.handlers, logger.propagate = old_handlers, old_propagation
            plan.Target, analyse.sample_frames, compile.validate = old_target, old_sampler, old_validate


def _runtime_audio_labels(prompt, manifest, audio_ids):
    # OpenH3-IR follows ComfyUI's paired-audio-first convention. Diffusers
    # assigns audio labels in the caller's ordered reference list.
    labels = {asset_id: f"<Audio {i}>" for i, asset_id in enumerate(audio_ids, 1)}
    replacements = {entry.label: labels[entry.sha256] for entry in manifest if entry.sha256 in labels}
    return re.sub(r"<Audio \d+>", lambda match: replacements.get(match[0], match[0]), prompt)


def enhance_prompt(brief, *, task, export_frames, canvas, aspect="16:9",
                   references=(), reference_inputs=(), first_frame=None, last_frame=None,
                   seed=None, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT):
    """Compile the export-length prompt through OpenH3-IR's pinned Python API."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise PromptEnhancementError("Prompt enhancement requires DEEPSEEK_API_KEY")
    if task not in TASKS or not brief.strip() or not 120 <= export_frames <= 360:
        raise PromptEnhancementError("PE needs a supported task, a prompt and 120 to 360 export frames")
    if task == "fl2va" and export_frames % 17 != 5:
        raise PromptEnhancementError("FL2VA PE requires an aligned export length for the closing frame")
    anchor_count = (first_frame is not None, last_frame is not None)
    if anchor_count != (task in ("f2va", "fl2va"), task == "fl2va"):
        raise PromptEnhancementError("PE keyframes do not match the task")
    if bool(references) != (task == "r2va"):
        raise PromptEnhancementError("Only R2VA PE accepts ordered media references")
    if not math.isfinite(timeout) or timeout <= 0:
        raise PromptEnhancementError("PE timeout must be positive and finite")
    if tuple(item.kind for item in references) != tuple(item.kind for item in reference_inputs):
        raise PromptEnhancementError("PE reference input order does not match the decoded references")
    base_url = _api_base(base_url)
    _pin_open_h3_ir()
    from h3ir.compile import compile_brief
    from h3ir.models import Brief, Mode

    modes = {"t2va": Mode.T2VA, "f2va": Mode.I2VA, "fl2va": Mode.FL2VA, "r2va": Mode.REF2VA}
    expected_mode = modes[task]
    try:
        with TemporaryDirectory(prefix="fasth3-pe-") as work:
            assets, samples, audio_ids, visual_count = _assets(
                task, references, reference_inputs, first_frame, last_frame, export_frames, Path(work))
            request = Brief(intent=brief, assets=assets, seconds=export_frames / 24,
                            aspect=aspect, canvas=canvas, mode=expected_mode)
            with _deepseek_backend(base_url, model, key, timeout) as backend, _compiler_inputs(samples, export_frames):
                document = compile_brief(request, backend=backend, seed=seed, use_cache=False,
                                         thinking_planning=True, thinking_prose=True)
    except PromptEnhancementError:
        raise
    except Exception as error:
        # Vendor HTTP errors may contain provider response text or local paths.
        raise PromptEnhancementError(f"OpenH3-IR prompt compilation failed ({type(error).__name__})") from None
    if not document.ok or document.mode != expected_mode or not document.prompt.strip():
        raise PromptEnhancementError("OpenH3-IR returned an invalid prompt or a different task mode")
    text = _runtime_audio_labels(document.prompt, document.plan.manifest, audio_ids)
    return EnhancedPrompt(text, document.mode.value, document.source, document.prompt_tokens, visual_count,
                          tuple(finding.rule for finding in document.findings))
