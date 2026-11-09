"""Installed FastH3 command-line entry point."""
import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

from .prompt_enhance import TASKS


@dataclass(frozen=True)
class ReferenceInput:
    kind: str
    path: Path

    @property
    def runner_spec(self):
        return f"{self.kind}:{self.path}"


def _reference_input(value):
    kind, sep, path = value.partition(":")
    if not sep or kind not in ("image", "video", "audio") or not path:
        raise argparse.ArgumentTypeError("Reference must be image:PATH, video:PATH or audio:PATH")
    return ReferenceInput(kind, Path(path))


def _request_arguments(parser, task):
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path, help="Read the exact prompt from a UTF-8 file")
    parser.add_argument("--num-frames", type=int, default=124, help="Exported frame count, 120 to 360 at 24 fps")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--aspect-ratio", default="16:9", help="Released 768p aspect ratio")
    if task in ("f2va", "fl2va"):
        parser.add_argument("--first-frame", type=Path, required=True, help="Opening-frame image")
    if task == "fl2va":
        parser.add_argument("--last-frame", type=Path, required=True, help="Closing-frame image")
    if task == "r2va":
        parser.add_argument("--reference", type=_reference_input, required=True, action="append",
                            help="Ordered image:PATH, video:PATH or audio:PATH (repeatable)")


def _validate_frame_count(parser, args):
    if not 120 <= args.num_frames <= 360:
        parser.error("--num-frames must be between 120 and 360")
    if args.task == "fl2va" and args.num_frames % 17 != 5:
        parser.error("FL2VA requires an aligned 17n+5 frame count (e.g. 124 or 192) to retain the closing-frame time")


def _prepare(argv):
    from .prepare_weights import main

    return main(argv)


def _doctor(argv):
    from .doctor import main

    return main(argv)


def _model_assets(models_dir, task):
    from .prepare_weights import generation_assets

    return generation_assets(models_dir, task)


def _read_prompt(args):
    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            raise ValueError("--prompt-file must be a readable UTF-8 file") from None
    else:
        prompt = args.prompt
    if not prompt.strip():
        raise ValueError("Prompt must contain text")
    return prompt


def _open_frame(path):
    from PIL import Image, ImageOps

    if path is None:
        return None
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _input_media(args, task, canvas):
    from .runner import load_references

    specs = tuple(getattr(args, "reference", ()) or ())
    refs = ()
    if task == "r2va":
        counts = Counter(ref.kind for ref in specs)
        if not (counts["image"] or counts["video"]):
            raise ValueError("R2VA requires an image or video reference")
        if len(specs) > 12 or counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
            raise ValueError("R2VA permits up to 9 images, 3 videos, 3 audio files and 12 total references")
        refs = load_references([ref.runner_spec for ref in specs], video_canvas=canvas,
                               video_frames=args.num_frames)
    return specs, refs, _open_frame(getattr(args, "first_frame", None)), _open_frame(getattr(args, "last_frame", None))


def _rewrite(prompt, *, task, args, canvas, specs, refs, first, last, receipt, **provider):
    from .prompt_enhance import DEFAULT_MODEL, OPEN_H3_IR_VERSION, enhance_prompt

    started = perf_counter()
    receipt.update(compiler=f"OpenH3-IR {OPEN_H3_IR_VERSION}", model=provider.get("model", DEFAULT_MODEL),
                   vision=True, thinking=True, status="started")
    try:
        enhanced = enhance_prompt(prompt, task=task, export_frames=args.num_frames, canvas=canvas,
                                  aspect=args.aspect_ratio, references=refs, reference_inputs=specs,
                                  first_frame=first, last_frame=last, seed=args.seed, **provider)
    except Exception:
        receipt["status"] = "failed"
        raise
    else:
        receipt.update(status="complete", source=enhanced.source, mode=enhanced.mode,
                       visual_samples=enhanced.visual_samples, prompt_tokens=enhanced.prompt_tokens,
                       findings=enhanced.findings, prompt_sha256=sha256(enhanced.text.encode("utf-8")).hexdigest())
        return enhanced.text
    finally:
        receipt["elapsed_seconds"] = perf_counter() - started


def _canvas(aspect):
    from .canvas import released_canvas

    return released_canvas(aspect)


def _run_generation(args):
    from fasth3_platform import BudgetGuard
    from .diagnostics import PipelineDiagnostics
    from .long_video import aligned_video_frames
    from .runner import run_h3, save_result

    started = perf_counter()
    prompt = _read_prompt(args)
    height, width, contract = _canvas(args.aspect_ratio)
    assets = _model_assets(args.models_dir, args.task)
    aligned = aligned_video_frames(args.num_frames)
    output = args.output.resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise ValueError("Output video or its metadata already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    guard = BudgetGuard(device="cuda:0", max_gib=10., reserve_mib=256)
    diagnostics = PipelineDiagnostics(guard=guard, debug_checks=args.debug_checks)
    reference_options = ({"reference_image_short_edge": args.reference_image_short_edge}
                         if args.task == "r2va" else {})
    report = {"status": "started", "task": args.task, "output": None, "seed": args.seed,
              "width": width, "height": height, "fps": 24, "num_frames": args.num_frames,
              "aligned_num_frames": aligned, "duration_seconds": args.num_frames / 24,
              "aligned_duration_seconds": aligned / 24, "canvas_contract": contract,
              "debug_checks": args.debug_checks, **reference_options}
    try:
        with guard:
            video_canvas = tuple(contract[name] for name in ("canvas_multiple", "canvas_short_edge", "canvas_max_pixels"))
            specs, refs, first, last = _input_media(args, args.task, video_canvas)
            guard("references decoded")
            if args.pe:
                report["prompt_enhancement"] = {}
                prompt = _rewrite(prompt, task=args.task, args=args, canvas=(width, height),
                                  specs=specs, refs=refs, first=first, last=last,
                                  receipt=report["prompt_enhancement"])
            result = run_h3(task=args.task, first_frame=first, last_frame=last, prompt=prompt,
                            references=refs, guard=guard, diagnostics=diagnostics, device="cuda:0",
                            height=height, width=width, num_frames=args.num_frames, seed=args.seed,
                            canvas_config=contract, debug_checks=args.debug_checks,
                            **reference_options, **assets)
            report.update(status="complete", output=str(save_result(result, output, num_frames=args.num_frames)))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(transformer_forwards=diagnostics.forward_count, stages=diagnostics.events,
                      streaming=diagnostics.streaming, guard_samples=guard.samples,
                      elapsed_seconds=perf_counter() - started)
        if "model" in diagnostics.streaming:
            report["model"] = diagnostics.streaming["model"]
        metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("guard_samples", "streaming")}, ensure_ascii=False))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare":
        return _prepare(argv[1:])
    if argv and argv[0] == "doctor":
        return _doctor(argv[1:])

    parser = argparse.ArgumentParser(prog="fasth3", description="Local FastH3 video and stereo audio generation")
    commands = parser.add_subparsers(dest="task", required=True)
    descriptions = {
        "t2va": "Generate from a text prompt",
        "f2va": "Generate from a first frame",
        "fl2va": "Generate between first and last frames",
        "r2va": "Generate from ordered media references",
    }
    for task in TASKS:
        command = commands.add_parser(task, help=descriptions[task], description=descriptions[task])
        _request_arguments(command, task)
        command.add_argument("--models-dir", required=True, type=Path, help="Root prepared by fasth3 prepare")
        command.add_argument("--output", required=True, type=Path, help="Output MP4 path")
        command.add_argument("--pe", action="store_true",
                             help="Enhance the prompt with DeepSeek Flash and bounded visual samples")
        command.add_argument("--debug-checks", action="store_true",
                             help="Enable deep numerical checks on weights, activations and stages")
        if task == "r2va":
            command.add_argument("--reference-image-short-edge", type=int, default=256,
                                 help="Reference image short edge for local conditioning (default: 256)")
    commands.add_parser("prepare", help="Download and prepare local weights for a task")
    commands.add_parser("doctor", help="Check the local environment")
    args = parser.parse_args(argv)
    _validate_frame_count(parser, args)
    if args.task == "r2va" and args.reference_image_short_edge <= 0:
        parser.error("--reference-image-short-edge must be positive")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must name an MP4 file")
    try:
        return _run_generation(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"fasth3: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
