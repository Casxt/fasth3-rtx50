"""Compile a FastH3 prompt with OpenH3-IR without loading model weights."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fasth3_pipeline.cli import _canvas, _input_media, _read_prompt, _request_arguments, _rewrite, _validate_frame_count
from fasth3_pipeline.prompt_enhance import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT, TASKS


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Older reference-only callers supplied the options directly.
    if argv and argv[0].startswith("--") and argv[0] != "--help":
        argv.insert(0, "r2va")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="task", required=True)
    for task in TASKS:
        command = commands.add_parser(task)
        _request_arguments(command, task)
        command.add_argument("--output", "--pe-output", dest="output", type=Path, required=True)
        command.add_argument("--pe-model", default=DEFAULT_MODEL)
        command.add_argument("--pe-base-url", default=DEFAULT_BASE_URL)
        command.add_argument("--pe-timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)
    _validate_frame_count(parser, args)
    output = args.output.resolve()
    receipt_path = output.with_name(output.name + ".pe.json")
    if output.exists() or receipt_path.exists():
        parser.error("Prompt output or its metadata already exists")

    from fasth3_platform import BudgetGuard
    from fasth3_pipeline.long_video import aligned_video_frames

    receipt = {}
    try:
        height, width, _ = _canvas(args.aspect_ratio)
        prompt = _read_prompt(args)
        with BudgetGuard(device="cpu") as guard:
            specs, refs, first, last = _input_media(args, args.task, (32, 320, 320 * 576))
            guard("references decoded")
            enhanced = _rewrite(prompt, task=args.task, args=args, canvas=(width, height),
                                specs=specs, refs=refs, first=first, last=last, receipt=receipt,
                                base_url=args.pe_base_url, model=args.pe_model, timeout=args.pe_timeout)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as file:
            file.write(enhanced)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FastH3 prompt enhancement: {error}", file=sys.stderr)
        return 1
    report = {"status": "complete", "task": args.task, "output": str(output),
              "num_frames": args.num_frames, "aligned_num_frames": aligned_video_frames(args.num_frames),
              "duration_seconds": args.num_frames / 24, "prompt_enhancement": receipt}
    with receipt_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
