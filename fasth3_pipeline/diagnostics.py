"""Optional real-stage checkpoints; no model weights or synthetic outputs."""
import json
from pathlib import Path
from time import perf_counter
import torch

STAGES = ("preprocess", "encoder", "reference-vae", "first-denoise", "denoise", "decode")


class PipelineDiagnostics:
    def __init__(self, directory=None, stop_after=None, guard=lambda phase: None, *, debug_checks=False):
        if stop_after is not None and stop_after not in STAGES:
            raise ValueError(f"Unknown stop stage: {stop_after}")
        self.directory = Path(directory).resolve() if directory else None
        self.stop_after = stop_after
        self.guard = guard
        self.debug_checks = debug_checks
        self.events = []
        self.forward_count = 0
        self.streaming = {}
        self.started = self.previous_checkpoint = perf_counter()
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, stage, tensors=None, **details):
        self.guard(f"checkpoint:{stage}")
        event = {"stage": stage, "transformer_forwards": self.forward_count, **details}
        nonfinite = []
        if tensors:
            event["tensors"] = {}
            for name, value in tensors.items():
                if value is None:
                    continue
                info = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if self.debug_checks and (value.is_floating_point() or value.is_complex()):
                    info["finite"] = bool(torch.isfinite(value).all().item())
                    if not info["finite"]:
                        nonfinite.append(name)
                event["tensors"][name] = info
        if self.directory and tensors:
            from safetensors.torch import save_file
            # Real tensors only; CPU snapshots never retain any CUDA storage.
            snapshot = {name: value.detach().to("cpu").contiguous().clone()
                        for name, value in tensors.items() if value is not None}
            path = self.directory / f"{len(self.events):02d}-{stage}.safetensors"
            save_file(snapshot, path)
            event["checkpoint"] = str(path)
            del snapshot
        now = perf_counter()
        event.update(elapsed_seconds=now-self.started,
                     stage_elapsed_seconds=now-self.previous_checkpoint)
        self.previous_checkpoint = now
        self.events.append(event)
        if self.directory:
            (self.directory / "stages.json").write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        if nonfinite:
            raise FloatingPointError(f"{stage}: non-finite checkpoint tensors: {', '.join(nonfinite)}")
        return self.stop_after == stage

    def stopped(self, stage):
        return {"stopped_after": stage, "transformer_forwards": self.forward_count}
