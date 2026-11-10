import json

from PIL import Image
import pytest

from fasth3_pipeline import cli, runner
from fasth3_pipeline.prepare_weights import OMNI_MODEL, V2_MODEL


@pytest.mark.parametrize(("task", "media", "model"), [
    ("t2va", [], V2_MODEL), ("f2va", ["--first-frame", "first.png"], OMNI_MODEL),
    ("fl2va", ["--first-frame", "first.png", "--last-frame", "last.png"], V2_MODEL),
    ("fl2va", ["--first-frame", "first.png", "--last-frame", "last.png", "--model", OMNI_MODEL], OMNI_MODEL),
    ("r2va", ["--reference", "video:clip.mp4"], OMNI_MODEL),
])
@pytest.mark.parametrize("debug_checks", [False, True])
def test_public_generation_preserves_task_roles_and_export_count(task, media, model, debug_checks, monkeypatch, tmp_path):
    import fasth3_platform

    export_frames = 124 if task == "fl2va" else 120

    class FakeGuard:
        def __init__(self, **kwargs):
            assert kwargs == {"device": "cuda:0", "max_gib": 10., "reserve_mib": 256}
            self.samples = []
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return None
        def __call__(self, phase):
            pass

    captured = {}
    monkeypatch.chdir(tmp_path)
    Image.new("L", (16, 8), color=3).save("first.png")
    Image.new("RGBA", (16, 8), color=(2, 3, 4, 255)).save("last.png")
    monkeypatch.setattr(fasth3_platform, "BudgetGuard", FakeGuard)

    def load(items, **kwargs):
        assert items == ["video:clip.mp4"]
        assert kwargs["video_frames"] == 120
        return ["decoded video"]

    def run(**kwargs):
        captured.update(kwargs)
        return {"test": True}

    def save(result, output, *, num_frames):
        assert result == {"test": True} and num_frames == export_frames
        return output

    monkeypatch.setattr(runner, "load_references", load)
    monkeypatch.setattr(runner, "run_h3", run)
    monkeypatch.setattr(runner, "save_result", save)
    output = tmp_path / "out.mp4"
    args = [task, "--models-dir", str(tmp_path / "models"), "--prompt", "brief", "--num-frames", str(export_frames),
            "--output", str(output), *media]
    if debug_checks:
        args.append("--debug-checks")
        if task == "r2va":
            args.extend(["--reference-image-short-edge", "768"])
    assert cli.main(args) == 0
    assert captured["task"] == task
    assert captured["model_source"] == model
    assert captured["transformer_path"].name == ("transformer" if model == V2_MODEL else "transformer_ref")
    assert captured["num_frames"] == export_frames
    assert (captured["height"], captured["width"]) == (768, 1344)
    assert captured["debug_checks"] is debug_checks
    assert captured["diagnostics"].debug_checks is debug_checks
    if task == "r2va":
        assert captured["references"] == ["decoded video"]
        assert captured["reference_image_short_edge"] == (768 if debug_checks else 256)
    else:
        assert not captured["references"]
        assert "reference_image_short_edge" not in captured
    assert (captured["first_frame"] is not None) == (task in ("f2va", "fl2va"))
    assert (captured["last_frame"] is not None) == (task == "fl2va")
    if captured["first_frame"] is not None:
        assert captured["first_frame"].mode == "RGB"
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert (report["num_frames"], report["aligned_num_frames"], report["duration_seconds"]) == (
        export_frames, 124, export_frames / 24)
    assert report["debug_checks"] is debug_checks
    assert report.get("reference_image_short_edge") == captured.get("reference_image_short_edge")


def test_public_cli_rejects_incompatible_media_and_tuning_flags(tmp_path):
    base = ["--models-dir", str(tmp_path), "--prompt", "brief", "--output", str(tmp_path/"result.mp4")]
    with pytest.raises(SystemExit):
        cli.main(["t2va", *base, "--reference", "image:plate.png"])
    with pytest.raises(SystemExit):
        cli.main(["fl2va", *base, "--first-frame", "plate.png"])
    with pytest.raises(SystemExit):
        cli.main(["r2va", *base, "--reference", "image:plate.png", "--offload-mode", "leaf"])
    with pytest.raises(SystemExit):
        cli.main(["f2va", *base, "--first-frame", "plate.png", "--reference-image-short-edge", "768"])
    with pytest.raises(SystemExit):
        cli.main(["r2va", *base, "--reference", "image:plate.png", "--reference-image-short-edge", "0"])


def test_fl2va_rejects_trimming_away_the_closing_frame(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["fl2va", "--models-dir", str(tmp_path), "--prompt", "brief", "--num-frames", "120",
                  "--first-frame", "first.png", "--last-frame", "last.png", "--output", str(tmp_path/"result.mp4")])
    assert "retain the closing-frame time" in capsys.readouterr().err
