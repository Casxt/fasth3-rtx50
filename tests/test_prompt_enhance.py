"""CPU contracts for the pinned OpenH3-IR adapter and CLI routing."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from fasth3_pipeline import cli, prompt_enhance as pe


def _video():
    frames = np.zeros((36, 12, 8, 3), dtype=np.uint8)
    return SimpleNamespace(kind="video", has_audio=True, fps=24.0, frames=frames)


def test_missing_key_fails_before_provider_or_import(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(pe.PromptEnhancementError, match="DEEPSEEK_API_KEY"):
        pe.enhance_prompt("A shot.", task="t2va", export_frames=120, canvas=(1344, 768))


def test_deepseek_wire_format_uses_native_thinking_and_image_parts():
    pytest.importorskip("h3ir")
    import httpx

    bodies = []

    def request(req):
        assert str(req.url) == "https://api.deepseek.com/chat/completions"
        assert req.headers["Authorization"] == "Bearer unit-test-key"
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"description":"test"}', "reasoning_content": "placeholder"}}]})

    with pe._deepseek_backend("https://api.deepseek.com", "deepseek-flash", "unit-test-key", 30) as backend:
        backend._client = httpx.Client(transport=httpx.MockTransport(request))
        visual = [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
        result = backend.json_call(visual, {"type": "object"}, required=("description",), seed=42)
        backend.chat(visual, thinking=False)
    assert result["description"] == "test"
    assert [body["thinking"]["type"] for body in bodies] == ["enabled", "disabled"]
    assert bodies[0]["messages"][0]["content"][1]["type"] == "image_url"
    assert all("chat_template_kwargs" not in body and "seed" not in body and "response_format" not in body
               for body in bodies)


def test_openh3_timeline_is_exact_export_duration_with_aligned_budget(monkeypatch):
    pytest.importorskip("h3ir")
    from h3ir import analyse, compile, plan

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-key")
    original = compile.compile_brief
    previous_target, previous_sampler = plan.Target, analyse.sample_frames

    class OfflineBackend:
        cfg = SimpleNamespace(model="offline")
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return None
        def require_available(self):
            return None
        def server_version(self):
            return "offline"

    monkeypatch.setattr(pe, "_deepseek_backend", lambda *_: OfflineBackend())

    def compile_offline(brief, **kwargs):
        assert brief.seconds == 5.0
        assert kwargs["thinking_planning"] and kwargs["thinking_prose"] and not kwargs["use_cache"]
        doc = original(brief, llm=False, **kwargs)
        assert doc.plan.target.frames == 124
        assert doc.plan.target.effective_seconds == 5.0
        return doc

    monkeypatch.setattr(compile, "compile_brief", compile_offline)
    result = pe.enhance_prompt("A lighthouse keeper lights the lamp in a storm.",
                               task="t2va", export_frames=120, canvas=(1344, 768))
    assert result.mode == "t2va" and result.prompt_tokens > 0
    assert plan.Target is previous_target and analyse.sample_frames is previous_sampler


@pytest.mark.parametrize(("task", "mode", "roles"), [
    ("t2va", "t2va", []),
    ("f2va", "i2va", ["frame_anchor_first"]),
    ("fl2va", "fl2va", ["frame_anchor_first", "frame_anchor_last"]),
    ("r2va", "ref2va", ["sfx", "subject", "sfx"]),
])
def test_openh3_explicit_modes_roles_and_runtime_audio_renumbering(task, mode, roles, monkeypatch, tmp_path):
    pytest.importorskip("h3ir")
    from h3ir import analyse, compile
    from h3ir.plan import build_manifest, Target

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-key")
    from contextlib import nullcontext
    monkeypatch.setattr(pe, "_deepseek_backend", lambda *_: nullcontext())
    original_source = tmp_path / "private.mp4"

    def compile_stub(brief, **kwargs):
        assert brief.mode.value == mode
        assert [asset.role.value for asset in brief.assets] == roles
        assert all(asset.role_stated for asset in brief.assets)
        assert kwargs["thinking_planning"] and kwargs["thinking_prose"]
        manifest = build_manifest(brief, Target.build(brief.seconds, brief.aspect, brief.canvas))
        if task == "r2va":
            video = brief.assets[1]
            frames = analyse.sample_frames(str(original_source), video.sha256)
            assert len(frames) == 3 and all(Path(frame).is_file() for frame in frames)
            assert brief.assets[2].paired_video_sha256 == video.sha256
            assert brief.assets[0].path is None and brief.assets[2].path is None
        text = "<Audio 1> with <Video 1>, then <Audio 2>" if task == "r2va" else "A complete prompt"
        return SimpleNamespace(mode=brief.mode, plan=SimpleNamespace(manifest=manifest), prompt=text,
                               ok=True, source="written", prompt_tokens=12, findings=[])

    monkeypatch.setattr(compile, "compile_brief", compile_stub)
    image = Image.new("RGB", (12, 8))
    audio = SimpleNamespace(kind="audio", has_audio=True)
    references = (audio, _video()) if task == "r2va" else ()
    specs = (cli.ReferenceInput("audio", tmp_path / "private.wav"),
             cli.ReferenceInput("video", original_source)) if task == "r2va" else ()
    export_frames = 124 if task == "fl2va" else 120
    result = pe.enhance_prompt("A complete brief", task=task, export_frames=export_frames, canvas=(1344, 768),
                               references=references, reference_inputs=specs,
                               first_frame=image if task in ("f2va", "fl2va") else None,
                               last_frame=image if task == "fl2va" else None)
    assert result.mode == mode
    if task == "r2va":
        assert result.text == "<Audio 2> with <Video 1>, then <Audio 1>"
        assert result.visual_samples == 3


def test_standalone_pe_only_saves_without_loading_weights_or_logging_prompt(monkeypatch, tmp_path, capsys):
    from tools.rewrite_prompt import main

    output = tmp_path / "prompt.txt"
    prompt = "Private synthetic test prompt"
    rendered = "Private synthetic enhancement"

    def enhance(brief, **kwargs):
        assert brief == prompt
        assert kwargs["task"] == "t2va" and kwargs["export_frames"] == 120
        return pe.EnhancedPrompt(rendered, "t2va", "written", 2, 0, ())

    monkeypatch.setattr(pe, "enhance_prompt", enhance)
    assert main(["t2va", "--output", str(output), "--prompt", prompt, "--num-frames", "120"]) == 0
    assert output.read_text(encoding="utf-8") == rendered
    receipt = json.loads(output.with_name("prompt.txt.pe.json").read_text(encoding="utf-8"))
    assert receipt["prompt_enhancement"]["thinking"] is True
    assert (receipt["num_frames"], receipt["aligned_num_frames"], receipt["duration_seconds"]) == (120, 124, 5.)
    printed = capsys.readouterr().out
    assert prompt not in printed and rendered not in printed
