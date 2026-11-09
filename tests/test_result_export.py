"""Validate the actual final exporter independently of the expensive model."""
import numpy as np
import pytest
import torch
from fasth3_pipeline.runner import save_result


def test_lightweight_video_processor_matches_h3_spec_and_preserves_unit_range():
    from fasth3_pipeline.runner import configure_lightweight_components
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep
    class Sink:
        def update_components(self, **values):
            vars(self).update(values)
    pipe = Sink()
    configure_lightweight_components(pipe)
    spec = next(s for s in MiniMaxH3VideoDecodeStep().expected_components if s.name == 'video_processor')
    for key, value in spec.config.items():
        assert getattr(pipe.video_processor.config, key) == value
    # Decoder has already recovered [0,1]; black must stay black, not become .5.
    video = torch.tensor([0., .25, 1.]).reshape(1, 3, 1, 1, 1)
    actual = pipe.video_processor.postprocess_video(video, output_type='np')
    np.testing.assert_array_equal(actual.reshape(-1), [0., .25, 1.])


def tiny_result():
    frames = np.zeros((1, 12, 16, 16, 3), dtype=np.float32)
    frames[:, :, :, :, 1] = np.linspace(0, 1, 12)[None, :, None, None]
    return {'videos': frames, 'audio': torch.zeros(1, 2, 16000), 'sampling_rate': 32000}


@pytest.mark.parametrize('kind', ['video', 'audio'])
def test_export_rejects_nonfinite_before_writing(tmp_path, kind):
    result = tiny_result()
    if kind == 'video':
        result['videos'][0, 0, 0, 0, 0] = np.nan
    else:
        result['audio'][0, 0, 0] = float('inf')
    output = tmp_path / 'must-not-exist.mp4'
    with pytest.raises(FloatingPointError, match=f'decoded {kind}'):
        save_result(result, output)
    assert not output.exists()


def test_actual_short_mp4_has_stereo_audio_and_video(tmp_path):
    import av
    output = save_result(tiny_result(), tmp_path / 'synthetic.mp4')
    assert output.stat().st_size > 0
    with av.open(str(output)) as media:
        stream = media.streams.video[0]
        assert (stream.width, stream.height) == (16, 16)
        assert float(stream.average_rate) == 24
        audio = media.streams.audio[0]
        assert audio.rate == 32000
        assert len(audio.layout.channels) == 2
        decoded = list(media.decode(video=0))
        assert len(decoded) == 12
        assert decoded[-1].to_ndarray(format="rgb24")[8, 8, 1] > decoded[0].to_ndarray(format="rgb24")[8, 8, 1]
    with av.open(str(output)) as media:
        assert sum(frame.samples for frame in media.decode(audio=0)) >= 16000


def test_trimmed_long_export_keeps_all_requested_frames(tmp_path):
    import av

    result = {'videos': np.zeros((1, 362, 16, 16, 3), dtype=np.uint8),
              'audio': torch.zeros(1, 2, 482400), 'sampling_rate': 32000}
    output = save_result(result, tmp_path / 'trimmed.mp4', num_frames=360)
    with av.open(str(output)) as media:
        assert len(list(media.decode(video=0))) == 360
        audio = media.streams.audio[0]
        assert float(audio.duration * audio.time_base) == pytest.approx(15, abs=.001)
