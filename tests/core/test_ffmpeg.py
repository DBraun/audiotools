import shlex
import subprocess
import tempfile

import numpy as np
import pyloudnorm
import pytest
import torch

from audiotools import AudioSignal


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
def test_ffmpeg_resample(sample_rate):
    array = np.random.randn(4, 2, 16000)
    sr = 16000

    signal = AudioSignal(array, sample_rate=sr)

    signal = signal.ffmpeg_resample(sample_rate)
    assert signal.sample_rate == sample_rate
    assert signal.signal_length == sample_rate


def test_ffmpeg_loudness():
    np.random.seed(0)
    array = np.random.randn(16, 2, 16000)
    array /= np.abs(array).max()

    gains = np.random.rand(array.shape[0])[:, None, None]
    array = array * gains

    meter = pyloudnorm.Meter(16000)
    py_loudness = [meter.integrated_loudness(array[i].T) for i in range(array.shape[0])]

    ffmpeg_loudness_iso = AudioSignal(array, 16000).ffmpeg_loudness()
    assert np.allclose(py_loudness, ffmpeg_loudness_iso, atol=1)

    # if you normalize and then write, it should still work.
    # if ffmpeg is float64, this fails
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        x = AudioSignal(torch.randn(44100 * 10), 44100)
        x.ffmpeg_loudness(-24)
        x.normalize(-24)
        x.write(f.name)


def test_ffmpeg_load():
    audio_path = "tests/audio/spk/f10_script4_produced.wav"
    # convert to mp3 with ffmpeg
    og_signal = AudioSignal(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        command = f"ffmpeg -i {audio_path} {f.name} -y -hide_banner -loglevel error"
        subprocess.check_call(shlex.split(command))

        signal_from_ffmpeg = AudioSignal.load_from_file_with_ffmpeg(f.name)
        assert og_signal.signal_length == signal_from_ffmpeg.signal_length