import typing
from typing import List

import jax.numpy as jnp

from .. import AudioSignal
from .. import STFTParams

from audiotools.metrics.distance import l1_loss


def multiscale_stft_loss(y_true: AudioSignal,
                         y_pred: AudioSignal,
                         window_lengths=[2048, 512],
                         loss_fn: typing.Callable = l1_loss,
                         clamp_eps: float = 1e-5,
                         mag_weight: float = 1.0,
                         log_weight: float = 1.0,
                         pow: float = 2.0,
                         match_stride: bool = False,
                         window_type: str = None
                         ):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    y_true : AudioSignal
        Estimate signal
    y_pred : AudioSignal
        Reference signal
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default l1_loss
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Returns
    -------
    jnp.ndarray
        Multi-scale STFT loss.

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    x = y_pred
    y = y_true
    stft_params = [
        STFTParams(
            window_length=w,
            hop_length=w // 4,
            match_stride=match_stride,
            window_type=window_type,
        )
        for w in window_lengths
    ]

    loss = 0.0

    def decibel_loudness(x: AudioSignal) -> jnp.ndarray:
        return jnp.log10(jnp.power(jnp.maximum(x.magnitude, clamp_eps), pow))

    for s in stft_params:
        x.stft(s.window_length, s.hop_length, s.window_type)
        y.stft(s.window_length, s.hop_length, s.window_type)
        loss = loss + log_weight * loss_fn(decibel_loudness(x), decibel_loudness(y))
        loss = loss + mag_weight * loss_fn(x.magnitude, y.magnitude)
    return loss


def mel_spectrogram_loss(y_true: AudioSignal,
                         y_pred: AudioSignal,
                         n_mels=[150, 80],
                         window_lengths=[2048, 512],
                         loss_fn: typing.Callable = l1_loss,
                         clamp_eps: float = 1e-5,
                         mag_weight: float = 1.0,
                         log_weight: float = 1.0,
                         pow: float = 2.0,
                         match_stride: bool = False,
                         mel_fmin=[0.0, 0.0],
                         mel_fmax=[None, None],
                         window_type: str = None,
                         ):
    """Compute distance between mel spectrograms. Can be used in a multi-scale way.

    Parameters
    ----------
    y_true : AudioSignal
        Estimate signal
    y_pred : AudioSignal
        Reference signal
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Returns
    -------
    jnp.ndarray
        Mel loss.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    x = y_pred
    y = y_true

    stft_params = [
        STFTParams(
            window_length=w,
            hop_length=w // 4,
            match_stride=match_stride,
            window_type=window_type,
        )
        for w in window_lengths
    ]

    loss = 0.0
    for n_mels, fmin, fmax, s in zip(n_mels, mel_fmin, mel_fmax, stft_params):
        kwargs = {
            "window_length": s.window_length,
            "hop_length": s.hop_length,
            "window_type": s.window_type,
        }
        x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
        y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

        loss = loss + log_weight * loss_fn(
            jnp.log10(jnp.pow(jnp.maximum(x_mels, clamp_eps), pow)),
            jnp.log10(jnp.pow(jnp.maximum(y_mels, clamp_eps), pow)),
        )
        loss = loss + mag_weight * loss_fn(x_mels, y_mels)
    return loss


def phase_loss(y_true: AudioSignal,
               y_pred: AudioSignal,
               window_length: int = 2048,
               hop_length: int = 512
               ):
    """Computes phase loss between an estimate and a reference signal.

    Parameters
    ----------
    x : AudioSignal
        Estimate signal
    y : AudioSignal
        Reference signal
    window_length : int, optional
        Length of STFT window, by default 2048
    hop_length : int, optional
        Hop length of STFT window, by default 512
    weight : float, optional
        Weight of loss, by default 1.0

    Returns
    -------
    jnp.ndarray
        Phase loss.
    """

    x = y_pred
    y = y_true

    s = STFTParams(window_length, hop_length)

    x.stft(s.window_length, s.hop_length, s.window_type)
    y.stft(s.window_length, s.hop_length, s.window_type)

    # Take circular difference
    diff = x.phase - y.phase
    diff = diff.at[diff < -jnp.pi].set(diff[diff < -jnp.pi] + 2 * jnp.pi)
    diff = diff.at[diff > jnp.pi].set(diff[diff > jnp.pi - 2 * jnp.pi])

    # Scale true magnitude to weights in [0, 1]
    x_min, x_max = x.magnitude.min(), x.magnitude.max()
    weights = (x.magnitude - x_min) / (x_max - x_min)

    # Take weighted mean of all phase errors
    loss = ((weights * diff) ** 2).mean()
    return loss
