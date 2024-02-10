import jax
import flax.linen as nn
import jax.numpy as jnp

from .. import AudioSignal


def l1_loss(y_true: AudioSignal,
            y_pred: AudioSignal,
            reduction='mean',
            attribute='audio_data') -> jnp.ndarray:

    if isinstance(y_true, AudioSignal):
        y_true = getattr(y_true, attribute)
        y_pred = getattr(y_pred, attribute)

    errors = jnp.abs(y_pred - y_true)
    if reduction == 'none':
        return errors
    elif reduction == 'mean':
        return jnp.mean(errors)
    elif reduction == 'sum':
        return jnp.sum(errors)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


def sisdr_loss(y_true: AudioSignal,
               y_pred: AudioSignal,
               scaling: int=True,
               reduction: str='mean',
               zero_mean: int = True,
               clip_min: int=None,
               attribute: str='audio_data'):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    y_true : AudioSignal
        Estimate AudioSignal
    y_pred : AudioSignal
        Reference AudioSignal
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    if isinstance(y_true, AudioSignal):
        y_true = getattr(y_true, attribute)
        y_pred = getattr(y_pred, attribute)

    eps = 1e-8
    # nb, nc, nt
    references = y_true
    estimates = y_pred

    nb = references.shape[0]
    references = references.reshape(nb, 1, -1).transpose(0, 2, 1)
    estimates = estimates.reshape(nb, 1, -1).transpose(0, 2, 1)

    # samples now on axis 1
    if zero_mean:
        mean_reference = references.mean(axis=1, keepdims=True)
        mean_estimate = estimates.mean(axis=1, keepdims=True)
    else:
        mean_reference = 0
        mean_estimate = 0

    _references = references - mean_reference
    _estimates = estimates - mean_estimate

    references_projection = (_references**2).sum(axis=-2) + eps
    references_on_estimates = (_estimates * _references).sum(axis=-2) + eps

    scale = (
        jnp.expand_dims(references_on_estimates / references_projection, 1)
        if scaling
        else 1
    )

    e_true = scale * _references
    e_res = _estimates - e_true

    signal = (e_true**2).sum(axis=1)
    noise = (e_res**2).sum(axis=1)
    sdr = -10 * jnp.log10(signal / noise + eps)

    if clip_min is not None:
        sdr = jnp.clip(sdr, a_min=clip_min)

    if reduction == "mean":
        sdr = sdr.mean()
    elif reduction == "sum":
        sdr = sdr.sum()
    return sdr
