import jax
import jax.numpy as jnp
import flax.linen as nn

from ...core import AudioSignal
from ...core import STFTParams
from ...core import util


class SpectralGate(nn.Module):
    """Spectral gating algorithm for noise reduction,
    as in Audacity/Ocenaudio. The steps are as follows:

    1.  An FFT is calculated over the noise audio clip
    2.  Statistics are calculated over FFT of the the noise
        (in frequency)
    3.  A threshold is calculated based upon the statistics
        of the noise (and the desired sensitivity of the algorithm)
    4.  An FFT is calculated over the signal
    5.  A mask is determined by comparing the signal FFT to the
        threshold
    6.  The mask is smoothed with a filter over frequency and time
    7.  The mask is appled to the FFT of the signal, and is inverted

    Implementation inspired by Tim Sainburg's noisereduce:

    https://timsainburg.com/noise-reduction-python.html

    Parameters
    ----------
    n_freq : int, optional
        Number of frequency bins to smooth by, by default 3
    n_time : int, optional
        Number of time bins to smooth by, by default 5
    """

    n_freq: int = 3
    n_time: int = 5

    @nn.compact
    def __call__(
        self,
        audio_signal: AudioSignal,
        nz_signal: AudioSignal,
        denoise_amount: float = 1.0,
        n_std: float = 3.0,
        win_length: int = 2048,
        hop_length: int = 512,
    ):
        """Perform noise reduction.

        Parameters
        ----------
        audio_signal : AudioSignal
            Audio signal that noise will be removed from.
        nz_signal : AudioSignal, optional
            Noise signal to compute noise statistics from.
        denoise_amount : float, optional
            Amount to denoise by, by default 1.0
        n_std : float, optional
            Number of standard deviations above which to consider
            noise, by default 3.0
        win_length : int, optional
            Length of window for STFT, by default 2048
        hop_length : int, optional
            Hop length for STFT, by default 512

        Returns
        -------
        AudioSignal
            Denoised audio signal.
        """
        smoothing_filter = jnp.outer(
            jnp.concatenate(
                [
                    jnp.linspace(0, 1, self.n_freq + 2)[:-1],
                    jnp.linspace(1, 0, self.n_freq + 2),
                ]
            )[..., 1:-1],
            jnp.concatenate(
                [
                    jnp.linspace(0, 1, self.n_time + 2)[:-1],
                    jnp.linspace(1, 0, self.n_time + 2),
                ]
            )[..., 1:-1],
        )
        smoothing_filter = smoothing_filter / smoothing_filter.sum()
        smoothing_filter = jnp.expand_dims(smoothing_filter, (0, 0))

        stft_params = STFTParams(win_length, hop_length, "sqrt_hann")

        audio_signal = audio_signal.clone()
        audio_signal.stft_data = None
        audio_signal.stft_params = stft_params

        nz_signal = nz_signal.clone()
        nz_signal.stft_params = stft_params

        nz_stft_db = 20 * jnp.log10(jnp.maximum(nz_signal.magnitude, 1e-4))
        nz_freq_mean = nz_stft_db.mean(keepdims=True, axis=-1)
        nz_freq_std = nz_stft_db.std(keepdims=True, axis=-1)

        nz_thresh = nz_freq_mean + nz_freq_std * n_std

        stft_db = 20 * jnp.log10(jnp.maximum(audio_signal.magnitude, 1e-4))
        nb, nac, nf, nt = stft_db.shape
        # db_thresh = nz_thresh.expand(nb, nac, -1, nt)
        db_thresh = jnp.tile(nz_thresh, (nb/nz_thresh.shape[0], nac/nz_thresh.shape[1], 1, nt/nz_thresh.shape[3]))  # todo:

        stft_mask = (stft_db < db_thresh).astype(jnp.float32)
        shape = stft_mask.shape

        stft_mask = stft_mask.reshape(nb * nac, 1, nf, nt)
        pad_tuple = (
            smoothing_filter.shape[-2] // 2,
            smoothing_filter.shape[-1] // 2,
        )
        stft_mask = F.conv2d(stft_mask, smoothing_filter, padding=pad_tuple)
        stft_mask = stft_mask.reshape(*shape)
        stft_mask = stft_mask * util.ensure_tensor(denoise_amount, ndim=stft_mask.ndim)
        stft_mask = 1 - stft_mask

        audio_signal.stft_data *= stft_mask
        audio_signal.istft()

        return audio_signal
