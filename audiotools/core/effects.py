import typing

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jax import lax
import math

from typing import Optional

from . import util


def sinc(x):
    """Sinc function implemented in JAX."""
    return jnp.sinc(x / jnp.pi)


class ResampleFrac(nn.Module):
    old_sr: int
    new_sr: int
    zeros: int = 24
    rolloff: float = 0.945

    @nn.compact
    def __call__(self, x, output_length: Optional[int] = None, full: bool = False):
        gcd = math.gcd(self.old_sr, self.new_sr)
        old_sr = self.old_sr // gcd
        new_sr = self.new_sr // gcd
        zeros = self.zeros
        rolloff = self.rolloff

        if old_sr == new_sr:
            return x

        sr = min(new_sr, old_sr) * rolloff
        _width = math.ceil(zeros * old_sr / sr)
        idx = jnp.arange(-_width, _width + old_sr)
        kernels = []
        for i in range(new_sr):
            t = (-i / new_sr + idx / old_sr) * sr
            t = jnp.clip(t, -zeros, zeros)
            t *= jnp.pi
            window = jnp.cos(t / zeros / 2) ** 2
            kernel = sinc(t) * window
            kernel /= kernel.sum()
            kernels.append(kernel)
        kernel = jnp.stack(kernels).reshape((new_sr, 1, -1))

        # Padding and convolution in JAX
        def apply_kernel(batch):
            return lax.conv_general_dilated(batch[:, None, :], kernel, (old_sr,),
                                            ((int(_width), int(_width + old_sr)),))

        y = jax.vmap(apply_kernel)(x)
        y = y.reshape(x.shape[:-1] + (-1,))

        float_output_length = new_sr * x.shape[-1] / old_sr
        max_output_length = jnp.ceil(float_output_length).astype(int)
        default_output_length = jnp.floor(float_output_length).astype(int)

        if output_length is None:
            applied_output_length = max_output_length if full else default_output_length
        elif output_length < 0 or output_length > max_output_length:
            raise ValueError(f"output_length must be between 0 and {max_output_length}")
        else:
            applied_output_length = output_length
            if full:
                raise ValueError("You cannot pass both full=True and output_length")

        return y[..., :applied_output_length]


class EffectMixin:
    GAIN_FACTOR = jnp.log(10) / 20
    """Gain factor for converting between amplitude and decibels."""
    CODEC_PRESETS = {
        "8-bit": {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
        "GSM-FR": {"format": "gsm"},
        "MP3": {"format": "mp3", "compression": -9},
        "Vorbis": {"format": "vorbis", "compression": -1},
        "Ogg": {
            "format": "ogg",
            "compression": -1,
        },
        "Amr-nb": {"format": "amr-nb"},
    }
    """Presets for applying codecs via torchaudio."""

    def mix(
        self,
        other,
        snr: typing.Union[jnp.ndarray, float] = 10,
        other_eq: typing.Union[jnp.ndarray] = None,
    ):
        """Mixes noise with signal at specified
        signal-to-noise ratio. Optionally, the
        other signal can be equalized in-place.


        Parameters
        ----------
        other : AudioSignal
            AudioSignal object to mix with.
        snr : typing.Union[torch.Tensor, np.ndarray, float], optional
            Signal to noise ratio, by default 10
        other_eq : typing.Union[torch.Tensor, np.ndarray], optional
            EQ curve to apply to other signal, if any, by default None

        Returns
        -------
        AudioSignal
            In-place modification of AudioSignal.
        """
        snr = util.ensure_tensor(snr)

        pad_len = max(0, self.signal_length - other.signal_length)
        other.zero_pad(0, pad_len)
        other.truncate_samples(self.signal_length)
        if other_eq is not None:
            other = other.equalizer(other_eq)

        tgt_loudness = self.loudness() - snr
        other = other.normalize(tgt_loudness)

        self.audio_data = self.audio_data + other.audio_data
        return self

    def apply_ir(
        self,
        ir,
        drr: typing.Union[jnp.ndarray, float] = None,
        ir_eq: typing.Union[jnp.ndarray] = None,
        use_original_phase: bool = False,
    ):
        """Applies an impulse response to the signal. If ` is`ir_eq``
        is specified, the impulse response is equalized before
        it is applied, using the given curve.

        Parameters
        ----------
        ir : AudioSignal
            Impulse response to convolve with.
        drr : typing.Union[torch.Tensor, np.ndarray, float], optional
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None
        ir_eq : typing.Union[torch.Tensor, np.ndarray], optional
            Equalization that will be applied to impulse response
            if specified, by default None
        use_original_phase : bool, optional
            Whether to use the original phase, instead of the convolved
            phase, by default False

        Returns
        -------
        AudioSignal
            Signal with impulse response applied to it
        """
        if ir_eq is not None:
            ir = ir.equalizer(ir_eq)
        if drr is not None:
            ir = ir.alter_drr(drr)

        # Save the peak before
        max_spk = jnp.abs(self.audio_data).max(axis=-1, keepdims=True)

        # Augment the impulse response to simulate microphone effects
        # and with varying direct-to-reverberant ratio.
        phase = self.phase
        self.convolve(ir)

        # Use the input phase
        if use_original_phase:
            self.stft()
            self.stft_data = self.magnitude * jnp.exp(1j * phase)
            self.istft()

        # Rescale to the input's amplitude
        max_transformed = jnp.abs(self.audio_data).max(axis=-1, keepdims=True)
        scale_factor = jnp.maximum(max_spk, 1e-8) / jnp.maximum(max_transformed, 1e-8)
        self = self * scale_factor

        return self

    def ensure_max_of_audio(self, max: float = 1.0):
        """Ensures that ``abs(audio_data) <= max``.

        Parameters
        ----------
        max : float, optional
            Max absolute value of signal, by default 1.0

        Returns
        -------
        AudioSignal
            Signal with values scaled between -max and max.
        """
        peak = jnp.abs(self.audio_data).max(axis=-1, keepdims=True)
        peak_gain = jnp.ones_like(peak)
        peak_gain = peak_gain.at[peak > max].set(max / peak[peak > max])
        self.audio_data = self.audio_data * peak_gain
        return self

    def normalize(self, db: typing.Union[jnp.ndarray, float] = -24.0):
        """Normalizes the signal's volume to the specified db, in LUFS.
        This is GPU-compatible, making for very fast loudness normalization.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray, float], optional
            Loudness to normalize to, by default -24.0

        Returns
        -------
        AudioSignal
            Normalized audio signal.
        """
        db = util.ensure_tensor(db)
        ref_db = self.loudness()
        gain = db - ref_db
        gain = jnp.exp(gain * self.GAIN_FACTOR)

        self.audio_data = self.audio_data * gain[:, None, None]
        return self


class ImpulseResponseMixin:
    """These functions are generally only used with AudioSignals that are derived
    from impulse responses, not other sources like music or speech. These methods
    are used to replicate the data augmentation described in [1].

    1.  Bryan, Nicholas J. "Impulse response data augmentation and deep
        neural networks for blind room acoustic parameter estimation."
        ICASSP 2020-2020 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2020.
    """

    def decompose_ir(self):
        """Decomposes an impulse response into early and late
        field responses.
        """
        # Equations 1 and 2
        # -----------------
        # Breaking up into early
        # response + late field response.

        td = jnp.argmax(self.audio_data, axis=-1, keepdims=True)
        t0 = int(self.sample_rate * 0.0025)

        idx = jnp.arange(self.audio_data.shape[-1])[None, None, :]
        idx = idx.expand(self.batch_size, -1, -1)
        early_idx = (idx >= td - t0) * (idx <= td + t0)

        early_response = jnp.zeros_like(self.audio_data)
        early_response = early_response.at[early_idx].set(self.audio_data[early_idx])

        late_idx = ~early_idx
        late_field = jnp.zeros_like(self.audio_data)
        late_field = late_field.at[late_idx].set(self.audio_data[late_idx])

        # Equation 4
        # ----------
        # Decompose early response into windowed
        # direct path and windowed residual.

        window = jnp.zeros_like(self.audio_data)
        for idx in range(self.batch_size):
            window_idx = early_idx[idx, 0].nonzero()
            window = window.at[idx, ..., window_idx].set(self.get_window("hann", window_idx.shape[-1]))
        return early_response, late_field, window

    def measure_drr(self):
        """Measures the direct-to-reverberant ratio of the impulse
        response.

        Returns
        -------
        float
            Direct-to-reverberant ratio
        """
        early_response, late_field, _ = self.decompose_ir()
        num = (early_response**2).sum(axis=-1)
        den = (late_field**2).sum(axis=-1)
        drr = 10 * jnp.log10(num / den)
        return drr

    @staticmethod
    def solve_alpha(early_response, late_field, wd, target_drr):
        """Used to solve for the alpha value, which is used
        to alter the drr.
        """
        # Equation 5
        # ----------
        # Apply the good ol' quadratic formula.

        wd_sq = wd**2
        wd_sq_1 = (1 - wd) ** 2
        e_sq = early_response**2
        l_sq = late_field**2
        a = (wd_sq * e_sq).sum(axis=-1)
        b = (2 * (1 - wd) * wd * e_sq).sum(axis=-1)
        c = (wd_sq_1 * e_sq).sum(axis=-1) - jnp.power(10, target_drr / 10) * l_sq.sum(
            axis=-1
        )

        expr = ((b**2) - 4 * a * c).sqrt()
        alpha = jnp.maximum(
            (-b - expr) / (2 * a),
            (-b + expr) / (2 * a),
        )
        return alpha

    def alter_drr(self, drr: typing.Union[jnp.ndarray, np.ndarray, float]):
        """Alters the direct-to-reverberant ratio of the impulse response.

        Parameters
        ----------
        drr : typing.Union[torch.Tensor, np.ndarray, float]
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None

        Returns
        -------
        AudioSignal
            Altered impulse response.
        """
        drr = util.ensure_tensor(drr, 2, self.batch_size)

        early_response, late_field, window = self.decompose_ir()
        alpha = self.solve_alpha(early_response, late_field, window, drr)
        min_alpha = (
            jnp.abs(late_field).max(axis=-1) / jnp.abs(early_response).max(axis=-1)
        )
        alpha = jnp.maximum(alpha, min_alpha)[..., None]

        aug_ir_data = (
            alpha * window * early_response
            + ((1 - window) * early_response)
            + late_field
        )
        self.audio_data = aug_ir_data
        self.ensure_max_of_audio()
        return self


if __name__ == '__main__':

    # Example usage:
    model = ResampleFrac(old_sr=4, new_sr=5)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1000,)))
    x = jnp.ones((1000,))
    y = model.apply(params, x)

