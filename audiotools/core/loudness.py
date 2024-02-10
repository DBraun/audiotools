import jax
import jax.numpy as jnp
import jaxloudnorm as jln


class LoudnessMixin:
    _loudness = None
    MIN_LOUDNESS = -70
    """Minimum loudness possible."""

    def loudness(
        self, filter_class: str = "K-weighting", block_size: float = 0.400, **kwargs
    ):
        """Calculates loudness using an implementation of ITU-R BS.1770-4.
        Allows control over gating block size and frequency weighting filters for
        additional control. Measure the integrated gated loudness of a signal.

        API is derived from PyLoudnorm, but this implementation is ported to PyTorch
        and is tensorized across batches. When on GPU, an FIR approximation of the IIR
        filters is used to compute loudness for speed.

        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification.

        Parameters
        ----------
        filter_class : str, optional
            Class of weighting filter used.
            K-weighting' (default), 'Fenton/Lee 1'
            'Fenton/Lee 2', 'Dash et al.'
            by default "K-weighting"
        block_size : float, optional
            Gating block size in seconds, by default 0.400
        kwargs : dict, optional
            Keyword arguments to :py:func:`audiotools.core.loudness.Meter`.

        Returns
        -------
        jnp.ndarray
            Loudness of audio data.
        """
        if self._loudness is not None:
            return self._loudness
        original_length = self.signal_length
        if self.signal_duration < 0.5:
            pad_len = int((0.5 - self.signal_duration) * self.sample_rate)
            self.zero_pad(0, pad_len)

        # create BS.1770 meter
        meter = jln.Meter(
            self.sample_rate, filter_class=filter_class, block_size=block_size, **kwargs
        )

        # measure loudness
        loudness = jax.vmap(meter.integrated_loudness)(self.audio_data.transpose(0, 2, 1))
        self.truncate_samples(original_length)
        min_loudness = jnp.ones_like(loudness) * self.MIN_LOUDNESS

        self._loudness = jnp.maximum(loudness, min_loudness)

        return self._loudness


if __name__ == '__main__':

    from audiotools.core.audio_signal import AudioSignal

    key = jax.random.key(0)
    SAMPLE_RATE = 44100

    x = jax.random.uniform(key, shape=(1, int(SAMPLE_RATE)), minval=-1, maxval=1)

    signal = AudioSignal(x, SAMPLE_RATE)

    loudness = signal.loudness()
    print('loudness: ', loudness)
