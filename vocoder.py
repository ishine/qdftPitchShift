import numpy as np

class Vocoder:

    def __init__(self, samplerate):

        self.samplerate = samplerate
        self.nyquist = samplerate / 2

        self.rad2hz = samplerate / (2 * np.pi)
        self.hz2rad = (2 * np.pi) / samplerate

    def analyze(self, dft):

        dft = np.atleast_2d(dft)

        # TODO handle invalid freqs
        if False:

            with np.errstate(divide='ignore', invalid='ignore'):
                phase = dft / np.roll(dft, shift=1, axis=0)
                phase = np.angle(phase)

            assert np.min(phase[np.isfinite(phase)]) >= -np.pi
            assert np.max(phase[np.isfinite(phase)]) <= +np.pi

        else:

            phase = dft / (np.roll(dft, shift=1, axis=0) + np.finfo(float).eps)
            phase = np.angle(phase)

            assert np.min(phase) >= -np.pi
            assert np.max(phase) <= +np.pi

        magn = np.abs(dft)
        freq = phase * self.rad2hz

        # TODO handle invalid freqs
        if False:

            mask = ~np.isfinite(freq) | (freq < 0)
            magn[mask] = 0
            freq[mask] = 0

            mask = freq > self.nyquist
            magn[mask] = 0
            freq[mask] = self.nyquist

            # magn *= 2 # wtf

        assert not np.any(~np.isfinite(magn))
        assert not np.any(~np.isfinite(freq))

        return magn, freq

    def synthesize(self, magn, freq):

        magn = np.atleast_2d(magn)
        freq = np.atleast_2d(freq)

        assert not np.any(~np.isfinite(magn))
        assert not np.any(~np.isfinite(freq))

        # TODO handle invalid freqs
        if False:

            assert np.min(freq) >= 0
            assert np.max(freq) <= self.nyquist

        phase = freq * self.hz2rad
        np.cumsum(phase, axis=0, out=phase)

        dft = magn * np.exp(1j * phase)

        return dft
