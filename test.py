import matplotlib.pyplot as plot
import numpy as np
import soundfile

from qdft import QDFT
from vocoder import Vocoder

sr = 48000            # sample rate in hertz
ny = sr / 2           # nyquist frequency in hertz

bw = (5e3, ny - 1e3)  # lowest and highest frequency in hertz to be resolved
r  = 24               # octave resolution, e.g. number of DFT bins per octave

d = 1                 # test signal duration in seconds
n = int(sr * d)       # test signal duration in samples
t = np.arange(n) / sr # time vector

if False: # single frequency test signal

    f = 10e3

    x = np.sin(2 * np.pi * f * t)

else: # chirp test signal

    lfo = np.sin(np.pi * t / d)

    f = lfo * 9e3 + 1e3

    x = np.exp(2j * np.pi * f / sr)
    np.cumprod(x, out=x)
    x = np.imag(x)

qdft = QDFT(sr, bw, r)
vocoder = Vocoder(sr)

dft = qdft.qdft(x)

magn, freq = vocoder.analyze(dft)

if True: # spectral processing

    freq *= 0.5

dft = vocoder.synthesize(magn, freq)

y = qdft.iqdft(dft)

if True:

    soundfile.write('x.wav', np.squeeze(x.T), sr)
    soundfile.write('y.wav', np.squeeze(y.T), sr)

if True:

    print('Frequency Bins [Hz]', qdft.frequencies)
    imshow = dict(aspect='auto', cmap='inferno', interpolation='nearest', origin='lower')

    plot.figure('Magnitude [dB]')
    plot.imshow(20 * np.log10(magn.T + np.finfo(float).eps), vmin=-120, vmax=0, **imshow)

    plot.figure('Instantaneous Frequency [Hz]')
    plot.imshow(freq.T, vmin=0, vmax=None, **imshow)

    plot.figure()
    plot.plot(t, x, label='x', color='blue', alpha=0.5)
    plot.plot(t, y, label='y', color='red', alpha=0.5)
    plot.legend()

    plot.show()
