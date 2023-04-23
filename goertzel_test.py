
import matplotlib.pyplot as plt
import math
import wave
import librosa
import numpy as np

def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.
    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.
    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.
    Example of usage :

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

if __name__ == '__main__':

    frames, sr = librosa.load('untitled.wav', sr=None)
    c_major_freqs = [(250, 270), (285, 298), (325, 335), (345, 355), (380, 400), (430, 450), (480, 500)]
    onsets = librosa.onset.onset_detect(y=frames, sr=sr, units='samples')
    notes = []
    for index, onset in enumerate(onsets):
        if index == len(onsets) - 1:
            freqs, results = goertzel(frames[onset:], sr, *c_major_freqs)
        else:
            freqs, results = goertzel(frames[onset:onsets[index+1]], sr, *c_major_freqs)
        # print(freqs[results.index(max(results, key=lambda x: x[2]))])
        notes.append(freqs[results.index(max(results, key=lambda x: x[2]))])

    print(notes)
    with plt.xkcd():
        plt.plot(onsets, notes, linestyle='None', marker='o')
        plt.show()
