import matplotlib.pyplot as plt
import math
import cmath
import librosa
import numpy as np
from midiutil import MIDIFile
import time


def goertzel(sample, sample_rate, *freqs):
    """
    Inspired by sebpiq at
    https://stackoverflow.com/questions/13499852/scipy-fourier-transform-of-a-few-selected-frequencies
    and
    https://gist.github.com/sebpiq/4128537
    Another useful source: https://netwerkt.wordpress.com/2011/08/25/goertzel-filter/

    Returns results list containing tuples of the form (frequency, power)

    >>> results = goertzel(some_sample, 44100, (400, 500), (1000, 1100))
    """
    N = len(sample)
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start * N / sample_rate))
        k_end = int(math.ceil(f_end * N / sample_rate))
        bins = bins.union(range(k_start, k_end))

    n_range = range(0, N)
    results = []
    for k in bins:
        f_normalized = k / N
        w = 2.0 * math.cos(2.0 * math.pi * f_normalized)
        s, s_prev, s_preprev = 0.0, 0.0, 0.0
        for n in n_range:
            s = sample[n] + w * s_prev - s_preprev
            s_preprev, s_prev = s_prev, s

        results.append((f_normalized * sample_rate, s_preprev**2 + s_prev**2 - w * s_preprev * s_prev))
    return results


def dft(sample, sample_rate, *freqs):
    '''
    Calculates the DFT frequency bins passed in *freqs

    Returns results list containing tuples of the form (frequency, power)

    Inspired by: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.02-Discrete-Fourier-Transform.html

    >>> results = dft(some_samples, 44100, (400, 500), (1000, 1100))
    '''
    N = len(sample)
    bins = set()
    for f_range in freqs:
        start_bin = int(f_range[0] * N / sample_rate)
        end_bin = int(f_range[1] * N / sample_rate) + 1
        bins = bins.union(range(start_bin, end_bin))

    results = []
    for k in bins:
        result = complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            result += sample[n] * complex(math.cos(angle), -math.sin(angle))
        results.append((k * sample_rate / N, abs(result)))
    return results


def fft(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    
    source: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = [cmath.exp(-2j * cmath.pi * k / N) for k in range(N)]
        
        X = []
        for k in range(N):
            if k < N // 2:
                X.append(X_even[k] + factor[k] * X_odd[k])
            else:
                X.append(X_even[k - N // 2] + factor[k] * X_odd[k - N // 2])
        
        return X

def fft_wrapper(sample, sampling_rate):
    '''
    Inspired by https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html

    >>> results = dft(some_samples, sampling_rate)
    '''
    SN = len(sample)
    pow2_sample_len = 2 ** math.ceil(math.log2(SN))
    sample.extend([0] * (pow2_sample_len - SN))
    X = fft(sample)
    N = len(X) // 2
    X = [abs(value) for value in X[:N]]
    n = list(range(N))
    T = N * 2 / sampling_rate
    freq = [value / T for value in n]
    return [(freq[i], X[i]) for i in range(N)]


if __name__ == '__main__':
    frames, sr = librosa.load('data/audio/untitled.wav', sr=None)
    bpm = librosa.feature.tempo(y=frames, sr=sr)[0]

    c_major_freqs_base = [(258, 262), (291, 295), (327, 331),
                     (347, 351), (390, 394), (438, 442), (491, 495)]
    c_major_freqs = c_major_freqs_base + [(x[0] * 2, x[1] * 2) for x in c_major_freqs_base]

    onsets = librosa.onset.onset_detect(y=frames, sr=sr, units='samples')
    notes_goertzel = []
    notes_fft = []
    notes_dft = []
    goertzel_times = []
    dft_times = []
    fft_times = []
    for index, onset in enumerate(onsets):
        sample = []
        if index == len(onsets) - 1:
            sample = frames[onset:]
        else:
            sample =frames[onset:onsets[index+1]]
        sample = list(sample)

        goertzel_start = time.time()
        results_goertzel = goertzel(sample, sr, *c_major_freqs)
        goertzel_end = time.time()
        goertzel_times.append(goertzel_end - goertzel_start)
        
        fft_start = time.time()
        results_fft = fft_wrapper(sample, sr)
        fft_end = time.time()
        fft_times.append(fft_end - fft_start)
        
        dft_start = time.time()
        results_dft = dft(sample, sr, *c_major_freqs)
        dft_end = time.time()
        dft_times.append(dft_end - dft_start)


        print(max(results_goertzel, key=lambda x: x[1]))
        print(max(results_fft, key=lambda x: x[1]))
        print(max(results_dft, key=lambda x: x[1]))
        print()

        notes_goertzel.append(max(results_goertzel, key=lambda x: x[1])[0])
        notes_fft.append(max(results_fft, key=lambda x: x[1])[0])
        notes_dft.append(max(results_dft, key=lambda x: x[1])[0])

    plt.style.use("seaborn")

    # Create a time plot comparison of FFT and Goertzel
    plt.plot(range(1, len(goertzel_times) + 1), goertzel_times, label='Goertzel')
    plt.plot(range(1, len(fft_times) + 1), fft_times, label='FFT')
    plt.plot(range(1, len(dft_times) + 1), dft_times, label='DFT')
    plt.xlabel('Onset number')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()

    # Create a plot of the notes detected
    plt.plot(range(1, len(notes_goertzel) + 1), notes_goertzel, linestyle='None', marker='D')
    plt.plot(range(1, len(notes_fft) + 1), notes_fft, linestyle='None', marker='o')
    plt.plot(range(1, len(notes_dft) + 1), notes_dft, linestyle='None', marker='*')
    plt.xlabel('Onset number')
    plt.ylabel('Frequency (Hz)')
    plt.legend(['Goertzel', 'FFT', 'DFT'])
    plt.show()

    # Comparison of runtime per number of bins

    times_goertzel = []
    times_fft = []
    times_dft = []

    c_major_freqs += [(x[0] * 4, x[1] * 4) for x in c_major_freqs_base]
    c_major_freqs += [(x[0] * 8, x[1] * 8) for x in c_major_freqs_base]
    c_major_freqs += [(x[0] * 16, x[1] * 16) for x in c_major_freqs_base]
    print(c_major_freqs)
    print(len(c_major_freqs))

    for bin_num in range(0, len(c_major_freqs), 2):
        goertzel_start = time.time()
        for index, onset in enumerate(onsets):
            if index == len(onsets) - 1:
                results_goertzel = goertzel(frames[onset:], sr, *c_major_freqs[:bin_num])
            else:
                results_goertzel = goertzel(frames[onset:onsets[index+1]], sr, *c_major_freqs[:bin_num])
        goertzel_end = time.time()
        times_goertzel.append(goertzel_end - goertzel_start)

        fft_start = time.time()
        for index, onset in enumerate(onsets):
            if index == len(onsets) - 1:
                results_fft = fft_wrapper(frames[onset:], sr)
            else:
                results_fft = fft_wrapper(frames[onset:onsets[index+1]], sr)
        fft_end = time.time()
        times_fft.append(fft_end - fft_start)

        dft_start = time.time()
        for index, onset in enumerate(onsets):
            if index == len(onsets) - 1:
                results_dft = dft(frames[onset:], sr, *c_major_freqs[:bin_num])
            else:
                results_dft = dft(frames[onset:onsets[index+1]], sr, *c_major_freqs[:bin_num])
        dft_end = time.time()
        times_dft.append(dft_end - dft_start)

    plt.plot(range(1, len(times_goertzel) + 1), times_goertzel, label='Goertzel')
    plt.plot(range(1, len(times_fft) + 1), times_fft, label='FFT')
    plt.plot(range(1, len(times_dft) + 1), times_dft, label='DFT')
    plt.xlabel('Number of bins')
    plt.ylabel('Time (s)')
    plt.xticks(range(1, len(times_goertzel) + 1), [2*x for x in range(1, len(times_goertzel) + 1)])
    plt.legend(['Goertzel', 'FFT', 'DFT'])
    plt.show()

