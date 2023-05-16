import matplotlib.pyplot as plt
import math
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
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X

def fft_wrapper(sample, sampling_rate):
    '''
    inpired by https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html

    >>> results = dft(some_samples, sampling_rate)
    '''
    SN = len(sample)
    pow2_sample_len = 2 ** math.ceil(math.log2(SN))
    sample = np.pad(sample, (0, pow2_sample_len-SN), 'constant')
    X = fft(sample)
    N = len(X) // 2
    X = abs(X[:N] / N)
    n = np.arange(N)
    T = N * 2 / sampling_rate
    freq = n / T
    return [(freq[i], X[i]) for i in range(N)]

if __name__ == '__main__':
    frames, sr = librosa.load('untitled.wav', sr=None)
    bpm = librosa.feature.tempo(y=frames, sr=sr)[0]
    c_major_freqs = [(258, 262), (291, 295), (327, 331),
                     (347, 351), (390, 394), (438, 442), (491, 495),  (521, 525)]
    c_major_freqs += [(x[0] * 2, x[1] * 2) for x in c_major_freqs]
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
        # if index == 1 or index == 2:
        # plt.plot([r[0] for r in results], [r[1] for r in results])
        # plt.show()
        notes_goertzel.append(max(results_goertzel, key=lambda x: x[1])[0])
        notes_fft.append(max(results_fft, key=lambda x: x[1])[0])
        notes_dft.append(max(results_dft, key=lambda x: x[1])[0])

    # Create a time plot comparison of FFT and Goertzel
    plt.plot(range(1, len(goertzel_times) + 1), goertzel_times, label='Goertzel')
    plt.plot(range(1, len(fft_times) + 1), fft_times, label='FFT')
    plt.plot(range(1, len(dft_times) + 1), dft_times, label='DFT')
    plt.xlabel('Onset number')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()
    
    # Create a plot of the notes detected
    plt.plot(range(1, len(notes_goertzel) + 1), notes_goertzel, linestyle='None', marker='o')
    plt.plot(range(1, len(notes_fft) + 1), notes_fft, linestyle='None', marker='D')
    plt.plot(range(1, len(notes_dft) + 1), notes_dft, linestyle='None', marker='*')
    plt.xlabel('Onset number')
    plt.ylabel('Frequency (Hz)')
    plt.legend(['Goertzel', 'FFT', 'DFT'])
    plt.show()
    
    # midi = MIDIFile(1)
    # quarter_note = 60 / bpm
    # midi.addTempo(0, 0, bpm)
    # onsets = [x / quarter_note / sr for x in onsets]
    # offsets = [x for x in onsets[1:]] + [onsets[-1] + 1]
    # durations = [x - y for x, y in zip(offsets, onsets)]
    # for index, note in enumerate(notes):
    #     midi.addNote(0, 0, round(librosa.hz_to_midi(note)),
    #                  onsets[index], durations[index], 100)

    # plt.plot(onsets, notes, linestyle='None', marker='o')
    # plt.show()
