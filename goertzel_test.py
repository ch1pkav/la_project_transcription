
import matplotlib.pyplot as plt
import math
import wave
import librosa
import numpy as np
from midiutil import MIDIFile


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

    >>> results = dft(some_samples, 44100, (400, 500), (1000, 1100))
    '''
    N = len(sample)
    bins = set()
    for f_range in freqs:
        bins = bins.union(range(f_range[0], f_range[1]))

    results = []
    for k in bins:
        result = complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            result += sample[n] * complex(math.cos(angle), -math.sin(angle))
        results.append((k * sample_rate / N, abs(result)))
    return results

def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = abs(np.dot(e, x))

    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T 
    
    return freq, X

def FFT(x):
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
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
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
    X = FFT(sample)
    N = len(X) // 2
    X = abs(X[:N] / N)
    n = np.arange(N)
    T = N * 2 / sampling_rate
    freq = n / T
    return [(freq[i], X[i]) for i in range(N)]

if __name__ == '__main__':
    frames, sr = librosa.load('untitled.wav', sr=None)
    bpm = librosa.feature.tempo(y=frames, sr=sr)[0]
    c_major_freqs = [(250, 270), (288, 298), (320, 340),
                     (345, 355), (380, 400), (430, 450), (480, 500)]
    # c_major_freqs += [(x[0] * 2, x[1] * 2) for x in c_major_freqs]
    onsets = librosa.onset.onset_detect(y=frames, sr=sr, units='samples')
    notes = []
    for index, onset in enumerate(onsets):
        if index == len(onsets) - 1:
            # freqs, X = DFT(frames[onset:])
            # results = goertzel(frames[onset:], sr, *c_major_freqs)
            # results = dft(frames[onset:], sr, *c_major_freqs)
            results = fft_wrapper(frames[onset:], sr)
        else:
            # freqs, X = DFT(frames[onset:onsets[index+1]])
            # results = goertzel(frames[onset:onsets[index+1]], sr, *c_major_freqs)
            # results = dft(frames[onset:onsets[index+1]], sr, *c_major_freqs)
            results = fft_wrapper(frames[onset:onsets[index+1]], sr)

        # print(freqs[np.argmax(X)])

        print(max(results, key=lambda x: x[1]))
        # if index == 1 or index == 2:
        # plt.plot([r[0] for r in results], [r[1] for r in results])
        # plt.show()
        notes.append(max(results, key=lambda x: x[1])[0])

    midi = MIDIFile(1)
    quarter_note = 60 / bpm
    midi.addTempo(0, 0, bpm)
    onsets = [x / quarter_note / sr for x in onsets]
    offsets = [x for x in onsets[1:]] + [onsets[-1] + 1]
    durations = [x - y for x, y in zip(offsets, onsets)]
    for index, note in enumerate(notes):
        midi.addNote(0, 0, round(librosa.hz_to_midi(note)),
                     onsets[index], durations[index], 100)

    with open("untitled.mid", "wb") as output_file:
        midi.writeFile(output_file)

    # plt.plot(onsets, notes, linestyle='None', marker='o')
    # plt.show()
