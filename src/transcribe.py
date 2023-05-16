#!/usr/bin/env python3
# Usage: python3 transcribe.py <infile> <mode>
from goertzel_test import goertzel, fft_wrapper, dft
import librosa
from midiutil import MIDIFile
from sys import argv

if __name__ == '__main__':
    filename = argv[2]
    filename_no_ext = ".".join(filename.split('.')[:-1])

    mode = argv[3] if len(argv) > 3 else "fft"

    frames, sr = librosa.load(filename, sr=None)
    bpm = librosa.feature.tempo(y=frames, sr=sr)[0]

    c_major_freqs_base = [(258, 262), (291, 295), (327, 331),
                     (347, 351), (390, 394), (438, 442), (491, 495)]
    c_major_freqs = c_major_freqs_base + [(x[0] * 2, x[1] * 2) for x in c_major_freqs_base]

    onsets = librosa.onset.onset_detect(y=frames, sr=sr, units='samples')

    notes = []

    if mode == "goertzel":
        for index, onset in enumerate(onsets):
            if index == len(onsets) - 1:
                results = goertzel(frames[onset:], sr, *c_major_freqs)
            else:
                results = goertzel(frames[onset:onsets[index+1]], sr, *c_major_freqs)

            notes.append(max(results, key=lambda x: x[1])[0])
    elif mode == "fft":
        for index, onset in enumerate(onsets):
            if index == len(onsets) - 1:
                results = fft_wrapper(frames[onset:], sr)
            else:
                results = fft_wrapper(frames[onset:onsets[index+1]], sr)

            notes.append(max(results, key=lambda x: x[1])[0])
    elif mode == "dft":
        for index, onset in enumerate(onsets):
            if index == len(onsets) - 1:
                results = dft(frames[onset:], sr)
            else:
                results = dft(frames[onset:onsets[index+1]], sr)

            notes.append(max(results, key=lambda x: x[1])[0])

    midi = MIDIFile(1)
    quarter_note = 60 / bpm
    midi.addTempo(0, 0, bpm)
    onsets = [x / quarter_note / sr for x in onsets]
    offsets = [x for x in onsets[1:]] + [onsets[-1] + 1]
    durations = [x - y for x, y in zip(offsets, onsets)]
    for index, note in enumerate(notes):
        midi.addNote(0, 0, round(librosa.hz_to_midi(note)), onsets[index], durations[index], 100)

    print("Writing:", filename_no_ext+".mid")
    with open(filename_no_ext+".mid", 'wb') as output_file:
        midi.writeFile(output_file)
