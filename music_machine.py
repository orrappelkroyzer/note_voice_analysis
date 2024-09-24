import os, sys
from pathlib import Path
import re

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json", add_date=False)
import pandas as pd
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
from scipy.io.wavfile import write as wavwrite


# Define the number of harmony cycles (always 4)
num_cycles = 4


note_order = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']

# Define the major and minor chord patterns for each note
major_chords = {
    'C': ['C', 'E', 'G'],
    'C#': ['C#', 'E#', 'G#'],
    'D': ['D', 'F#', 'A'],
    'D#': ['D#', 'F#', 'A#'],
    'E': ['E', 'G#', 'B'],
    'F': ['F', 'A', 'C'],
    'F#': ['F#', 'A#', 'C#'],
    'G': ['G', 'B', 'D'],
    'G#': ['G#', 'B', 'D#'],
    'A': ['A', 'C#', 'E'],
    'A#': ['A#', 'C#', 'F#'],
    'B': ['B', 'D#', 'F#']
}

minor_chords = {
    'C': ['C', 'D#', 'G'],
    'C#': ['C#', 'E', 'G#'],
    'D': ['D', 'F', 'A'],
    'D#': ['D#', 'F#', 'A#'],
    'E': ['E', 'G', 'B'],
    'F': ['F', 'G#', 'C'],
    'F#': ['F#', 'A', 'C#'],
    'G': ['G', 'A#', 'D'],
    'G#': ['G#', 'B', 'D#'],
    'A': ['A', 'C', 'E'],
    'A#': ['A#', 'C#', 'E#'],
    'B': ['B', 'D', 'F#']
}


def read_naf():    
    logger.info("Reading naf")
    naf = pd.read_csv(Path(config['input_dir']) / "notes and frequencies.csv")
    naf = naf.set_index("Unnamed: 0")
    naf.index.name = "Note"
    naf.columns.name = "Octava"
    naf = naf.stack().sort_values()
    return naf

def music_machine(voice_segment_duration, first_dominant_tone, second_dominant_tone, melody_type, num_cycles):

    # # Determine if the harmony uses flats or sharps
    # harmony_uses_flats = 'b' in first_dominant_tone or 'b' in second_dominant_tone

    # Divide the voice segment duration into equal parts for harmony and melody
    cycle_duration = voice_segment_duration / num_cycles

    # Initialize the list to store harmony and melody
    harmony = []
    melody = []

    # Generate harmony and melody for each cycle
    for cycle in range(num_cycles):
        # Determine the dominant tone for this cycle
        if cycle == 0 or cycle == num_cycles - 1:
            dominant_tone = first_dominant_tone
        else:
            dominant_tone = second_dominant_tone

        # Add the harmony tone for this cycle
        harmony.append(dominant_tone)

        # Find the chord pattern based on the selected melody type
        if melody_type == "Major":
            chord_pattern = major_chords[dominant_tone]
        elif melody_type == "Minor":
            chord_pattern = minor_chords[dominant_tone]
        else:
            raise ValueError(f"Invalid melody type: {melody_type}")

        # Add the chord pattern notes to the melody
        melody += chord_pattern

        logger.info(f"Cycle {cycle + 1} (Duration: {cycle_duration:.2f} seconds):")
        logger.info(f"Harmony = {harmony[cycle]}")
        logger.info(f"Melody = {' '.join(chord_pattern)}")

    # Ensure the total duration of the generated melody matches the voice segment duration
    while len(melody) < len(harmony):
        # Add extra time to the last melody note
        melody.append(melody[-1])

    # Trim the melody to match the voice segment duration
    melody = melody[:int(voice_segment_duration)]
    return harmony, melody


def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return wave

def create_wav_from_notes(melody, filename, duration, sample_rate=44100):
    naf = read_naf().unstack()
    frequencies = [naf.loc[x, '4'] for x in melody]
    audio_data = np.array([])
    for frequency in frequencies:
        note_data = generate_sine_wave(frequency, duration, sample_rate)
        audio_data = np.concatenate((audio_data, note_data))

    # Normalize to 16-bit range
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

    # Write to WAV file
    logger.info(f"Writing audio to {filename}")
    wavwrite(filename, sample_rate, audio_data)


def frequency_to_midi(frequency):
    """Convert a frequency to the nearest MIDI note number."""
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    return int(round(midi_note))

def create_midi_from_notes(melody, filename, duration, tempo=500000):
    # Create a new MIDI file and track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo (default is 500000 microseconds per beat, which is 120 BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    naf = read_naf().unstack()
    frequencies = [naf.loc[x, '4'] for x in melody]

    # Convert each frequency to a MIDI note and assign durations
    for frequency in frequencies:
        midi_note = frequency_to_midi(frequency)

        # Note on (velocity 64 is a standard value, you can modify it)
        track.append(Message('note_on', note=midi_note, velocity=64, time=0))
        
        # Note off (after the given duration, time in ticks)
        ticks_per_beat = mid.ticks_per_beat
        time_per_note = int((duration / 4) * ticks_per_beat)  # 4 = quarter note length
        track.append(Message('note_off', note=midi_note, velocity=64, time=time_per_note))

    # Save the MIDI file
    logger.info(f"Writing audio to {filename}")
    mid.save(filename)