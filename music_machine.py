import os, sys
from pathlib import Path
import re

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json", add_date=False)
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
    'G#': ['G#', 'B#', 'D#'],
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
