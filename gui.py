import os, sys
from pathlib import Path
import re

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
from utils.plotly_utils import fix_and_write
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json", add_date=False)
from process import read_naf, read_wav, histogram, plot_histogram_by_time
import scipy.stats as stats
import pandas as pd
from scipy.spatial.distance import jensenshannon
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from music_machine import music_machine
import plotly.express as px
import wave
import numpy as np
from scipy.io.wavfile import write as wavwrite
import simpleaudio as sa

def hist_file(input_file):
    naf = read_naf()
    df = read_wav(input_file, naf)
    df = df[stats.zscore(df['frequency']) <= 3]
    hist =  histogram(df, naf)
    histogram_over_time = plot_histogram_by_time(df, naf, input_file, return_fig=True)
    return hist, histogram_over_time

def extract_major_notes(input_file):
    t = hist_file(input_file)[0].sum()
    return [x.strip() for x in t.sort_values(ascending=False).index[:2]]


def create_histogram(input_path, output_filename):
    input_path = Path(input_path)
    if input_path.is_dir():
        outputs = [hist_file(x)[0] for x in input_path.glob("*.wav")]
        hist = sum(outputs)
    else:
        hist, (fig, xaxis) = hist_file(input_path)
        fix_and_write(fig=fig, 
                      filename=f"{Path(output_filename).stem}_notes_over_time_histogram", 
                      xaxes=xaxis, 
                      output_dir=Path(output_filename).parent)

    fig = px.imshow(hist)
    fix_and_write(fig=fig, filename=Path(output_filename).stem, output_dir=Path(output_filename).parent)
    

def create_baseline(input_dir, output_filename):
    input_files =  Path(input_dir).glob("*.wav")
    dfs = [hist_file(x) for x in input_files]
    hist = sum(dfs)
    hist.to_csv(output_filename)

def compare_to_baseline(input_files, baseline_file, threshold):
    input_files = re.findall(r'\{(.*?)\}', input_files)
    messages = []
    for input_file in input_files:
        hist = hist_file(input_file)
        baseline = pd.read_csv(baseline_file)
        baseline.drop(columns='Octava', inplace=True)
        dist = jensenshannon(hist.values.flatten(), baseline.values.flatten())
        logger.info(f"Distance between {Path(input_file).name} and {Path(baseline_file).name}: {dist}")
        message =  f"The file {Path(input_file).name} is " + \
                    f"different from" if {dist > threshold} else "close to" + \
                    str(Path(baseline_file).name)
        messages += [message]
    return messages
        

def create_baseline_popup():
    def browse_input_dir():
        directory = filedialog.askdirectory(initialdir=config['input_dir'])
        if directory:
            input_dir_entry.delete(0, tk.END)
            input_dir_entry.insert(0, directory)
        popup.lift()

    def browse_output_filename():
        filename = filedialog.asksaveasfilename(initialdir=config['processed_input_dir'], defaultextension=".csv")
        if filename:
            output_filename_entry.delete(0, tk.END)
            output_filename_entry.insert(0, filename)
        popup.lift()

    def submit():
        input_dir = input_dir_entry.get()
        output_filename = output_filename_entry.get()
        create_baseline(input_dir, output_filename)
        popup.destroy()

    popup = tk.Toplevel()
    tk.Label(popup, text="Select Input Directory:").pack(pady=(5, 0))
    input_dir_entry = tk.Entry(popup, width=50)
    input_dir_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_input_dir).pack()

    tk.Label(popup, text="Select Output Filename:").pack(pady=(5, 0))
    output_filename_entry = tk.Entry(popup, width=50)
    output_filename_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_output_filename).pack()

    tk.Button(popup, text="Submit", command=submit).pack(pady=10)

def compare_to_baseline_popup():
    def browse_baseline_file():
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], initialdir=config['processed_input_dir'])
        if file:
            baseline_file_entry.delete(0, tk.END)
            baseline_file_entry.insert(0, file)
        popup.lift()

    def browse_input_files():
        input_files = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")], initialdir=config['input_dir'])  
        if input_files:
            input_file_entry.delete(0, tk.END)
            input_file_entry.insert(0, input_files)
        popup.lift()

    def show_custom_message(message):
        message_popup = tk.Toplevel()
        message_popup.title("Result")
        message_label = tk.Label(message_popup, text=message, font=("Helvetica", 12))
        message_label.pack(pady=20, padx=20)
        tk.Button(message_popup, text="Close", command=message_popup.destroy).pack(pady=10)


    def submit():
        baseline_file = baseline_file_entry.get()
        input_files = input_file_entry.get()
        threshold = slider.get() / 100.0
        result = compare_to_baseline(input_files, baseline_file, threshold)
        show_custom_message("\n".join(result))

    popup = tk.Toplevel()
    
    tk.Label(popup, text="Select Baseline File:").pack(pady=(5, 0))
    baseline_file_entry = tk.Entry(popup, width=50)
    baseline_file_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_baseline_file).pack()

    tk.Label(popup, text="Select Input File:").pack(pady=(5, 0))
    input_file_entry = tk.Entry(popup, width=50)
    input_file_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_input_files).pack()

    slider_range = (config['max_threshold'] - config['min_threshold']) * 100
    slider_start = config['min_threshold'] * 100

    slider = tk.Scale(popup, from_=slider_start, to=slider_start + slider_range, orient=tk.HORIZONTAL)
    slider.pack(pady=(5, 5))

    tk.Button(popup, text="Submit", command=submit).pack(pady=(5, 10))
    tk.Button(popup, text="Close", command=popup.destroy).pack(pady=(0, 10))

def create_histogram_popup():
    
    def browse_input_dir():
        directory = filedialog.askopenfilename(initialdir=config['input_dir'])
        if directory:
            input_dir_entry.delete(0, tk.END)
            input_dir_entry.insert(0, directory)
        popup.lift()

    def browse_output_filename():
        filename = filedialog.asksaveasfilename(initialdir=config['processed_input_dir'], defaultextension=".csv")
        if filename:
            output_filename_entry.delete(0, tk.END)
            output_filename_entry.insert(0, filename)
        popup.lift()

    def submit():
        input_dir = input_dir_entry.get()
        output_filename = output_filename_entry.get()
        create_histogram(input_dir, output_filename)
        popup.destroy()

    popup = tk.Toplevel()
    tk.Label(popup, text="Select Input Directory:").pack(pady=(5, 0))
    input_dir_entry = tk.Entry(popup, width=50)
    input_dir_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_input_dir).pack()

    tk.Label(popup, text="Select Output Filename:").pack(pady=(5, 0))
    output_filename_entry = tk.Entry(popup, width=50)
    output_filename_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_output_filename).pack()

    tk.Button(popup, text="Submit", command=submit).pack(pady=10)

def get_wav_length(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration_seconds = n_frames / frame_rate
        return duration_seconds

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
    wavwrite(filename, sample_rate, audio_data)


def music_machine_popup():
    def browse_input_dir():
        input_file = filedialog.askopenfilename(initialdir=config['input_dir'])
        if input_file:
            input_file_entry.delete(0, tk.END)
            input_file_entry.insert(0, input_file)
        popup.lift()

    def browse_output_filename():
        filename = filedialog.asksaveasfilename(initialdir=config['melodies_dir'], defaultextension=".wav")
        if filename:
            output_filename_entry.delete(0, tk.END)
            output_filename_entry.insert(0, filename)
        popup.lift()

    def play_wav(filename):
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until the audio file is done playing

    def submit():
        input_file = input_file_entry.get()
        output_file = output_filename_entry.get()
        voice_segment_duration = int(get_wav_length(str(input_file)))
        melody_type = melody_type_combobox.get() 
        num_cycles = int(cycles_entry.get())
        first_dominant_tone, second_dominant_tone = extract_major_notes(input_file)
        harmony, melody = music_machine(voice_segment_duration, first_dominant_tone, second_dominant_tone, melody_type, num_cycles)
        logger.info(f"Harmony: {harmony}\nMelody: {melody}. writing to {output_file}")
        logger.info(voice_segment_duration)
        create_wav_from_notes(melody=melody, filename=output_file, duration=voice_segment_duration)
        logger.info("wav file created")
        # logger.info(f"Playing melody")
        # #play_wav(output_file)

        popup.destroy()

    popup = tk.Toplevel()
    tk.Label(popup, text="Select Input File:").pack(pady=(5, 0))
    input_file_entry = tk.Entry(popup, width=50)
    input_file_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_input_dir).pack()
    tk.Label(popup, text="Select Output Filename:").pack(pady=(5, 0))
    output_filename_entry = tk.Entry(popup, width=50)
    output_filename_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_output_filename).pack()
    
    tk.Label(popup, text="Melody Type:").pack()
    melody_type_combobox = ttk.Combobox(popup, values=["Major", "Minor"])
    melody_type_combobox.pack(pady=(5, 5))
    
    tk.Label(popup, text="Number of Cycles:").pack()
    cycles_entry = tk.Entry(popup)
    cycles_entry.pack(pady=(5, 5))
    tk.Button(popup, text="Submit", command=submit).pack(pady=10)

def main():
    root = tk.Tk()
    root.title("Python GUI")
    root.geometry("300x350")  # Adjust the size as per your requirement

    tk.Button(root, text="Create Baseline", command=create_baseline_popup, padx=10, pady=10).pack(pady=20)
    tk.Button(root, text="Compare to Baseline", command=compare_to_baseline_popup, padx=10, pady=10).pack(pady=20)
    tk.Button(root, text="Create Histogram", command=create_histogram_popup, padx=10, pady=10).pack(pady=20)
    tk.Button(root, text="Music Machine", command=music_machine_popup, padx=10, pady=10).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()