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
from music_machine import music_machine, create_wav_from_notes, create_midi_from_notes
import plotly.express as px
import wave
import numpy as np

import traceback

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
    hist, (fig, xaxis) = hist_file(input_path)
    fix_and_write(fig=fig, 
                    filename=f"{Path(output_filename).stem}_notes_over_time_histogram", 
                    xaxes=xaxis, 
                    output_dir=Path(output_filename).parent)
    hist.to_csv(f"{output_filename}.csv")
    save_historgram(hist, output_filename)

def plot_histogram(input_path, output_filename):
    input_path = Path(input_path)
    if input_path.is_dir():
        outputs = [pd.read_csv(x) for x in input_path.glob("*.wav")]
        hist = sum(outputs)
    else:
        hist = pd.read_csv(input_path)
        
    save_historgram(hist, output_filename)

def save_historgram(hist, output_filename):
    output_filename = Path(output_filename)
    fig = px.imshow(hist)
    fix_and_write(fig=fig, filename=Path(output_filename).stem, output_dir=Path(output_filename).parent)
    message_popup = tk.Toplevel()
    message_popup.title("Result")
    message_label = tk.Label(message_popup, text=f'Histogram written to {output_filename.stem}.png', font=("Helvetica", 12))
    message_label.pack(pady=20, padx=20)
    tk.Button(message_popup, text="Close", command=message_popup.destroy).pack(pady=10)
    

def create_baseline(input_dir, output_filename):
    input_dir = Path(input_dir)
    outputs = [hist_file(x)[0] for x in input_dir.glob("*.wav")]
    outputs += [hist_file(x)[0] for x in input_dir.glob("*.csv")]
    if len(outputs) == 0:
        logger.error(f"No files found in {input_dir}")
        message_popup = tk.Toplevel()
        message_popup.title("Error")
        message_label = tk.Label(message_popup, text=f"No files found in {input_dir}", font=("Helvetica", 12))
        message_label.pack(pady=20, padx=20)
        tk.Button(message_popup, text="Close", command=message_popup.destroy).pack(pady=10)
        return
    hist = sum(outputs)
    hist.to_csv(output_filename)
    logger.info(f"Baseline written to {output_filename}")
    message_popup = tk.Toplevel()
    message_popup.title("Result")
    message_label = tk.Label(message_popup, text=f'Baseline written to {output_filename}', font=("Helvetica", 12))
    message_label.pack(pady=20, padx=20)
    tk.Button(message_popup, text="Close", command=message_popup.destroy).pack(pady=10)
    save_historgram(hist, output_filename)

def compare_to_baseline(input_files, baseline_file, threshold):
    input_files = re.findall(r'\{(.*?)\}', input_files)
    
    messages = []
    for input_file in input_files:
        logger.info(f"Comparing {Path(input_file).name} to {Path(baseline_file).name}")
        hist = hist_file(input_file)[0]
        baseline = pd.read_csv(baseline_file)
        baseline.drop(columns='Octava', inplace=True)
        hist_flat = hist.values.flatten()
        baseline_flat = baseline.values.flatten()
        hist_flat = np.maximum(hist_flat, 0)  # Make all values non-negative
        baseline_flat = np.maximum(baseline_flat, 0)  # Make all values non-negative
        hist_flat = hist_flat / np.sum(hist_flat) if np.sum(hist_flat) > 0 else hist_flat
        baseline_flat = baseline_flat / np.sum(baseline_flat) if np.sum(baseline_flat) > 0 else baseline_flat
        epsilon = 1e-10
        hist_flat = hist_flat + epsilon
        baseline_flat = baseline_flat + epsilon
        dist = jensenshannon(hist_flat, baseline_flat)
        
        logger.info(f"Distance between {Path(input_file).name} and {baseline_file}: {dist}")
        message =  f"The file {Path(input_file).name} is " + \
                    (f"different from " if {dist > threshold} else "close to ") + \
                    str(Path(baseline_file).name) + f" with a distance of {round(dist, 2)}"
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
        try:
            input_dir = input_dir_entry.get()
            output_filename = output_filename_entry.get()
            create_baseline(input_dir, output_filename)
            popup.destroy()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Internal error: {e}")

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
        input_files = filedialog.askopenfilenames(filetypes=[("WAV and CSV files", "*.wav *.csv")], initialdir=config['input_dir'])  
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
        try:
            baseline_file = baseline_file_entry.get()
            input_files = input_file_entry.get()
            threshold = slider.get() / 100.0
            result = compare_to_baseline(input_files, baseline_file, threshold)
            show_custom_message("\n".join(result))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Internal error: {e}")

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
    
    def browse_input_file():
        directory = filedialog.askopenfilename(initialdir=config['input_dir'], filetypes=[("WAV and CSV files", "*.wav *.csv")])
        if directory:
            input_dir_entry.delete(0, tk.END)
            input_dir_entry.insert(0, directory)
        popup.lift()

    def browse_output_filename():
        filename = filedialog.asksaveasfilename(initialdir=config['processed_input_dir'])
        if filename:
            output_filename_entry.delete(0, tk.END)
            output_filename_entry.insert(0, filename)
        popup.lift()

    def ch_submit():
        try:
            input_dir = input_dir_entry.get()
            output_filename = output_filename_entry.get()
            create_histogram(input_dir, output_filename)
            popup.destroy()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Internal error: {e}")

    def ph_submit():
        try:
            input_dir = input_dir_entry.get()
            output_filename = output_filename_entry.get()
            plot_histogram(input_dir, output_filename)
            popup.destroy()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Internal error: {e}")

    popup = tk.Toplevel()
    tk.Label(popup, text="Select Input Filename:").pack(pady=(5, 0))
    input_dir_entry = tk.Entry(popup, width=50)
    input_dir_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_input_file).pack()

    tk.Label(popup, text="Select Output Filename:").pack(pady=(5, 0))
    output_filename_entry = tk.Entry(popup, width=50)
    output_filename_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_output_filename).pack()

    tk.Button(popup, text="Create Histogram", command=ch_submit).pack(pady=10)
    tk.Button(popup, text="Plot Histogram", command=ph_submit).pack(pady=5)

def get_wav_length(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration_seconds = n_frames / frame_rate
        return duration_seconds




def music_machine_popup():
    def browse_input_file():
        input_file = filedialog.askopenfilename(initialdir=config['input_dir'])
        if input_file:
            input_file_entry.delete(0, tk.END)
            input_file_entry.insert(0, input_file)
        popup.lift()

    def browse_output_filename():
        filename = filedialog.asksaveasfilename(initialdir=config['melodies_dir'])
        if filename:
            output_filename_entry.delete(0, tk.END)
            output_filename_entry.insert(0, filename)
        popup.lift()


    def ask_format(melody, filename, duration):
        """Ask the user for the desired output format (WAV or MIDI)."""
        format_popup = tk.Toplevel(popup)
        format_popup.title("Choose Format")

        # Make the popup modal
        format_popup.grab_set()

        def choose_format(selected_format):
            format_popup.destroy()
            if selected_format == "WAV":
                create_wav_from_notes(melody=melody, filename=filename, duration=duration)
                logger.info("WAV file created")
            elif selected_format == "MIDI":
                create_midi_from_notes(melody=melody, filename=filename, duration=duration)
                logger.info("MIDI file created")
            popup.destroy()

        tk.Label(format_popup, text="Select the output format:").pack(pady=10)
        tk.Button(format_popup, text="WAV", command=lambda: choose_format("WAV")).pack(pady=5)
        tk.Button(format_popup, text="MIDI", command=lambda: choose_format("MIDI")).pack(pady=5)

        # Wait for the user to make a choice before continuing
        popup.wait_window(format_popup)


    def submit():
        try:
            input_file = input_file_entry.get()
            output_file = output_filename_entry.get()
            voice_segment_duration = int(get_wav_length(str(input_file)))
            melody_type = melody_type_combobox.get() 
            num_cycles = int(cycles_entry.get())
            first_dominant_tone, second_dominant_tone = extract_major_notes(input_file)
            harmony, melody = music_machine(voice_segment_duration, first_dominant_tone, second_dominant_tone, melody_type, num_cycles)
            logger.info(f"Harmony: {harmony}\nMelody: {melody}. writing to {output_file}")
            ask_format(melody=melody, filename=output_file, duration=voice_segment_duration)
            logger.info("file created")
            # logger.info(f"Playing melody")
            # #play_wav(output_file)

            popup.destroy()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Internal error: {e}")

    popup = tk.Toplevel()
    tk.Label(popup, text="Select Input File:").pack(pady=(5, 0))
    input_file_entry = tk.Entry(popup, width=50)
    input_file_entry.pack(pady=(0, 5))
    tk.Button(popup, text="Browse", command=browse_input_file).pack()
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
    root.title("GarfunkeL")
    root.geometry("300x350")  # Adjust the size as per your requirement

    tk.Button(root, text="Create Baseline from a Direcoty", command=create_baseline_popup, padx=10, pady=10).pack(pady=20)
    tk.Button(root, text="Create/plot Histogram of a Single File", command=create_histogram_popup, padx=10, pady=10).pack(pady=20)
    tk.Button(root, text="Compare to Baseline", command=compare_to_baseline_popup, padx=10, pady=10).pack(pady=20)
    tk.Button(root, text="Music Machine", command=music_machine_popup, padx=10, pady=10).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()