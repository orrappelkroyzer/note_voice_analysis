# note_voice_analysis
This package runs a simple GUI to analyze voice files through musicological lens.

Usage:
1. Download the code and locate it in the relevant working directory
2. in the config.json file, fill in the values of the relevant parameters, as strings:
     "input_dir" : <path of directory where input wav files are located>
    "processed_input_dir" : <path of directory where interim calculations will be stored>,
    "output_dir" : <path of directory where output graphs will be written>,
    "melodies_dir" : <path of directory where melodies generated by the algorithm will be stored>,
3. download python
4. run python gui.py

Possible actions in the GUI
1. Create a histogram of notes from all files in the given directory. The output is a CSV file that can be used as a baseline for later comparisons.
2. Compare a baseline created in the previous action to another file
3. Create histogram files from a single file
4. Create background music matching given file