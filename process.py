import crepe
from scipy.io import wavfile
import os, sys
from pathlib import Path
import plotly.express as px

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)

from utils.utils import load_config, get_logger
from utils.plotly_utils import fix_and_write
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json")
import pandas as pd, numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import chisquare
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn import metrics
import plotly.graph_objects as go
from scipy.cluster.vq import kmeans2

def read_naf():    
    logger.info("Reading naf")
    naf = pd.read_csv(Path(config['input_dir']) / "notes and frequencies.csv")
    naf = naf.set_index("Unnamed: 0")
    naf.index.name = "Note"
    naf.columns.name = "Octava"
    naf = naf.stack().sort_values()
    return naf

    

def read_wav(filename, naf):
    logger.info(f"Reading wav file {filename}")
    sr, audio = wavfile.read(filename)
    d = dict(zip(['time', 'frequency', 'confidence', 'activation'], crepe.predict(audio, sr, viterbi=True)))
    del d['activation']
    df = pd.DataFrame(d)
    df = df[df['confidence'] > 0.8]
    naf_borders = [0] + list((naf.values[:-1] + naf.values[1:])/2) + [10000]
    naf_with_index = naf.reset_index()
    df = df.join(pd.DataFrame({'index' : pd.cut(df['frequency'], naf_borders, labels=naf_with_index.index)}).join(naf_with_index[['Note', 'Octava']], on='index')[['Note', 'Octava']])
    naf.name = 'Note Frequency'
    df = df.merge(naf, on=['Note', 'Octava'])
    return df

def hard_bin_histogram(df, naf_borders, naf):
    count, division = np.histogram(df['frequency'], bins = naf_borders)
    histogram = pd.Series(count, index=naf.index)
    return histogram

def soft_bin_histogram(df, naf_borders, naf):
    
    frequencies = df['frequency']
    bin_centers = naf.values
    histogram = np.zeros_like(bin_centers)
    for value in frequencies:
        distances = np.abs(bin_centers - value)
        
        # Find the two closest bins
        closest_bins = distances.argsort()[:2]
        
        # Calculate the proportion for each bin
        total_distance = distances[closest_bins].sum()
        proportions = (1 - distances[closest_bins] / total_distance)
        
        # Update the histogram
        histogram[closest_bins] += proportions
    histogram = pd.Series(histogram, index=naf.index)
    return histogram

# now `histogram` contains the result you want


def histogram(df, naf):
    logger.info("histogram")
    naf_borders = [0] + list((naf.values[:-1] + naf.values[1:])/2) + [10000]
    # histogram = hard_bin_histogram(df, naf_borders, naf)
    histogram = soft_bin_histogram(df, naf_borders, naf)
    histogram = histogram.unstack().T
    return histogram


def plot_histogram(df, naf, input_file):
    histogram = histogram(df, naf)
    hist = pd.concat([histogram, pd.DataFrame(pd.Series(histogram.sum(), name='Overall')).T])
    (Path(config['processed_input_dir']) / "csv").mkdir(parents=True, exist_ok=True)
    hist.to_csv(Path(config['processed_input_dir']) / "csv" / f"{input_file.stem}.csv")
    fig = px.imshow(hist.iloc[:-1])
    (Path(config['processed_input_dir']) / "images").mkdir(parents=True, exist_ok=True)
    fix_and_write(fig=fig, filename=f"{input_file.stem}_histogram", output_dir=Path(config['processed_input_dir']) / "images")

def histogram_by_time(df, naf, input_file):
    def foo(g):
        if len(g) < 10:
            return pd.Series()
        s= g['Note'].value_counts(normalize=True, ascending=False)
        if s[0] > 0.7:
            return pd.Series({'Note' : s.index[0], 'Octava' : g['Octava'].value_counts(normalize=True, ascending=False).index[0]})
        return pd.Series()
    df['second'] = df.time.apply(int)
    s = df.groupby('second').apply(foo).unstack().reset_index()
    
    if len(s) == 0:
        return

    dominant_note =  s.join(naf, on=["Note", "Octava"])
    return dominant_note

def plot_histogram_by_time(df, naf, input_file, return_fig=False):
    dominant_note = histogram_by_time(df, naf, input_file)
    fig = px.scatter(df, x='time', y='frequency', color='Note',
                            category_orders = {'Note' : ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G ', 'G#']})
    if dominant_note is not None:
        fig.add_trace(go.Scatter(x=[i+0.5 for i in dominant_note['second']],
                                y=dominant_note['Note Frequency'],
                                text = dominant_note['Note'],
                                mode="text+markers",
                                textposition='middle right',
                                marker=dict(symbol='x', size=10, color="black")))
    xaxis=dict(
            tickvals=list(range(int(df.time.max()+1))),  # Set the positions of the ticks
            ticktext=list(range(int(df.time.max()+1))),  # Set the labels for the ticks
        )
    if return_fig:
        return fig, xaxis
    (Path(config['processed_input_dir']) / 'histogram_over_time').mkdir(parents=True, exist_ok=True)
    fix_and_write(fig=fig, filename=f"{input_file.stem}_notes_over_time_histogram", xaxes=xaxis, output_dir=Path(config['processed_input_dir']) / 'histogram_over_time')



def per_file_process():
    naf = read_naf()
    Path(config['processed_input_dir']).mkdir(parents=True, exist_ok=True)
    if config.get('input_file', None) is not None:
        input_files = [Path(config['input_dir']) / config['input_file']]
    else:
        input_files =  Path(config['input_dir']).glob("*.wav")
    for input_file in input_files:
        df = read_wav(input_file, naf)
        df = df[stats.zscore(df['frequency']) <= 3]
        plot_histogram(df, naf, input_file)
        fig = px.scatter(df, x='time', y='frequency', color='Note', category_orders = {'Note' : ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G ', 'G#']})
        (Path(config['processed_input_dir']) / "notes_over_time").mkdir(parents=True, exist_ok=True)
        fix_and_write(fig=fig, filename="f{input_file.stem}_notes_over_time", output_dir=Path(config['processed_input_dir']) / "notes_over_time")
        # detect_planes(df, input_file)
        # find_peaks(df, input_file)
        plot_histogram_by_time(df, naf, input_file)

def dists():
    logger.info("dists")
    files = list((Path(config['processed_input_dir']) / 'csv').glob("*.csv"))
    dfs = dict(jensenshannon = pd.DataFrame(),
                chi2 = pd.DataFrame(),
                mutual_info = pd.DataFrame(),
                emd = pd.DataFrame())
    for f1 in files:
        df1 = pd.read_csv(f1).set_index("Unnamed: 0").iloc[:-1]
        df1 /= df1.sum().sum()
        df1_t = df1.loc[['0', '1', '2', '3', '4']]
        df1_t /= df1_t.sum().sum()
        df1_t += 0.000000001
        for f2 in files:
            df2 = pd.read_csv(f2).set_index("Unnamed: 0").iloc[:-1]
            df2 /= df2.sum().sum()
            df2_t = df2.loc[['0', '1', '2', '3', '4']]
            df2_t /= df2_t.sum().sum()
            df2_t += 0.000000001
            dfs['chi2'].loc[f1.stem, f2.stem] = chisquare(df1_t.values.flatten(), df2_t.values.flatten())[1]
            dfs['jensenshannon'].loc[f1.stem, f2.stem] = -jensenshannon(df1.values.flatten(), df2.values.flatten())
            dfs['mutual_info'].loc[f1.stem, f2.stem] = mutual_info_regression(df1.values.flatten().reshape(-1, 1), 
                                                                    df2.values.flatten())[0]
            dfs['emd'].loc[f1.stem, f2.stem] = -wasserstein_distance(range(108), range(108), df1.values.flatten(), df2.values.flatten())
    
    return dfs

def dists_bruno():
    df = pd.read_csv(Path(config['input_dir']) / "Bruno new" / "bruno_voice_analysis_octave_and_pitch_class.csv")
    df['Unnamed: 0'] = df['Unnamed: 0'].str[1:-5]
    df.set_index("Unnamed: 0", inplace=True)
    df.index.name = None
    df = df.apply(lambda row: row/row.sum(), axis=1)
    dfs = dict(jensenshannon = pd.DataFrame(),
                chi2 = pd.DataFrame(),
                mutual_info = pd.DataFrame(),
                emd = pd.DataFrame())
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            dfs['chi2'].loc[df.index[i], df.index[j]] = chisquare(df.iloc[i], df.iloc[j])[1]
            dfs['jensenshannon'].loc[df.index[i], df.index[j]] = -jensenshannon(df.iloc[i], df.iloc[j])
            dfs['mutual_info'].loc[df.index[i], df.index[j]] = mutual_info_regression(df.iloc[i].values.reshape(-1, 1), 
                                                                    df.iloc[j].values)[0]
            dfs['emd'].loc[df.index[i], df.index[j]] = -wasserstein_distance(range(73), range(73), df.iloc[i], df.iloc[j])
    return dfs

same_person = [('A D Part A 1.opus-wav', 'A D Part C 1.opus-wav'),
 ( 'Amir D T3-T1-wav', 'amir D T1T1-wav'),
 ('DNT1A1-wav', 'DNT3A1-wav'),
 ('FK1T1A1-wav', 'FK3T3A1-wav'),
 ('HG1T1A1-wav', 'HG3T3A1-wav'),
 ('I H Part A 1.opus-wav', 'I H Part C 1.opus-wav'),
 ('LLT1A1-wav', 'LLT3A1-wav'),
 ('MELT1A1-wav', 'MELT3A1-wav'),
 ('MK1T1A1-wav', 'MK3T3A1-wav'),
 ('N R part A 1.opus-wav', 'NR Part C 1.opus-wav'),
 ('NF3T3A1-wav', 'NFT1A1-wav'),
 ('NILT1A1-wav', 'NILT3A1-wav'),
 # ('RM3A1-wav', 'RMT1A1-wav'),
 # ('SBT1A1-wav', 'SBT3A1-wav'),
 ('SHILABT11', 'SHILABT31'),
 ('SHS PART A 1.opus-wav', 'SHS PART C 1.opus-wav'),
 ('YK1T1A1-wav', 'YK3T3A3-wav')]

def extract_upper_triangle(df):
    # Stack the DataFrame to get a multi-level index Series
    s = df.stack()

    # Filter to only keep correlations from the upper triangle
    s = s[s.index.get_level_values(0) < s.index.get_level_values(1)]

    # Convert the series to a DataFrame
    df_pairs = s.reset_index()
    df_pairs.columns = [1, 2, 'dist']
    return df_pairs[df_pairs[1] != df_pairs[2]]

def rocs(dfs, subdir = "us"):
    logger.info("rocs")
    (config['output_dir'] / subdir).mkdir(parents=True, exist_ok=True)
    dfs = {k : extract_upper_triangle(v) for k, v in dfs.items()}
    dfs = {k : v.set_index([1,2]) for k, v in dfs.items()}
    for k in dfs.keys():
        dfs[k].loc[same_person, 'GT'] = 2
    dfs = {k : v.reset_index().fillna(1) for k, v in dfs.items()}
    for k, v in dfs.items():
        v.to_csv(config['output_dir'] / subdir / f"{k}.csv", index=False)
        df = v.set_index([1,2])['dist'].unstack()
        df.index.name = None
        df.columns.name = None
        fig = px.imshow(df)
        fix_and_write(fig=fig, filename=k, height_factor=2, width_factor=2, output_dir=Path(config['output_dir']) / subdir)
    
    
    auc_scores = {}
    fig = go.Figure()
    for k, df in dfs.items():
        s = df['dist']
        # s = (s - s.min()) / (s.max() - s.min())
        fpr, tpr, thresholds = metrics.roc_curve(df['GT'], s, pos_label=2)
        fig.add_trace(go.Scatter(x=fpr, y=tpr,  name=k))
        fig.update_layout(title="Distances ROC Curve",
                     xaxis_title="FPR",
                     yaxis_title="TPR")
        auc_scores[k] = metrics.auc(fpr, tpr)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='random',
                           line=dict(color='black', dash='dash')))
    fix_and_write(fig=fig,
                  filename="roc",
                  output_dir=config['output_dir'] / subdir)     
    pd.Series(auc_scores).to_csv(config['output_dir'] / subdir / f"auc.csv")

def clustering(df):
  
    centroids, labels = kmeans2(df, 17, minit='points')
    s = pd.Series(labels, df.index)
    s.to_csv(config['output_dir'] / f"js_clusters.csv")

def main():
    # per_file_process()

    dfs = dists_bruno()
    rocs(dfs, 'bruno')
    dfs = dists()
    rocs(dfs, 'us')
    # clustering(dfs['jensenshannon'])
    
    
        
if __name__ == "__main__":             
    main()