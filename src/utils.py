import os, yaml, csv
import numpy as np
import librosa
from scipy.stats import gamma

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    return yaml.safe_load(open(path, "r"))

def extract_logmel(path, sr, n_fft, hop_length, n_mels):
    wav, _ = librosa.load(path, sr=sr)
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(S)

def make_windows(logmel, context):
    frames = []
    for i in range(logmel.shape[1] - context + 1):
        win = logmel[:, i:i+context].reshape(-1)
        frames.append(win)
    return np.stack(frames)

def fit_gamma_threshold(scores, percentile):
    a, loc, scale = gamma.fit(scores, floc=0)
    return float(gamma.ppf(percentile/100.0, a, loc=loc, scale=scale))

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
