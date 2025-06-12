import yaml, numpy as np, librosa

# -------------------------------------------------
# 1. CONFIG
# -------------------------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------------------------
# 2. FEATURE EXTRACTION
# -------------------------------------------------
def extract_logmel(
        y, sr,
        n_fft:       int,
        hop_length:  int,
        n_mels:      int) -> np.ndarray:
    """
    Returns [n_mels, T] log-Mel spectrogram in *float32*.
    """
    S = librosa.feature.melspectrogram(
            y, sr=sr, n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def make_windows(logmel: np.ndarray, context: int) -> np.ndarray:
    """
    Stack <context> consecutive frames → [N, context*n_mels]
    """
    F, T = logmel.shape
    if T < context:        # edge-case
        return np.empty((0, F*context), np.float32)

    win = np.lib.stride_tricks.sliding_window_view(
              logmel, window_shape=(1, context), axis=1)
    win = win.squeeze(0)          # shape (F, N, context)
    win = win.transpose(1, 0, 2)  # (N, F, context)
    win = win.reshape(win.shape[0], -1)
    return win.astype(np.float32)
