"""Shared HMM scoring and sliding-window detection functions."""
import numpy as np

def make_windows(array, window_size=40, step_size=5):
    """Return overlapping windows and their start positions."""
    windows, starts = [], []
    for start in range(0, len(array) - window_size + 1, step_size):
        windows.append(array[start:start + window_size])
        starts.append(start)
    return windows, starts

def window_log_likelihood(model, windows):
    """Average log-likelihood per observation for each window."""
    scores = []
    for window in windows:
        scores.append(model.score(window) / len(window))
    return np.asarray(scores)

def detect_stream(model, scaler, readings, threshold,
                  window_size=40, step_size=5):
    """Score a continuous stream and mark every point in anomalous windows."""
    scaled = scaler.transform(np.asarray(readings, dtype=float))
    windows, starts = make_windows(scaled, window_size, step_size)
    scores = window_log_likelihood(model, windows)
    flags = scores < threshold

    point_flags = np.zeros(len(scaled), dtype=int)
    for start, flagged in zip(starts, flags):
        if flagged:
            point_flags[start:start + window_size] = 1

    return scores, np.asarray(starts), flags.astype(int), point_flags
