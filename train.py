"""Train and evaluate a Gaussian HMM anomaly detector."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from detector import make_windows, window_log_likelihood, detect_stream

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
EVAL = ROOT / "evaluation"
MODELS.mkdir(exist_ok=True)
EVAL.mkdir(exist_ok=True)

FEATURES = ["temperature", "vibration"]
WINDOW_SIZE = 40
STEP_SIZE = 5

def main():
    train_df = pd.read_csv(DATA / "normal_sensor_data.csv")
    test_df = pd.read_csv(DATA / "test_sensor_stream.csv")

    # Preserve sequence boundaries so transitions are not learned across files.
    sequences = []
    lengths = []
    for _, group in train_df.groupby("sequence_id", sort=True):
        seq = group[FEATURES].to_numpy(dtype=float)
        sequences.append(seq)
        lengths.append(len(seq))
    X_train = np.vstack(sequences)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaled_sequences = [scaler.transform(seq) for seq in sequences]
    X_test = test_df[FEATURES].to_numpy(dtype=float)

    # Select hidden-state count using BIC on normal training sequences.
    # Lower BIC is preferred; candidates are 2, 3 and 4 hidden states.
    candidates = [2, 3, 4]
    selection_rows = []
    best_model, best_states, best_bic = None, None, float("inf")
    n_samples, n_features = X_train_scaled.shape

    for n_states in candidates:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=300,
            tol=0.001,
            random_state=42,
            min_covar=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_scaled, lengths)

        log_likelihood = model.score(X_train_scaled, lengths)
        # Number of free parameters for diagonal-covariance Gaussian HMM.
        n_params = (
            (n_states - 1)
            + n_states * (n_states - 1)
            + n_states * n_features
            + n_states * n_features
        )
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        selection_rows.append({
            "hidden_states": n_states,
            "log_likelihood": log_likelihood,
            "parameters": n_params,
            "BIC": bic,
        })
        print(f"States={n_states} | Log likelihood={log_likelihood:.2f} | BIC={bic:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_states = n_states

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(EVAL / "model_selection.csv", index=False)

    # Build normal training windows and derive a low-likelihood threshold.
    training_scores = []
    for seq in scaled_sequences:
        windows, _ = make_windows(seq, WINDOW_SIZE, STEP_SIZE)
        training_scores.extend(window_log_likelihood(best_model, windows))
    threshold = float(np.percentile(training_scores, 5))

    # Detect anomalous regions in the continuous test stream.
    test_scores, starts, window_flags, point_flags = detect_stream(
        best_model, scaler, X_test, threshold, WINDOW_SIZE, STEP_SIZE
    )
    y_true = test_df["is_anomaly"].to_numpy(dtype=int)

    metrics = {
        "selected_hidden_states": int(best_states),
        "selection_method": "Bayesian Information Criterion (BIC)",
        "best_bic": float(best_bic),
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, point_flags)),
        "precision": float(precision_score(y_true, point_flags, zero_division=0)),
        "recall": float(recall_score(y_true, point_flags, zero_division=0)),
        "f1_score": float(f1_score(y_true, point_flags, zero_division=0)),
        "roc_auc_not_reported": "Thresholded point predictions; likelihood scores are window-level.",
    }
    with open(EVAL / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    result_df = test_df.copy()
    result_df["predicted_anomaly"] = point_flags
    result_df.to_csv(EVAL / "detection_results.csv", index=False)

    # Save model and preprocessing together for consistent API predictions.
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "threshold": threshold,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "features": FEATURES,
        "hidden_states": best_states,
    }, MODELS / "hmm_anomaly_model.joblib")

    # Plot raw readings and shade detected anomaly regions.
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    times = test_df["time"].to_numpy()
    axes[0].plot(times, test_df["temperature"], linewidth=1, label="Temperature")
    axes[0].set_ylabel("Temperature")
    axes[0].set_title("Sensor readings with detected anomaly regions")
    axes[0].legend()
    axes[1].plot(times, test_df["vibration"], linewidth=1, label="Vibration")
    axes[1].set_ylabel("Vibration")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    # Shade contiguous predicted anomaly intervals on both plots.
    for ax in axes:
        in_region = False
        start = 0
        for i, flag in enumerate(point_flags):
            if flag and not in_region:
                start, in_region = i, True
            elif not flag and in_region:
                ax.axvspan(times[start], times[i - 1], alpha=0.2, color="red")
                in_region = False
        if in_region:
            ax.axvspan(times[start], times[-1], alpha=0.2, color="red")
    fig.tight_layout()
    fig.savefig(EVAL / "anomaly_detection_plot.png", dpi=160)
    plt.close(fig)

    # Confusion matrix image.
    cm = confusion_matrix(y_true, point_flags)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Normal", "Anomaly"])
    ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(EVAL / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nMODEL SELECTION")
    print(selection_df.to_string(index=False))
    print(f"\nSelected hidden states: {best_states}")
    print(f"Anomaly threshold (average log-likelihood): {threshold:.4f}")
    print("\nEVALUATION")
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"{key}: {metrics[key]:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, point_flags, target_names=["Normal", "Anomaly"], zero_division=0))
    print("\nModel saved: models/hmm_anomaly_model.joblib")
    print("Graphs and result files saved in evaluation/")

if __name__ == "__main__":
    main()
