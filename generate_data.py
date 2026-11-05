"""Generate reproducible simulated machine-sensor data for the HMM assignment."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Three normal machine operating regimes: idle, normal operation, high load.
MEANS = np.array([
    [35.0, 0.20],   # temperature, vibration
    [60.0, 0.65],
    [78.0, 1.05],
])
STDS = np.array([
    [2.0, 0.07],
    [3.0, 0.12],
    [3.5, 0.15],
])
TRANSITION = np.array([
    [0.92, 0.07, 0.01],
    [0.05, 0.90, 0.05],
    [0.02, 0.10, 0.88],
])

def make_normal_sequence(length, rng):
    states = np.zeros(length, dtype=int)
    states[0] = 0
    for i in range(1, length):
        states[i] = rng.choice(3, p=TRANSITION[states[i - 1]])
    readings = np.vstack([
        rng.normal(MEANS[state], STDS[state])
        for state in states
    ])
    return readings, states

def main():
    rng = np.random.default_rng(42)

    # Separate normal sequences are retained as separate HMM sequences.
    train_rows = []
    for sequence_id in range(25):
        readings, states = make_normal_sequence(250, rng)
        for t, (temp, vibration) in enumerate(readings):
            train_rows.append({
                "sequence_id": sequence_id,
                "time": t,
                "temperature": temp,
                "vibration": vibration,
                "state": int(states[t]),
            })
    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(DATA_DIR / "normal_sensor_data.csv", index=False)

    # One continuous test stream. Abnormal regions are injected deliberately.
    test_readings, test_states = make_normal_sequence(1800, rng)
    labels = np.zeros(len(test_readings), dtype=int)

    anomaly_ranges = [(420, 490), (960, 1035), (1450, 1520)]
    for start, end in anomaly_ranges:
        labels[start:end] = 1
        # Simulate overheating and excessive vibration.
        test_readings[start:end, 0] += rng.normal(28, 4, end - start)
        test_readings[start:end, 1] += rng.normal(1.25, 0.20, end - start)

    test_df = pd.DataFrame({
        "time": np.arange(len(test_readings)),
        "temperature": test_readings[:, 0],
        "vibration": test_readings[:, 1],
        "is_anomaly": labels,
        "normal_state": test_states,
    })
    test_df.to_csv(DATA_DIR / "test_sensor_stream.csv", index=False)

    print("Simulated sensor datasets generated successfully.")
    print(f"Normal training rows: {len(train_df)}")
    print(f"Test stream rows: {len(test_df)}")
    print(f"Injected anomaly readings: {int(labels.sum())}")
    print(f"Files saved in: {DATA_DIR}")

if __name__ == "__main__":
    main()
