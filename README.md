# HMM-Based Machine Sensor Anomaly Detector

## 1. Project Overview

This project implements a machine sensor anomaly detection system using a Gaussian Hidden Markov Model (HMM).

The system analyzes sequential sensor readings, including temperature and vibration, to identify unusual machine behavior that may indicate potential equipment problems.

The project uses Python and the `hmmlearn` library to train the model on normal operating data and detect anomalies in a continuous test stream.

**Dataset Note:** Since no college-provided dataset was available, a reproducible simulated sensor dataset was generated using `generate_data.py`. It contains normal machine operating patterns and deliberately injected anomalies for testing.

## 2. Project Objectives

- Train a Gaussian Hidden Markov Model.
- Analyze sequential machine sensor readings.
- Calculate log-likelihood scores.
- Establish an anomaly detection threshold.
- Implement real-time sliding-window detection.
- Visualize detected anomalies.
- Deploy the detection system using FastAPI.

## 3. Technologies Used

- Python 3.11
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- hmmlearn
- Joblib
- FastAPI
- Uvicorn

## 4. Dataset Description

Two datasets are generated:

**Normal Training Dataset**
- 25 separate normal operating sequences.
- 250 readings per sequence.
- Total: 6,250 readings.
- Features: Temperature and Vibration.

**Test Dataset**
- 1,800 sequential sensor readings.
- Contains normal readings and deliberately injected anomalies.
- Includes actual anomaly labels for evaluation.

## 5. Data Preprocessing

The following preprocessing steps are performed:

1. Load the generated sensor datasets.
2. Extract temperature and vibration features.
3. Apply StandardScaler to normalize the sensor values.
4. Preserve separate training sequences for HMM training.
5. Prepare the continuous test stream for anomaly detection.

## 6. Gaussian Hidden Markov Model

A Hidden Markov Model is a probabilistic model that identifies hidden operating states from observed sensor readings.

The model uses:

- Hidden states to represent different machine operating conditions.
- Gaussian emissions to model temperature and vibration.
- Transition probabilities to represent changes between operating states.
- Log-likelihood to measure how well a sequence matches normal operating behavior.

The `hmmlearn` library is used to implement Gaussian HMMs.

## 7. Model Selection

Three models were compared using different numbers of hidden states:

- 2 hidden states
- 3 hidden states
- 4 hidden states

The Bayesian Information Criterion (BIC) was used to select the best model.

The model with the lowest BIC was selected because BIC balances model fit and complexity.

**Selected Model: 4 Hidden States**

The selected model represents the best balance among the tested candidates according to the BIC comparison.

The model comparison results are saved in:

`evaluation/model_selection.csv`

## 8. Anomaly Threshold Calculation

The trained HMM calculates the average log-likelihood of normal training windows.

The 5th percentile of the normal training-window likelihood distribution is selected as the anomaly threshold.

A test window is classified as anomalous when its average log-likelihood is below this threshold.

Lower likelihood indicates that the observed sensor pattern