# Q2behave/FidgetSense ML pipeline


# Windowed Feature Extraction for IMU-based Behavioral Analysis

/machine_learning/feature_extraction/windowed_full_activity_features.py

## Overview
This script performs **windowed feature extraction** on accelerometer and gyroscope (IMU) data collected from behavioral experiments. It processes full activity recordings, applies filtering and moving-average smoothing, extracts **power spectral density (PSD) features**, **statistical features**, and combines them into per-window feature sets. The resulting CSV outputs can be used for machine learning or behavioral analysis.

The pipeline handles multiple participants, activity types, and automatically maps behaviors from file names.

---

## Key Features

1. **Preprocessing / Filtering**
   - Applies **detrending**, **band-pass filtering (1–5 Hz)**, and **moving average smoothing** to IMU signals.
   - Computes **max acceleration** and **max gyroscope magnitude** per sample.
   - Designed to reduce drift and high-frequency noise before feature extraction.

2. **Windowed Feature Extraction**
   - Splits each activity recording into **overlapping windows**.
   - Extracts **PSD features** using Welch’s method.
   - Extracts **statistical features**: mean, standard deviation, skewness, kurtosis, and other metrics.
   - Combines PSD and statistical features into a **single feature dataframe per window**.

3. **Behavior Mapping**
   - Maps behavior labels from filenames (`NoHyp`, `chair`, `Manipulating`, etc.) to numeric codes.
   - Supports automated handling of “Talking” behaviors and ignores small files (<30 rows).

4. **Metadata Handling**
   - Adds **participant ID**, **activity ID**, and **filename** to each feature dataframe for traceability.

5. **Output**
   - Three CSV files per run:
     - `stat_features_two_second.csv` → statistical features
     - `psd_features_two_second.csv` → PSD features
     - `all_features_two_second.csv` → combined features

---

## Requirements

- Python 3.9+
- Libraries:
  ```text
  pandas, numpy, scipy, matplotlib, plotly