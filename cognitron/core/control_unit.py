
!pip install numpy pandas matplotlib scikit-learn requests

import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import time

# ==========================================
# 🧠 Enhanced Bit-Level Anomaly Detection System
# ==========================================

# =========================
# 📦 Moving Average Detector
# =========================
class MovingAverage:
    def __init__(self, window=20, threshold=3.0):
        self.window = window
        self.threshold = threshold

    def detect(self, data):
        df = pd.Series(data)
        moving_mean = df.rolling(window=self.window).mean()
        moving_std = df.rolling(window=self.window).std()
        z_scores = abs((df - moving_mean) / (moving_std + 1e-10))
        anomalies = z_scores > self.threshold
        return anomalies.values, z_scores.values

# =========================
# 🌲 Isolation Forest Detector
# =========================
class IsolationForestDetector:
    def __init__(self, window=20, contamination=0.1):
        self.window = window
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect(self, data):
        features = []
        for i in range(len(data)):
            if i < self.window:
                features.append([data[i], 0, 0, 0, 0])
                continue
            window_data = data[i - self.window:i]
            features.append([
                data[i],
                np.mean(window_data),
                np.std(window_data),
                np.percentile(window_data, 75) - np.percentile(window_data, 25),
                (data[i] - np.mean(window_data)) / (np.std(window_data) + 1e-10)
            ])
        features = np.array(features)
        self.model.fit(features)
        scores = -self.model.score_samples(features)
        predictions = self.model.predict(features) == -1
        return predictions, scores

# =========================
# ♻️ PFT Fusion Logic
# =========================
class PFTFusion:
    def __init__(self, temperature=0.5, window_size=10):
        self.T = temperature
        self.entropy_window = []
        self.window_size = window_size

    def update_entropy(self, value):
        self.entropy_window.append(value)
        if len(self.entropy_window) > self.window_size:
            self.entropy_window.pop(0)

    def compute_entropy(self):
        if len(self.entropy_window) < 2:
            return 0
        p = np.abs(self.entropy_window) / np.sum(np.abs(self.entropy_window))
        return -np.sum(p * np.log(p + 1e-10))

    def fusion(self, a, b):
        self.update_entropy(a)
        e = self.compute_entropy()
        F = (a - b) - self.T * e
        return np.tanh(F)


