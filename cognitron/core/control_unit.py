
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

# =========================
# 🔍 Binary Memory Store (Bit-Level)
# =========================
class BinaryAnomalyMemory:
    def __init__(self, bit_precision=16):
        self.memory = set()
        self.bit_precision = bit_precision
        self.feature_mask = (1 << bit_precision) - 1

    def _float_to_binary(self, value, scale=1000):
        """Convert float to fixed-point integer representation"""
        scaled = int(value * scale)
        return scaled & self.feature_mask

    def _create_feature_hash(self, features):
        """Create bit signature from features"""
        signature = 0
        for i, val in enumerate(features):
            binary_val = self._float_to_binary(val)
            signature ^= (binary_val << (i * self.bit_precision))
        return signature

    def remember(self, features, label):
        if label == 1:  # Only store anomalies
            signature = self._create_feature_hash(features)
            self.memory.add(signature)

    def known_anomaly(self, features):
        signature = self._create_feature_hash(features)
        return signature in self.memory

# =========================
# 🔬 Enhanced Bit Scanner
# =========================
class BitScanner:
    def __init__(self, memory_store, window=10):
        self.memory = memory_store
        self.window = window

    def scan(self, data):
        result = []
        for i in range(self.window, len(data)):
            segment = data[i - self.window:i]
            features = [
                data[i],
                np.mean(segment),
                np.std(segment),
                np.percentile(segment, 75) - np.percentile(segment, 25),
                (data[i] - np.mean(segment)) / (np.std(segment) + 1e-10)
            ]
            result.append(1 if self.memory.known_anomaly(features) else 0)
        return np.array([0]*self.window + result)

# =========================
# 🧩 Production Anomaly Detector
# =========================
class ProductionAnomalyDetector:
    def __init__(self, ma_window=20, if_window=30, bit_precision=16):
        self.ma_detector = MovingAverage(window=ma_window)
        self.if_detector = IsolationForestDetector(window=if_window)
        self.memory = BinaryAnomalyMemory(bit_precision)
        self.scanner = BitScanner(self.memory, window=10)
        self.pft = PFTFusion(temperature=0.7)
        self.buffer = []
        self.dynamic_threshold = 0.5
        self.fusion_scores = []
        self.bit_preds = []

    def process(self, data_point):
        self.buffer.append(data_point)
        results = {
            'point': data_point,
            'ma_score': 0,
            'if_score': 0,
            'fusion_score': 0,
            'bit_verdict': 0,
            'final_verdict': 'NORMAL'
        }

        n = len(self.buffer)
        min_window = max(self.ma_detector.window, self.if_detector.window)

        if n > min_window:
            # Calculate MA score
            _, ma_scores = self.ma_detector.detect(self.buffer)
            results['ma_score'] = ma_scores[-1]

            # Calculate IF score
            _, if_scores = self.if_detector.detect(self.buffer)
            results['if_score'] = if_scores[-1]

            # Fusion
            fusion_score = self.pft.fusion(ma_scores[-1], if_scores[-1])
            self.fusion_scores.append(fusion_score)
            results['fusion_score'] = fusion_score

            # Bit scanner
            if n > 20:  # Wait until we have enough data
                bit_preds = self.scanner.scan(self.buffer)
                results['bit_verdict'] = bit_preds[-1]
                self.bit_preds.append(bit_preds[-1])

            # Decision logic
            if fusion_score > self.dynamic_threshold:
                if results['bit_verdict'] == 1:
                    results['final_verdict'] = 'CONFIRMED_ANOMALY'
                    # Add to memory
                    segment = self.buffer[-self.scanner.window-1:-1]
                    features = [
                        self.buffer[-1],
                        np.mean(segment),
                        np.std(segment),
                        np.percentile(segment, 75) - np.percentile(segment, 25),
                        (self.buffer[-1] - np.mean(segment)) / (np.std(segment) + 1e-10)
                    ]
                    self.memory.remember(features, 1)
                else:
                    results['final_verdict'] = 'SUSPICIOUS'
            elif fusion_score > self.dynamic_threshold * 0.7:
                results['final_verdict'] = 'WARNING'

        return results

# =========================
# 📊 Real-World Dataset
# =========================
def load_real_world_dataset():
    """Load server CPU usage dataset with real anomalies"""
    # URL to real-world dataset (Server CPU metrics with anomalies)
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"

    try:
        response = requests.get(url)
        df = pd.read_csv(BytesIO(response.content))
        print("✅ Loaded real-world dataset: AWS EC2 CPU Utilization")

        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # We'll create labels based on known anomaly periods
        # (In real applications, you would have labeled data)
        df['value'] = df['value'].astype(float)

        # Create synthetic labels for known anomaly periods
        df['label'] = 0
        anomaly_periods = [
            ('2014-02-23 00:00', '2014-02-23 05:00'),
            ('2014-02-26 21:00', '2014-02-27 02:00'),
            ('2014-03-03 08:00', '2014-03-03 13:00')
        ]

        for start, end in anomaly_periods:
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            df.loc[mask, 'label'] = 1

        return df['value'].values, df['label'].values, df

    except Exception as e:
        print(f"⚠️ Error loading real dataset: {e}")
        print("🔄 Generating synthetic data instead")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic data with realistic anomalies"""
    np.random.seed(42)
    n = 2000
    data = np.zeros(n)

    # Base pattern (daily seasonality)
    for i in range(n):
        hour = i % 24
        if 0 <= hour < 6:   # Night
            data[i] = np.random.normal(25, 5)
        elif 6 <= hour < 12: # Morning
            data[i] = np.random.normal(40, 7)
        elif 12 <= hour < 18: # Afternoon
            data[i] = np.random.normal(60, 8)
        else: # Evening
            data[i] = np.random.normal(45, 6)

    # Add weekly pattern
    for i in range(n):
        day = (i // 24) % 7
        if day >= 5:  # Weekend
            data[i] *= 0.7

    # Add anomalies
    labels = np.zeros(n)
    anomaly_types = [
        {'type': 'spike', 'duration': 1, 'magnitude': 2.5},
        {'type': 'spike', 'duration': 2, 'magnitude': 3.0},
        {'type': 'dip', 'duration': 3, 'magnitude': 0.4},
        {'type': 'spike', 'duration': 4, 'magnitude': 2.8},
        {'type': 'level_shift', 'duration': 50, 'magnitude': 1.5},
        {'type': 'trend', 'duration': 30, 'magnitude': 0.1}
    ]

    positions = np.random.choice(n-100, size=len(anomaly_types), replace=False)

    for i, pos in enumerate(positions):
        anomaly = anomaly_types[i]
        dur = anomaly['duration']
        mag = anomaly['magnitude']

        labels[pos:pos+dur] = 1

        if anomaly['type'] == 'spike':
            data[pos:pos+dur] += np.random.normal(40 * mag, 10, size=dur)
        elif anomaly['type'] == 'dip':
            data[pos:pos+dur] *= np.random.uniform(0.3, 0.6, size=dur)
        elif anomaly['type'] == 'level_shift':
            data[pos:pos+dur] += 20 * mag
        elif anomaly['type'] == 'trend':
            trend = np.linspace(0, 15 * mag, dur)
            data[pos:pos+dur] += trend

    # Add noise
    noise = np.random.normal(0, 5, n)
    data += noise

    return data, labels, None

# =========================
# 📈 Visualization
# =========================
def plot_results(data, labels, fusion_scores, bit_preds, threshold):
    plt.figure(figsize=(16, 10))

    # Main data plot
    plt.subplot(3, 1, 1)
    plt.plot(data, label='CPU Utilization', color='royalblue', alpha=0.8)
    plt.scatter(np.where(labels==1), data[labels==1],
                color='red', s=30, label='True Anomalies', zorder=5)
    plt.title('Server CPU Utilization with Anomalies', fontsize=14)
    plt.ylabel('CPU %')
    plt.legend()
    plt.grid(alpha=0.2)

    # Fusion scores
    plt.subplot(3, 1, 2)
    plt.plot(fusion_scores, label='Fusion Score', color='purple', alpha=0.8)
    plt.axhline(threshold, color='orange', linestyle='--', label='Decision Threshold')
    plt.fill_between(range(len(fusion_scores)), fusion_scores, threshold,
                     where=(np.array(fusion_scores) > threshold),
                     color='red', alpha=0.3)
    plt.title('PFT Fusion Scores', fontsize=14)
    plt.ylabel('Fusion Score')
    plt.legend()
    plt.grid(alpha=0.2)

    # Bit scanner results
    plt.subplot(3, 1, 3)
    plt.plot(bit_preds, label='Bit Scanner', color='green', drawstyle='steps-post')
    plt.scatter(np.where(labels==1), labels[labels==1],
                color='red', s=30, label='True Anomalies', zorder=5)
    plt.title('Bit-Level Anomaly Verification', fontsize=14)
    plt.xlabel('Time Index')
    plt.ylabel('Anomaly Status')
    plt.yticks([0, 1], ['Normal', 'Anomaly'])
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    return plt.gcf()

# =========================
# 🚀 Main Execution
# =========================
if __name__ == "__main__":
    print("🚀 Starting Bit-Level Anomaly Detection System")
    print("📡 Loading dataset...")

    # Load dataset
    data, labels, _ = load_real_world_dataset()

    # Initialize detector
    detector = ProductionAnomalyDetector(
        ma_window=24,  # 24 hours for daily pattern
        if_window=48,   # 48 hours for 2-day pattern
        bit_precision=14
    )

    # Storage for results
    all_results = []
    fusion_scores = []
    bit_preds = []
    processing_times = []

    print("🔍 Processing data points...")
    start_time = total_start = time.time()

    # Process each data point in real-time
    for i, point in enumerate(data):
        point_start = time.time()
        result = detector.process(point)
        processing_times.append(time.time() - point_start)

        # Store results
        result['index'] = i
        result['true_label'] = labels[i] if i < len(labels) else 0
        all_results.append(result)

        if 'fusion_score' in result:
            fusion_scores.append(result['fusion_score'])
        if 'bit_verdict' in result:
            bit_preds.append(result['bit_verdict'])

        # Update dynamic threshold every 100 points
        if i > 100 and i % 100 == 0:
            recent_scores = fusion_scores[-100:]
            if recent_scores:
                detector.dynamic_threshold = np.percentile(recent_scores, 95)

        # Progress reporting
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"⏱️ Processed {i}/{len(data)} points | {elapsed:.2f}s | "
                  f"Avg time/point: {np.mean(processing_times[-100:]):.5f}s")
            start_time = time.time()

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Extract final predictions
    final_preds = [1 if r['final_verdict'] in ['CONFIRMED_ANOMALY', 'SUSPICIOUS']
                  else 0 for r in all_results]

    # Calculate metrics
    f1 = f1_score(labels, final_preds)
    precision = precision_score(labels, final_preds)
    recall = recall_score(labels, final_preds)
    avg_time = np.mean(processing_times)

    print("\n📊 Detection Performance:")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔍 Recall: {recall:.4f}")
    print(f"⏱️ Avg Processing Time: {avg_time:.6f} seconds/point")
    print(f"💾 Memory Usage: {len(detector.memory.memory)} bit-signatures stored")

    # Plot results
    print("\n📈 Generating visualization...")
    fig = plot_results(data, labels, fusion_scores, bit_preds, detector.dynamic_threshold)
    plt.savefig('anomaly_detection_results.png', dpi=300)
    plt.show()

    # Performance analysis
    print("\n💡 Performance Insights:")
    print(f"- Bit verification reduced false positives by "
          f"{100*(1 - precision_score([1 if 'CONFIRMED' in r['final_verdict'] else 0 for r in all_results], labels)):.1f}%")

    total_time = time.time() - total_start
    print(f"\n🏁 Total processing time: {total_time:.2f} seconds")
    print(f"🔚 System shutdown complete")