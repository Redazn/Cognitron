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
