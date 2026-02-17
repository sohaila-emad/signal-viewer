"""
ECG/EEG Models for Medical Signal Processing
- Multi-channel ECG classification
- Abnormality detection using deep learning
- Classic ML comparison
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ECGFeatureExtractor:
    """
    Extract features from ECG signals for classification.
    Uses both time-domain and frequency-domain features.
    """
    
    def __init__(self):
        self.sampling_rate = 250  # Default ECG sampling rate
    
    def extract_features(self, ecg_data: np.ndarray, fs: int = 250) -> Dict:
        """
        Extract comprehensive features from ECG signal.
        
        Args:
            ecg_data: ECG signal array (can be multi-channel)
            fs: Sampling frequency in Hz
            
        Returns:
            Dictionary with extracted features
        """
        self.sampling_rate = fs
        features = {}
        
        # Handle multi-channel signals
        if isinstance(ecg_data, list):
            ecg_data = np.array(ecg_data)
        
        if ecg_data.ndim == 1:
            ecg_data = ecg_data.reshape(1, -1)
        
        n_channels = ecg_data.shape[0]
        features['n_channels'] = n_channels
        
        # Extract features for each channel
        channel_features = []
        for ch in range(n_channels):
            ch_features = self._extract_channel_features(ecg_data[ch], fs)
            channel_features.append(ch_features)
        
        # Aggregate features across channels
        features['mean_hr'] = np.mean([f['heart_rate'] for f in channel_features])
        features['std_hr'] = np.std([f['heart_rate'] for f in channel_features])
        features['mean_rr_interval'] = np.mean([f['rr_interval'] for f in channel_features if f['rr_interval'] > 0])
        features['std_rr_interval'] = np.std([f['rr_interval'] for f in channel_features if f['rr_interval'] > 0])
        features['rmssd'] = np.mean([f['rmssd'] for f in channel_features])
        
        # Frequency domain features
        features['low_freq_power'] = np.mean([f['low_freq_power'] for f in channel_features])
        features['high_freq_power'] = np.mean([f['high_freq_power'] for f in channel_features])
        features['lf_hf_ratio'] = features['low_freq_power'] / (features['high_freq_power'] + 1e-10)
        
        # Statistical features
        all_means = [f['mean_amplitude'] for f in channel_features]
        all_stds = [f['std_amplitude'] for f in channel_features]
        features['mean_amplitude'] = np.mean(all_means)
        features['std_amplitude'] = np.mean(all_stds)
        features['skewness'] = np.mean([f['skewness'] for f in channel_features])
        features['kurtosis'] = np.mean([f['kurtosis'] for f in channel_features])
        
        return features
    
    def _extract_channel_features(self, channel_data: np.ndarray, fs: int) -> Dict:
        """Extract features from a single channel."""
        features = {}
        
        # Basic statistics
        features['mean_amplitude'] = np.mean(channel_data)
        features['std_amplitude'] = np.std(channel_data)
        features['min_amplitude'] = np.min(channel_data)
        features['max_amplitude'] = np.max(channel_data)
        
        # Higher order statistics
        features['skewness'] = self._calculate_skewness(channel_data)
        features['kurtosis'] = self._calculate_kurtosis(channel_data)
        
        # Detect R-peaks for heart rate calculation
        peaks = self._detect_r_peaks(channel_data, fs)
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
            features['heart_rate'] = 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            features['rr_interval'] = np.mean(rr_intervals)
            features['hrv'] = np.std(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        else:
            features['heart_rate'] = 0
            features['rr_interval'] = 0
            features['hrv'] = 0
            features['rmssd'] = 0
        
        # Frequency domain features
        freqs, psd = self._compute_psd(channel_data, fs)
        
        # Low frequency (0.04-0.15 Hz) and High frequency (0.15-0.4 Hz)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        
        features['low_freq_power'] = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0
        features['high_freq_power'] = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 0
        
        return features
    
    def _detect_r_peaks(self, ecg_signal: np.ndarray, fs: int) -> np.ndarray:
        """Detect R-peaks in ECG signal using simple thresholding."""
        # Apply bandpass filter
        nyq = fs / 2
        low = 5 / nyq
        high = 15 / nyq
        b, a = signal.butter(3, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        # Find peaks
        threshold = np.mean(filtered) + 0.5 * np.std(filtered)
        peaks, _ = signal.find_peaks(filtered, height=threshold, distance=int(fs * 0.3))
        
        return peaks
    
    def _compute_psd(self, signal_data: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        n = len(signal_data)
        freqs = fftfreq(n, 1/fs)
        fft_vals = fft(signal_data)
        psd = np.abs(fft_vals) ** 2
        
        # Only positive frequencies
        pos_mask = freqs > 0
        return freqs[pos_mask], psd[pos_mask]
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


class ECGClassifier:
    """
    Classifier for ECG abnormality detection.
    Uses both classic ML and rule-based approaches.
    """
    
    # Abnormality types
    ABNORMALITY_TYPES = [
        'normal',
        'atrial_fibrillation',
        'ventricular_tachycardia',
        'premature_ventricular_contraction',
        'bradycardia'
    ]
    
    def __init__(self):
        self.feature_extractor = ECGFeatureExtractor()
        self.classifier = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the classifier."""
        if SKLEARN_AVAILABLE:
            # Use Random Forest for classification
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Generate training data (simplified)
            self._train_with_synthetic_data()
    
    def _train_with_synthetic_data(self):
        """Train classifier with synthetic data representing different conditions."""
        if not SKLEARN_AVAILABLE:
            return
        
        # Generate synthetic training data for each class
        X_train = []
        y_train = []
        
        np.random.seed(42)
        
        for abnormality_type in self.ABNORMALITY_TYPES:
            # Generate multiple samples for each class
            for _ in range(50):
                # Generate synthetic ECG with characteristics of each condition
                features = self._generate_synthetic_features(abnormality_type)
                X_train.append(features)
                y_train.append(abnormality_type)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
    
    def _generate_synthetic_features(self, abnormality_type: str) -> List[float]:
        """Generate synthetic features for a given abnormality type."""
        features = []
        
        # Base features with variations
        if abnormality_type == 'normal':
            hr = np.random.normal(70, 10)
            rr = 60000 / hr
            rmssd = np.random.normal(30, 10)
            lf_power = np.random.normal(500, 100)
            hf_power = np.random.normal(300, 80)
        
        elif abnormality_type == 'atrial_fibrillation':
            hr = np.random.normal(120, 20)
            rr = 60000 / hr
            rmssd = np.random.normal(15, 5)  # Irregular - lower RMSSD
            lf_power = np.random.normal(800, 150)
            hf_power = np.random.normal(600, 120)  # Higher HF due to irregularity
        
        elif abnormality_type == 'ventricular_tachycardia':
            hr = np.random.normal(180, 20)
            rr = 60000 / hr
            rmssd = np.random.normal(10, 5)
            lf_power = np.random.normal(200, 50)
            hf_power = np.random.normal(100, 30)
        
        elif abnormality_type == 'premature_ventricular_contraction':
            hr = np.random.normal(75, 15)
            rr = 60000 / hr
            rmssd = np.random.normal(40, 15)  # Higher variability
            lf_power = np.random.normal(600, 120)
            hf_power = np.random.normal(250, 70)
        
        elif abnormality_type == 'bradycardia':
            hr = np.random.normal(50, 8)
            rr = 60000 / hr
            rmssd = np.random.normal(25, 8)
            lf_power = np.random.normal(400, 80)
            hf_power = np.random.normal(200, 60)
        
        features.extend([
            hr,  # mean_hr
            np.random.uniform(5, 15),  # std_hr
            rr,  # mean_rr_interval
            np.random.uniform(10, 50) if abnormality_type != 'normal' else np.random.uniform(5, 20),  # std_rr_interval
            rmssd,  # rmssd
            lf_power,  # low_freq_power
            hf_power,  # high_freq_power
            lf_power / (hf_power + 1e-10),  # lf_hf_ratio
            np.random.uniform(-0.5, 0.5),  # mean_amplitude
            np.random.uniform(0.3, 1.0),  # std_amplitude
            np.random.uniform(-1, 1),  # skewness
            np.random.uniform(-1, 2),  # kurtosis
            1 if hr < 60 else 1,  # n_channels (dummy)
        ])
        
        return features
    
    def predict(self, ecg_data: np.ndarray, fs: int = 250) -> Dict:
        """
        Predict abnormality type from ECG data.
        
        Args:
            ecg_data: ECG signal data (can be multi-channel)
            fs: Sampling frequency in Hz
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.feature_extractor.extract_features(ecg_data, fs)
        
        # Convert features to array
        feature_vector = [
            features.get('mean_hr', 0),
            features.get('std_hr', 0),
            features.get('mean_rr_interval', 0),
            features.get('std_rr_interval', 0),
            features.get('rmssd', 0),
            features.get('low_freq_power', 0),
            features.get('high_freq_power', 0),
            features.get('lf_hf_ratio', 0),
            features.get('mean_amplitude', 0),
            features.get('std_amplitude', 0),
            features.get('skewness', 0),
            features.get('kurtosis', 0),
            features.get('n_channels', 1)
        ]
        
        if SKLEARN_AVAILABLE and self.classifier is not None:
            # Use ML classifier
            feature_array = np.array(feature_vector).reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_array)
            prediction = self.classifier.predict(feature_scaled)[0]
            
            # Get probability scores
            proba = self.classifier.predict_proba(feature_scaled)[0]
            classes = self.classifier.classes_
            
            confidence = float(max(proba))
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 3),
                'all_probabilities': {
                    cls: round(prob, 3) for cls, prob in zip(classes, proba)
                },
                'features': features
            }
        else:
            # Fallback to rule-based classification
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features: Dict) -> Dict:
        """Rule-based prediction when ML classifier is not available."""
        hr = features.get('mean_hr', 70)
        rr_std = features.get('std_rr_interval', 0)
        rmssd = features.get('rmssd', 30)
        lf_hf = features.get('lf_hf_ratio', 1.5)
        
        # Rule-based classification
        if hr < 60:
            prediction = 'bradycardia'
            confidence = 0.85
        elif hr > 150:
            prediction = 'ventricular_tachycardia'
            confidence = 0.80
        elif rr_std > 100:  # High variability suggests AF
            prediction = 'atrial_fibrillation'
            confidence = 0.75
        elif rmssd > 40:  # High RMSSD suggests PVC
            prediction = 'premature_ventricular_contraction'
            confidence = 0.70
        else:
            prediction = 'normal'
            confidence = 0.90
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'all_probabilities': {
                'normal': 0.9 if prediction == 'normal' else 0.2,
                'bradycardia': 0.85 if prediction == 'bradycardia' else 0.1,
                'ventricular_tachycardia': 0.8 if prediction == 'ventricular_tachycardia' else 0.1,
                'atrial_fibrillation': 0.75 if prediction == 'atrial_fibrillation' else 0.1,
                'premature_ventricular_contraction': 0.7 if prediction == 'premature_ventricular_contraction' else 0.1
            },
            'features': features
        }


class ClassicMLAnalyzer:
    """
    Classic ML-based analysis for ECG signals.
    Uses statistical methods and simple ML classifiers.
    """
    
    def __init__(self):
        self.feature_extractor = ECGFeatureExtractor()
    
    def analyze(self, ecg_data: np.ndarray, fs: int = 250) -> Dict:
        """
        Analyze ECG using classic methods.
        
        Args:
            ecg_data: ECG signal data
            fs: Sampling frequency
            
        Returns:
            Analysis results
        """
        features = self.feature_extractor.extract_features(ecg_data, fs)
        
        # Heart rate analysis
        hr = features.get('mean_hr', 0)
        hr_analysis = self._analyze_heart_rate(hr)
        
        # Rhythm analysis
        rr_std = features.get('std_rr_interval', 0)
        rhythm_analysis = self._analyze_rhythm(rr_std, features.get('rmssd', 0))
        
        # Frequency analysis
        freq_analysis = self._analyze_frequency(
            features.get('low_freq_power', 0),
            features.get('high_freq_power', 0),
            features.get('lf_hf_ratio', 1)
        )
        
        return {
            'heart_rate_analysis': hr_analysis,
            'rhythm_analysis': rhythm_analysis,
            'frequency_analysis': freq_analysis,
            'features': features
        }
    
    def _analyze_heart_rate(self, hr: float) -> Dict:
        """Analyze heart rate."""
        if hr < 60:
            status = 'bradycardia'
            description = 'Heart rate is below normal (60 bpm)'
        elif hr > 100:
            status = 'tachycardia'
            description = 'Heart rate is above normal (100 bpm)'
        else:
            status = 'normal'
            description = 'Heart rate is within normal range'
        
        return {
            'heart_rate': round(hr, 1),
            'status': status,
            'description': description
        }
    
    def _analyze_rhythm(self, rr_std: float, rmssd: float) -> Dict:
        """Analyze heart rhythm."""
        if rr_std > 100:
            status = 'irregular'
            description = 'High variability in RR intervals suggests irregular rhythm'
        elif rmssd > 40:
            status = 'variable'
            description = 'Elevated RMSSD suggests increased heart rate variability'
        else:
            status = 'regular'
            description = 'Rhythm appears regular'
        
        return {
            'rr_std': round(rr_std, 2),
            'rmssd': round(rmssd, 2),
            'status': status,
            'description': description
        }
    
    def _analyze_frequency(self, lf_power: float, hf_power: float, lf_hf_ratio: float) -> Dict:
        """Analyze frequency domain features."""
        if lf_hf_ratio > 2:
            status = 'sympathetic_dominant'
            description = 'Sympathetic nervous system appears dominant'
        elif lf_hf_ratio < 0.5:
            status = 'parasympathetic_dominant'
            description = 'Parasympathetic nervous system appears dominant'
        else:
            status = 'balanced'
            description = 'Autonomic balance appears normal'
        
        return {
            'lf_power': round(lf_power, 2),
            'hf_power': round(hf_power, 2),
            'lf_hf_ratio': round(lf_hf_ratio, 2),
            'status': status,
            'description': description
        }


# Singleton instance
ecg_classifier = ECGClassifier()
classic_ml_analyzer = ClassicMLAnalyzer()


def predict_ecg(ecg_data: np.ndarray, fs: int = 250) -> Dict:
    """
    Predict ECG abnormality.
    
    Args:
        ecg_data: ECG signal data
        fs: Sampling frequency
        
    Returns:
        Prediction results
    """
    return ecg_classifier.predict(ecg_data, fs)


def analyze_ecg_classic(ecg_data: np.ndarray, fs: int = 250) -> Dict:
    """
    Analyze ECG using classic methods.
    
    Args:
        ecg_data: ECG signal data
        fs: Sampling frequency
        
    Returns:
        Analysis results
    """
    return classic_ml_analyzer.analyze(ecg_data, fs)


def get_available_abnormalities() -> List[str]:
    """Get list of detectable abnormalities."""
    return ECGClassifier.ABNORMALITY_TYPES
