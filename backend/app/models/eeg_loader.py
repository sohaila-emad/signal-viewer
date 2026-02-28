"""
EEG Model Loader for BIOT (Deep Learning) and Classical ML (Random Forest)
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Import BIOT from braindecode
try:
    from braindecode.models import BIOT
    BIOT_AVAILABLE = True
except ImportError:
    BIOT_AVAILABLE = False
    print("⚠ braindecode not installed. BIOT model will not be available.")

# For classical ML
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# For resampling and feature extraction
from scipy import signal as sp_signal

CONFIDENCE_THRESHOLD = 0.5

class BIOTLoader:
    """
    Loader for fine‑tuned BIOT model.
    Expects a model trained on 18 channels, 200 Hz, 10‑second windows (2000 samples).
    """
    
    # Mapping from class indices to abnormality names (6 classes)
    ABNORMALITY_CLASSES = {
        0: 'normal',
        1: 'seizure',
        2: 'alcoholism',
        3: 'motor_abnormality',
        4: 'mental_stress',
        5: 'epileptic_interictal'
    }

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.num_classes = len(self.ABNORMALITY_CLASSES)
        
        # Model parameters (must match training)
        self.sfreq = 200
        self.n_chans = 18
        self.window_sec = 10.0
        self.window_samples = int(self.window_sec * self.sfreq)  # 2000
        self.hop_length = 100
        # self.token_size = 200

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """Load the fine‑tuned BIOT model from a .pt file."""
        if not BIOT_AVAILABLE:
            print("✗ braindecode not installed. Cannot load BIOT model.")
            return False

        try:
             #Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model = BIOT(
            n_outputs=self.num_classes,
            n_chans=self.n_chans,
            n_times=self.window_samples,
            sfreq=self.sfreq,
            hop_length=self.hop_length,
            # token_size=self.token_size,
        
            )
            
            # # Initialize the base BIOT model (pretrained on six datasets)
            # self.model = BIOT.from_pretrained(
            #     "braindecode/biot-pretrained-six-datasets-18chs",
            #     n_outputs=2,  # dummy, will be replaced
            #     n_chans=self.n_chans,
            #     n_times=self.window_samples,
            #     sfreq=self.sfreq,
            #     hop_length=self.hop_length,
            #     # token_size=self.token_size,
            # )

            # Replace classification head to match number of classes
            in_features = self.model.final_layer.classification_head.in_features
            self.model.final_layer.classification_head = nn.Linear(in_features, self.num_classes)
            
            # Replace the classification head with a new one matching our number of classes
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.model_path = model_path
            print(f"✓ BIOT model loaded from {os.path.basename(model_path)}")
            return True

        except Exception as e:
            print(f"✗ Error loading BIOT model: {str(e)}")
            self.model = None
            return False

    def preprocess(self, eeg_data: np.ndarray, fs: float) -> np.ndarray:
        """
        Preprocess raw EEG data to match model input.
        Steps:
          - Resample to 200 Hz.
          - Select/pad to 18 channels.
          - Normalize per channel (z‑score).
          - Segment into windows of length self.window_samples with 50% overlap.
        
        Args:
            eeg_data: numpy array of shape (channels, time)
            fs: original sampling frequency
        
        Returns:
            array of shape (n_windows, self.n_chans, self.window_samples)
        """
        # 1. Resample to target frequency
        if fs != self.sfreq:
            eeg_data = self._resample(eeg_data, fs, self.sfreq)

        # 2. Channel selection / padding
        eeg_data = self._select_channels(eeg_data, self.n_chans)

        # 3. Per‑channel z‑score normalization
        eeg_data = self._normalize(eeg_data)

        # 4. Segment into overlapping windows
        windows = self._segment(eeg_data, self.window_samples)
        return windows

    def _resample(self, data: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
        """Linear interpolation resampling along time axis."""
        n_channels, n_samples = data.shape
        resample_ratio = target_fs / orig_fs
        new_samples = int(n_samples * resample_ratio)
        resampled = sp_signal.resample(data, new_samples, axis=1)
        return resampled

    def _select_channels(self, data: np.ndarray, target_chans: int) -> np.ndarray:
        """Select/pad to exactly target_chans channels."""
        n_ch = data.shape[0]
        if n_ch >= target_chans:
            # Evenly spaced selection
            idx = np.linspace(0, n_ch - 1, target_chans).astype(int)
            return data[idx]
        else:
            # Zero-pad missing channels
            pad = np.zeros((target_chans - n_ch, data.shape[1]), dtype=data.dtype)
            return np.vstack([data, pad])

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Per‑channel z‑score normalization."""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1
        return (data - mean) / std

    def _segment(self, data: np.ndarray, window_len: int) -> np.ndarray:
        """Create overlapping windows (50% overlap)."""
        n_ch, n_samples = data.shape
        step = window_len // 2
        windows = []
        for start in range(0, n_samples - window_len + 1, step):
            windows.append(data[:, start:start + window_len])
        return np.array(windows)

    def predict(self, eeg_data: np.ndarray, fs: float, temperature: float = 1.0) -> Dict:
        """
        Predict abnormality from raw EEG data.
        Aggregates predictions over windows (average probabilities).
        """
        if self.model is None:
            return {'error': 'Model not loaded', 'prediction': None, 'confidence': 0}

        try:
            # Preprocess: get windows
            windows = self.preprocess(eeg_data, fs)
            if windows.shape[0] == 0:
                return {'error': 'No valid windows extracted', 'prediction': None, 'confidence': 0}

            # Convert to tensor
            windows_t = torch.from_numpy(windows).float().to(self.device)  # (n_windows, 18, 2000)

            # Inference
            with torch.no_grad():
                logits = self.model(windows_t)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=1)  # (n_windows, n_classes)

            # Aggregate: average probabilities across windows
            avg_probs = probs.mean(dim=0).cpu().numpy()
            pred_idx = np.argmax(avg_probs)
            confidence = float(avg_probs[pred_idx])

            prediction = self.ABNORMALITY_CLASSES.get(pred_idx, 'unknown')
            below_threshold = confidence < CONFIDENCE_THRESHOLD
            if below_threshold:
                prediction = 'unknown'

            # All probabilities
            all_probs = {
                self.ABNORMALITY_CLASSES.get(i, f'class_{i}'): float(p)
                for i, p in enumerate(avg_probs)
            }

            return {
                'prediction': prediction,
                'confidence': round(confidence, 4),
                'below_threshold': below_threshold,
                'all_probabilities': {k: round(v, 4) for k, v in all_probs.items()},
                'model_type': 'deep_learning',
                'model_path': self.model_path,
                'temperature': temperature,
                'n_windows': windows.shape[0],
            }

        except Exception as e:
            return {'error': str(e), 'prediction': None, 'confidence': 0}


class EEGRandomForestLoader:
    """
    Loader for classical ML model (Random Forest) trained on EEG features.
    """
    # Same class mapping as BIOT
    ABNORMALITY_CLASSES = BIOTLoader.ABNORMALITY_CLASSES

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.sfreq = 200
        self.n_chans = 18
        self.window_sec = 10.0
        self.window_samples = int(self.window_sec * self.sfreq)  # 2000

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """Load Random Forest model from joblib."""
        if not JOBLIB_AVAILABLE:
            print("✗ joblib not installed.")
            return False
        try:
            self.model = joblib.load(model_path)
            print(f"✓ Random Forest model loaded from {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"✗ Error loading Random Forest: {str(e)}")
            return False

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract features from a single window (shape: channels, time).
        Must match the training script exactly.
        Returns a feature vector of length 180.
        """
        n_ch = window.shape[0]
        features = []

        # 1. Basic statistics per channel
        for ch in range(n_ch):
            sig = window[ch]
            features.append(np.mean(sig))
            features.append(np.std(sig))
            features.append(np.max(sig))
            features.append(np.min(sig))
            features.append(np.ptp(sig))
            features.append(np.sum(np.abs(np.diff(sig))))  # line length

        # 2. Frequency band powers per channel
        for ch in range(n_ch):
            sig = window[ch]
            freqs, psd = sp_signal.welch(sig, fs=self.sfreq, nperseg=min(256, len(sig)//2))
            # Define bands
            delta = np.mean(psd[(freqs >= 0.5) & (freqs < 4)]) if np.any((freqs >= 0.5) & (freqs < 4)) else 0
            theta = np.mean(psd[(freqs >= 4) & (freqs < 8)]) if np.any((freqs >= 4) & (freqs < 8)) else 0
            alpha = np.mean(psd[(freqs >= 8) & (freqs < 13)]) if np.any((freqs >= 8) & (freqs < 13)) else 0
            beta  = np.mean(psd[(freqs >= 13) & (freqs < 30)]) if np.any((freqs >= 13) & (freqs < 30)) else 0
            features.extend([delta, theta, alpha, beta])

        return np.array(features, dtype=np.float64)

    def preprocess(self, eeg_data: np.ndarray, fs: float) -> np.ndarray:
        """
        Preprocess raw EEG into windows and extract features for each window.
        Returns array of shape (n_windows, n_features).
        """
        # 1. Resample to target frequency
        if fs != self.sfreq:
            n_channels, n_samples = eeg_data.shape
            new_samples = int(n_samples * (self.sfreq / fs))
            eeg_data = sp_signal.resample(eeg_data, new_samples, axis=1)

        # 2. Channel selection
        n_ch = eeg_data.shape[0]
        if n_ch >= self.n_chans:
            idx = np.linspace(0, n_ch - 1, self.n_chans).astype(int)
            eeg_data = eeg_data[idx]
        else:
            pad = np.zeros((self.n_chans - n_ch, eeg_data.shape[1]), dtype=eeg_data.dtype)
            eeg_data = np.vstack([eeg_data, pad])

        # 3. Per‑channel z‑score normalization (same as training)
        mean = np.mean(eeg_data, axis=1, keepdims=True)
        std = np.std(eeg_data, axis=1, keepdims=True)
        std[std == 0] = 1
        eeg_data = (eeg_data - mean) / std

        # 4. Segment into windows
        window_len = self.window_samples
        step = window_len // 2
        windows = []
        for start in range(0, eeg_data.shape[1] - window_len + 1, step):
            windows.append(eeg_data[:, start:start + window_len])
        if len(windows) == 0:
            return np.array([])
        windows = np.array(windows)  # (n_windows, 18, 2000)

        # 5. Extract features for each window
        features = np.array([self.extract_features(w) for w in windows])
        return features

    def predict(self, eeg_data: np.ndarray, fs: float) -> Dict:
        """Predict using Random Forest."""
        if self.model is None:
            return {'error': 'Model not loaded', 'prediction': None, 'confidence': 0}

        try:
            # Preprocess and extract features
            X = self.preprocess(eeg_data, fs)  # (n_windows, n_features)
            if X.shape[0] == 0:
                return {'error': 'No valid windows', 'prediction': None, 'confidence': 0}

            # Predict probabilities for each window
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)  # (n_windows, n_classes)
                avg_probs = np.mean(probs, axis=0)
                pred_idx = np.argmax(avg_probs)
                confidence = float(avg_probs[pred_idx])
                classes = self.model.classes_
            else:
                # Fallback: majority vote
                preds = self.model.predict(X)
                unique, counts = np.unique(preds, return_counts=True)
                pred_idx = unique[np.argmax(counts)]
                confidence = np.max(counts) / len(preds)
                classes = unique

            prediction = self.ABNORMALITY_CLASSES.get(pred_idx, 'unknown')
            below_threshold = confidence < CONFIDENCE_THRESHOLD
            if below_threshold:
                prediction = 'unknown'

            # Build probability dict
            prob_dict = {}
            if hasattr(self.model, 'predict_proba'):
                for i, p in enumerate(avg_probs):
                    if i < len(classes):
                        class_name = self.ABNORMALITY_CLASSES.get(int(classes[i]), f'class_{classes[i]}')
                    else:
                        class_name = f'class_{i}'
                    prob_dict[class_name] = float(p)
            else:
                prob_dict[prediction] = 1.0

            return {
                'prediction': prediction,
                'confidence': round(confidence, 4),
                'below_threshold': below_threshold,
                'all_probabilities': {k: round(v, 4) for k, v in prob_dict.items()},
                'model_type': 'classical_ml',
                'model_path': self.model_path,
                'n_windows': X.shape[0],
            }

        except Exception as e:
            return {'error': str(e), 'prediction': None, 'confidence': 0}