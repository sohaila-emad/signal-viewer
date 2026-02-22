"""
Model Loader for ECG Deep Learning and Classical ML Models
Handles loading and inference for trained PyTorch and scikit-learn models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Try to import wfdb for reading WFDB files
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("⚠ wfdb not installed. WFDB file support disabled. Install with: pip install wfdb")

# Try to import joblib for classical ML model
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class ECGNet(nn.Module):
    """ECGNet architecture for ECG classification"""
    def __init__(self, num_classes=4):
        super(ECGNet, self).__init__()
        # 1D Convolutional layers to process the signal
        # Input: (Batch, 12 leads, 1000 samples)
        self.conv1 = nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # Flattens the signal to a single value per feature
        
        # Classification layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for Dense layer
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ECGNetLoader:
    """Loader for ECGNet deep learning model"""
    
    # PTB-XL 4-class mapping (Models.md §5): label_int → label
    ABNORMALITY_CLASSES_4 = {
        0: 'MI',    # Myocardial infarction
        1: 'STTC',  # ST/T change (ST–T changes)
        2: 'CD',    # Conduction disturbance
        3: 'HYP',   # Hypertrophy
    }
    
    # Fallback for 5-class checkpoints (if any)
    ABNORMALITY_CLASSES_5 = {
        0: 'MI',
        1: 'STTC',
        2: 'CD',
        3: 'HYP',
        4: 'unknown',
    }
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize ECGNet loader.
        
        Args:
            model_path: Path to the PyTorch model file
            device: 'cuda' or 'cpu' (defaults to cuda if available)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.num_classes = 4  # Trained model (final_ecgnet_model.pth) has 4 classes
        self.abnormality_classes = self.ABNORMALITY_CLASSES_4
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load PyTorch model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to determine number of classes from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Inspect checkpoint to determine output classes
            num_classes = 4  # Default (PTB-XL 4-class: MI, STTC, CD, HYP)
            if isinstance(checkpoint, dict):
                # Try to infer from state dict
                if 'fc2.weight' in checkpoint or 'fc2.weight' in checkpoint.get('model_state_dict', {}):
                    state_dict = checkpoint if 'fc2.weight' in checkpoint else checkpoint.get('model_state_dict', {})
                    if 'fc2.weight' in state_dict:
                        num_classes = state_dict['fc2.weight'].shape[0]
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    if 'fc2.weight' in state_dict:
                        num_classes = state_dict['fc2.weight'].shape[0]
            
            # Set class mapping based on number of classes
            if num_classes == 4:
                self.abnormality_classes = self.ABNORMALITY_CLASSES_4
            else:
                self.abnormality_classes = self.ABNORMALITY_CLASSES_5
            
            self.num_classes = num_classes
            
            # Initialize model with correct number of classes
            self.model = ECGNet(num_classes=num_classes)
            
            # Load weights
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.model_path = model_path
            
            print(f"✓ ECGNet loaded from {os.path.basename(model_path)} (num_classes={num_classes})")
            return True
            
        except Exception as e:
            print(f"✗ Error loading ECGNet from {model_path}: {str(e)}")
            self.model = None
            return False
    
    def preprocess(self, ecg_data: np.ndarray, target_shape: Tuple[int, int] = (12, 1000)) -> torch.Tensor:
        """
        Preprocess ECG data to model input shape (12, 1000).
        
        Args:
            ecg_data: ECG signal array
            target_shape: Target shape (channels, samples)
            
        Returns:
            Tensor of shape (1, 12, 1000) ready for inference
        """
        # Convert to numpy if needed
        if isinstance(ecg_data, list):
            ecg_data = np.array(ecg_data)
        
        # Handle different input shapes
        if ecg_data.ndim == 1:
            # Single channel - replicate to 12
            ecg_data = np.tile(ecg_data, (12, 1))
        elif ecg_data.ndim == 2:
            if ecg_data.shape[0] != 12:
                # Wrong lead count - try transposing
                if ecg_data.shape[1] == 12:
                    ecg_data = ecg_data.T
        
        n_samples = ecg_data.shape[1]
        target_samples = target_shape[1]
        
        # Resample if needed (model expects exactly 1000 samples per lead)
        if n_samples != target_samples:
            ecg_data = self._resample_signal(ecg_data, target_samples)
        
        # No normalization: training used raw signal (see Models.md).
        # Applying min-max or z-score at inference would mismatch training and hurt accuracy.
        
        # Add batch dimension: (12, 1000) -> (1, 12, 1000)
        tensor = torch.from_numpy(ecg_data).float().unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _resample_signal(self, signal: np.ndarray, target_samples: int) -> np.ndarray:
        """Resample signal to target number of samples"""
        n_channels, n_samples = signal.shape
        
        if n_samples == target_samples:
            return signal
        
        # Linear interpolation
        indices = np.linspace(0, n_samples - 1, target_samples)
        resampled = np.zeros((n_channels, target_samples))
        
        for ch in range(n_channels):
            resampled[ch] = np.interp(indices, np.arange(n_samples), signal[ch])
        
        return resampled
    
    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal per channel (min-max to [-1, 1]).
        NOT used at inference: training used raw signal (Models.md); using this would hurt accuracy.
        Kept only for optional experimentation.
        """
        normalized = signal.copy()
        
        for ch in range(signal.shape[0]):
            ch_min = np.min(signal[ch])
            ch_max = np.max(signal[ch])
            ch_range = ch_max - ch_min
            
            if ch_range > 0:
                # Min-max normalization to [-1, 1] range
                normalized[ch] = 2 * (signal[ch] - ch_min) / ch_range - 1
            else:
                normalized[ch] = signal[ch] - ch_min
        
        return normalized
    
    def predict(self, ecg_data: np.ndarray, temperature: float = 1.0) -> Dict:
        """
        Predict abnormality from ECG data with optional temperature scaling for calibration.
        
        Args:
            ecg_data: ECG signal array
            temperature: Temperature for softmax scaling (>1 = softer predictions, <1 = harder)
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': 0
            }
        
        try:
            # Preprocess
            tensor = self.preprocess(ecg_data)
            
            # Inference
            with torch.no_grad():
                logits = self.model(tensor)
                # Apply temperature scaling for better confidence calibration
                if temperature != 1.0:
                    logits = logits / temperature
                probabilities = F.softmax(logits, dim=1)
            
            # Get prediction and confidence
            probs_np = probabilities.cpu().numpy()[0]
            pred_idx = np.argmax(probs_np)
            confidence = float(probs_np[pred_idx])
            
            prediction = self.abnormality_classes.get(pred_idx, 'unknown')
            
            # All probabilities
            all_probs = {
                self.abnormality_classes.get(i, f'class_{i}'): float(p)
                for i, p in enumerate(probs_np)
            }
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 4),
                'all_probabilities': {k: round(v, 4) for k, v in all_probs.items()},
                'model_type': 'deep_learning',
                'model_path': self.model_path,
                'temperature': temperature
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction': None,
                'confidence': 0
            }


class ClassicalMLLoader:
    """Loader for Classical ML model (Random Forest). Trained with 4 classes (Models.md)."""
    
    # PTB-XL 4-class mapping (Models.md §5): label_int → label
    ABNORMALITY_CLASSES = {
        0: 'MI',    # Myocardial infarction
        1: 'STTC',  # ST/T change (ST–T changes)
        2: 'CD',    # Conduction disturbance
        3: 'HYP',   # Hypertrophy
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize Classical ML loader.
        
        Args:
            model_path: Path to the joblib model file
        """
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load scikit-learn model from joblib.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        if not JOBLIB_AVAILABLE:
            print("✗ joblib not installed. Cannot load classical ML model.")
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            print(f"✓ Classical ML model loaded from {os.path.basename(model_path)}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading classical ML model: {str(e)}")
            self.model = None
            return False
    
    # Target length for feature extraction (must match training: train_classical.py uses 1000-sample signals)
    TARGET_SAMPLES = 1000

    def extract_features(self, ecg_data: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from ECG data.
        
        Features: Mean, Std, Max, Min for each of 12 leads = 48 features.
        Order: 12 means, 12 stds, 12 maxs, 12 mins (same as train_classical.py).
        Uses exactly 1000 samples (resampled/cropped) to match training.
        
        Args:
            ecg_data: ECG signal array (12 leads, N samples)
            
        Returns:
            Feature vector of shape (48,)
        """
        # Convert to numpy if needed
        if isinstance(ecg_data, list):
            ecg_data = np.array(ecg_data)
        
        # Handle different input shapes
        if ecg_data.ndim == 1:
            # Single channel - replicate to 12
            ecg_data = np.tile(ecg_data, (12, 1))
        elif ecg_data.ndim == 2:
            if ecg_data.shape[0] != 12:
                if ecg_data.shape[1] == 12:
                    ecg_data = ecg_data.T
        
        n_leads, n_samples = ecg_data.shape
        # Use exactly TARGET_SAMPLES (1000) so statistics match training distribution
        if n_samples != self.TARGET_SAMPLES:
            ecg_data = self._resample_to_target(ecg_data, self.TARGET_SAMPLES)
        
        # Order: 12 means, then 12 stds, then 12 maxs, then 12 mins (match train_classical.py)
        f_mean = np.mean(ecg_data, axis=1)
        f_std = np.std(ecg_data, axis=1)
        f_max = np.max(ecg_data, axis=1)
        f_min = np.min(ecg_data, axis=1)
        features = np.concatenate([f_mean, f_std, f_max, f_min])
        return np.asarray(features, dtype=np.float64)
    
    def _resample_to_target(self, signal: np.ndarray, target_samples: int) -> np.ndarray:
        """Resample (12, N) to (12, target_samples) via linear interpolation."""
        n_channels, n_samples = signal.shape
        if n_samples == target_samples:
            return signal
        indices = np.linspace(0, n_samples - 1, target_samples)
        resampled = np.zeros((n_channels, target_samples))
        for ch in range(n_channels):
            resampled[ch] = np.interp(indices, np.arange(n_samples), signal[ch])
        return resampled
    
    def predict(self, ecg_data: np.ndarray) -> Dict:
        """
        Predict abnormality from ECG data.
        
        Args:
            ecg_data: ECG signal array
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': 0
            }
        
        try:
            # Extract features
            features = self.extract_features(ecg_data)
            features_reshaped = features.reshape(1, -1)
            
            # Get prediction
            pred_idx = self.model.predict(features_reshaped)[0]
            prediction = self.ABNORMALITY_CLASSES.get(pred_idx, 'unknown')
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features_reshaped)[0]
                confidence = float(np.max(probs))
                
                all_probs = {
                    self.ABNORMALITY_CLASSES.get(i, f'class_{i}'): float(p)
                    for i, p in enumerate(probs)
                }
            else:
                confidence = 1.0
                all_probs = {
                    prediction: 1.0,
                    **{k: 0.0 for k, v in self.ABNORMALITY_CLASSES.items() if v != prediction}
                }
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 4),
                'all_probabilities': {k: round(v, 4) for k, v in all_probs.items()},
                'model_type': 'classical_ml',
                'features_used': 48,  # Mean, Std, Max, Min × 12 leads
                'model_path': self.model_path
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction': None,
                'confidence': 0
            }


# Global model instances
ecgnet_model: Optional[ECGNetLoader] = None
classical_ml_model: Optional[ClassicalMLLoader] = None


def read_wfdb_file(wfdb_path: str) -> Tuple[np.ndarray, int, list]:
    """
    Read ECG data from WFDB format (.hea/.dat files).
    
    Args:
        wfdb_path: Path to WFDB file (without extension, e.g., './data/record')
        
    Returns:
        Tuple of (signal_data, sampling_rate, channel_names)
        
    Raises:
        ValueError: If wfdb is not available or file cannot be read
    """
    if not WFDB_AVAILABLE:
        raise ValueError("wfdb library not installed. Install with: pip install wfdb")
    
    try:
        # Read the WFDB record
        record = wfdb.rdrecord(wfdb_path)
        
        # Extract signal data (12 channels, N samples)
        signal_data = record.p_signal.T  # Transpose to (channels, samples)
        
        # Get sampling rate
        fs = record.fs
        
        # Get channel names
        channel_names = record.sig_name
        
        return signal_data, fs, channel_names
        
    except Exception as e:
        raise ValueError(f"Error reading WFDB file: {str(e)}")


def predict_from_wfdb(wfdb_path: str, temperature: float = 1.0) -> Dict:
    """
    Predict ECG abnormality from WFDB file using both models.
    
    Args:
        wfdb_path: Path to WFDB file (without extension)
        temperature: Temperature scaling for ECGNet confidence calibration
        
    Returns:
        Dictionary with predictions from both models
    """
    try:
        # Read WFDB file
        signal_data, fs, channel_names = read_wfdb_file(wfdb_path)
        
        results = {
            'source': 'wfdb_file',
            'wfdb_path': wfdb_path,
            'channels': len(channel_names),
            'channel_names': channel_names,
            'fs': fs,
            'samples': signal_data.shape[1]
        }
        
        # Get predictions from both models
        if ecgnet_model is not None:
            results['ecgnet'] = ecgnet_model.predict(signal_data, temperature=temperature)
        
        if classical_ml_model is not None:
            results['classical_ml'] = classical_ml_model.predict(signal_data)
        
        return results
        
    except Exception as e:
        return {'error': str(e)}


def initialize_models(models_dir: str = None) -> Dict[str, bool]:
    """
    Initialize all models.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary with initialization status
    """
    global ecgnet_model, classical_ml_model
    
    if models_dir is None:
        # Use default models directory
        models_dir = Path(__file__).parent.parent.parent.parent / 'models'
    
    models_dir = Path(models_dir)
    
    status = {
        'ecgnet': False,
        'classical_ml': False,
        'models_dir': str(models_dir),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Load ECGNet deep learning model
    ecgnet_paths = [
        models_dir / 'final_ecgnet_model.pth',
        models_dir / 'best_ecgnet_model.pth'
    ]
    
    for model_path in ecgnet_paths:
        if model_path.exists():
            ecgnet_model = ECGNetLoader()
            status['ecgnet'] = ecgnet_model.load_model(str(model_path))
            break
    
    if not status['ecgnet']:
        print(f"⚠ ECGNet model not found in {models_dir}")
    
    # Load Classical ML model (Random Forest from joblib)
    classical_path = models_dir / 'classical_rf_model.joblib'
    if classical_path.exists():
        classical_ml_model = ClassicalMLLoader()
        status['classical_ml'] = classical_ml_model.load_model(str(classical_path))
    else:
        print(f"⚠ Classical ML model not found: {classical_path}")
    
    return status


def get_ecgnet_model() -> Optional[ECGNetLoader]:
    """Get ECGNet model instance"""
    return ecgnet_model


def get_classical_ml_model() -> Optional[ClassicalMLLoader]:
    """Get classical ML model instance"""
    return classical_ml_model
