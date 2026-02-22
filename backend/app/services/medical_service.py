"""
Medical Service for Signal Processing
Provides service layer for medical signal processing (ECG/EEG)
"""

import numpy as np
from typing import Dict, List, Optional
from ..models.ecg_model import (
    predict_ecg,
    analyze_ecg_classic
)
from ..models.model_loader import (
    initialize_models,
    get_ecgnet_model,
    get_classical_ml_model
)


class MedicalService:
    """Service for medical signal processing operations."""
    
    # Define supported abnormalities
    ABNORMALITY_TYPES = [
        'normal',
        'atrial_fibrillation',
        'ventricular_tachycardia',
        'premature_ventricular_contraction',
        'bradycardia'
    ]
    
    def __init__(self):
        # Initialize real trained models
        print("\n" + "="*60)
        print("Initializing ECG Analysis Service")
        print("="*60)
        self.model_status = initialize_models()
        
        # Check which models are available
        self.ecgnet_model = get_ecgnet_model()
        self.classical_ml_model = get_classical_ml_model()
        
        self.use_real_models = self.model_status['ecgnet'] or self.model_status['classical_ml']
        
        if self.use_real_models:
            print("\n✓ REAL trained models are ACTIVE")
            print(f"  - ECGNet (Deep Learning): {'LOADED' if self.model_status['ecgnet'] else 'Not available'}")
            print(f"  - Classical ML (Random Forest): {'LOADED' if self.model_status['classical_ml'] else 'Not available'}")
            print(f"  - Device: {self.model_status['device']}")
        else:
            print("\n⚠ No real models found, using SYNTHETIC predictions")
            print(f"  - Models directory: {self.model_status['models_dir']}")
        print("="*60 + "\n")
    
    def get_signals(self) -> List[Dict]:
        """Get list of available medical signals."""
        return [
            {'id': 1, 'name': 'Normal ECG', 'type': 'normal', 'description': 'Normal heart rhythm'},
            {'id': 2, 'name': 'Atrial Fibrillation', 'type': 'atrial_fibrillation', 'description': 'Irregular heart rhythm'},
            {'id': 3, 'name': 'Ventricular Tachycardia', 'type': 'ventricular_tachycardia', 'description': 'Fast heart rate from ventricles'},
            {'id': 4, 'name': 'Premature Ventricular Contraction', 'type': 'premature_ventricular_contraction', 'description': 'Extra heartbeats'},
            {'id': 5, 'name': 'Bradycardia', 'type': 'bradycardia', 'description': 'Slow heart rate'}
        ]
    
    def get_signal_data(self, signal_id: str) -> Dict:
        """Get specific signal data."""
        # Map signal IDs to abnormality types
        abnormality_map = {
            '1': 'normal',
            '2': 'atrial_fibrillation',
            '3': 'ventricular_tachycardia',
            '4': 'premature_ventricular_contraction',
            '5': 'bradycardia'
        }
        
        abnormality = abnormality_map.get(signal_id, 'normal')
        
        # Generate synthetic ECG data for the given abnormality
        data = self._generate_ecg_data(abnormality)
        
        return data
    
    def _generate_ecg_data(self, abnormality_type: str, channels: int = 12, 
                          duration: float = 10, fs: int = 250) -> Dict:
        """
        Generate synthetic ECG data for a given abnormality type.
        
        Args:
            abnormality_type: Type of abnormality
            channels: Number of channels
            duration: Duration in seconds
            fs: Sampling frequency
            
        Returns:
            Dictionary with ECG data
        """
        t = np.linspace(0, duration, int(duration * fs))
        data = []
        
        for ch in range(channels):
            signal = self._generate_ecg_signal(abnormality_type, t, ch)
            data.append(signal.tolist())
        
        return {
            'data': data,
            'time': t.tolist(),
            'channels': channels,
            'fs': fs,
            'abnormality': abnormality_type,
            'duration': duration
        }
    
    def _generate_ecg_signal(self, abnormality_type: str, t: np.ndarray, 
                           channel: int) -> np.ndarray:
        """Generate ECG signal for a specific abnormality type."""
        np.random.seed(channel)
        
        # Base frequency components
        if abnormality_type == 'normal':
            # Normal ECG: ~1 Hz base with harmonics
            signal = (np.sin(2 * np.pi * 1.2 * t) + 
                     0.5 * np.sin(2 * np.pi * 2.4 * t) +
                     0.3 * np.sin(2 * np.pi * 3.6 * t))
            
            # Add PQRST complex pattern
            for i, time in enumerate(t):
                phase = time % 1.0
                if 0.1 < phase < 0.2:  # P wave
                    signal[i] += 0.3 * np.sin((phase - 0.1) * 10 * np.pi)
                elif 0.35 < phase < 0.45:  # QRS complex
                    signal[i] += 1.5 * np.sin((phase - 0.35) * 20 * np.pi)
                elif 0.45 < phase < 0.55:  # R peak
                    signal[i] += 0.5
                elif 0.55 < phase < 0.65:  # S wave
                    signal[i] -= 0.8 * np.sin((phase - 0.55) * 15 * np.pi)
                elif 0.7 < phase < 0.9:  # T wave
                    signal[i] += 0.4 * np.sin((phase - 0.7) * 5 * np.pi)
        
        elif abnormality_type == 'atrial_fibrillation':
            # Irregular rhythm with no clear P waves
            signal = np.zeros(len(t))
            for i, time in enumerate(t):
                # Irregular R-R intervals
                rr_interval = 0.7 + 0.3 * np.random.random()
                phase = (time % rr_interval) / rr_interval
                
                if 0.1 < phase < 0.2:
                    signal[i] = 0.2 * np.random.random()
                elif 0.3 < phase < 0.4:
                    signal[i] = 1.2 + 0.3 * np.random.random()
                elif 0.4 < phase < 0.45:
                    signal[i] = 1.5 + 0.4 * np.random.random()
                elif 0.45 < phase < 0.5:
                    signal[i] = -0.6 + 0.2 * np.random.random()
        
        elif abnormality_type == 'ventricular_tachycardia':
            # Fast, wide QRS complexes
            signal = np.zeros(len(t))
            for i, time in enumerate(t):
                # Fast regular rhythm
                phase = (time % 0.33) / 0.33  # ~180 bpm
                
                if 0.1 < phase < 0.4:
                    signal[i] = 1.3 * np.sin(phase * np.pi)
                elif 0.4 < phase < 0.5:
                    signal[i] = -0.5
        
        elif abnormality_type == 'premature_ventricular_contraction':
            # Normal rhythm with occasional premature beats
            signal = np.zeros(len(t))
            for i, time in enumerate(t):
                # Regular rhythm with occasional early beats
                beat_time = time % 0.85
                
                # Random premature beat
                if np.random.random() < 0.05:
                    beat_time = time % 0.4
                
                phase = beat_time / 0.85
                
                if 0.1 < phase < 0.2:
                    signal[i] = 0.3
                elif 0.35 < phase < 0.5:
                    signal[i] = 1.4 + 0.3 * np.random.random()
                elif 0.5 < phase < 0.55:
                    signal[i] = -0.7
                elif 0.7 < phase < 0.9:
                    signal[i] = 0.35
        
        elif abnormality_type == 'bradycardia':
            # Slow heart rate
            signal = np.zeros(len(t))
            for i, time in enumerate(t):
                phase = (time % 1.2) / 1.2  # ~50 bpm
                
                if 0.1 < phase < 0.2:
                    signal[i] = 0.3 * np.sin((phase - 0.1) * 10 * np.pi)
                elif 0.35 < phase < 0.45:
                    signal[i] = 1.4 * np.sin((phase - 0.35) * 15 * np.pi)
                elif 0.45 < phase < 0.55:
                    signal[i] = 1.6
                elif 0.55 < phase < 0.65:
                    signal[i] = -0.7 * np.sin((phase - 0.55) * 10 * np.pi)
                elif 0.7 < phase < 0.9:
                    signal[i] = 0.35 * np.sin((phase - 0.7) * 5 * np.pi)
        
        else:
            # Default normal
            signal = np.sin(2 * np.pi * 1 * t)
        
        # Add some noise
        signal += 0.05 * np.random.randn(len(t))
        
        # Add channel-specific variation
        signal += 0.1 * np.sin(2 * np.pi * 0.5 * t + channel)
        
        return signal
    
    def predict_abnormality(self, signal_data: Dict, temperature: float = 2.0) -> Dict:
        """
        Predict abnormality from ECG signal data.
        
        Uses real trained models (Deep Learning + Classical ML) with fallback to synthetic predictions.
        
        Args:
            signal_data: Dictionary with ECG data
            temperature: Temperature scaling for ECGNet (>1 = softer predictions for better calibration)
            
        Returns:
            Prediction results with deep learning and classical ML comparisons
        """
        # Convert data to numpy array (handle as multi-channel 12 leads)
        data = signal_data.get('data', [])
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                # Multi-channel format: convert to (12, N)
                ecg_data = np.array(data)
            else:
                ecg_data = np.array(data)
        else:
            ecg_data = data
        
        fs = signal_data.get('fs', 250)
        
        results = {}
        
        # === REAL MODEL PREDICTIONS ===
        if self.use_real_models:
            results['using_real_models'] = True
            
            # ECGNet Deep Learning Model with temperature scaling for better calibration
            if self.ecgnet_model is not None:
                results['ecgnet'] = self.ecgnet_model.predict(ecg_data, temperature=temperature)
            else:
                results['ecgnet'] = {'prediction': None, 'confidence': 0, 'error': 'Model not loaded'}
            
            # Classical ML Model (Random Forest)
            if self.classical_ml_model is not None:
                results['classical_ml'] = self.classical_ml_model.predict(ecg_data)
            else:
                results['classical_ml'] = {'prediction': None, 'confidence': 0, 'error': 'Model not loaded'}
            
            # Comparison between models
            results['comparison'] = self._compare_models(results['ecgnet'], results['classical_ml'])
            
        else:
            results['using_real_models'] = False
            print("⚠ Using SYNTHETIC predictions (real models not available)")
            
            # Fallback to synthetic predictions
            synthetic_ai = predict_ecg(ecg_data, fs)
            synthetic_classic = analyze_ecg_classic(ecg_data, fs)
            
            results['synthetic_ai'] = synthetic_ai
            results['synthetic_analysis'] = synthetic_classic
        
        return results
    
    def _compare_models(self, ecgnet_result: Dict, classical_result: Dict) -> Dict:
        """
        Compare predictions from ECGNet and Classical ML models.
        
        Args:
            ecgnet_result: ECGNet prediction result
            classical_result: Classical ML prediction result
            
        Returns:
            Comparison dictionary
        """
        ecgnet_pred = ecgnet_result.get('prediction', 'unknown')
        ecgnet_conf = ecgnet_result.get('confidence', 0)
        
        classical_pred = classical_result.get('prediction', 'unknown')
        classical_conf = classical_result.get('confidence', 0)
        
        # Agreement check
        agreement = ecgnet_pred == classical_pred
        
        # Ensemble prediction: prefer agreement, otherwise choose higher confidence
        if agreement:
            ensemble_pred = ecgnet_pred
            ensemble_conf = (ecgnet_conf + classical_conf) / 2
            confidence_source = 'Both models agree'
        else:
            if ecgnet_conf > classical_conf:
                ensemble_pred = ecgnet_pred
                ensemble_conf = ecgnet_conf
                confidence_source = f'ECGNet ({ecgnet_conf:.2%}) > Classical ({classical_conf:.2%})'
            else:
                ensemble_pred = classical_pred
                ensemble_conf = classical_conf
                confidence_source = f'Classical ({classical_conf:.2%}) > ECGNet ({ecgnet_conf:.2%})'
        
        return {
            'ecgnet_prediction': ecgnet_pred,
            'ecgnet_confidence': ecgnet_conf,
            'classical_prediction': classical_pred,
            'classical_confidence': classical_conf,
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': ensemble_conf,
            'agreement': agreement,
            'confidence_source': confidence_source,
            'summary': f"Ensemble prediction: {ensemble_pred} ({ensemble_conf:.2%} confidence)"
        }
    
    def _compare_ai_classic(self, ai_prediction: Dict, classic_analysis: Dict) -> Dict:
        """Compare AI and classic ML predictions (legacy method - kept for reference)."""
        # Map AI prediction to classic analysis
        ai_type = ai_prediction.get('prediction', 'unknown')
        
        classic_heart_rate = classic_analysis.get('heart_rate_analysis', {})
        classic_rhythm = classic_analysis.get('rhythm_analysis', {})
        
        # Determine classic ML prediction based on analysis
        if classic_heart_rate.get('status') == 'bradycardia':
            classic_prediction = 'bradycardia'
        elif classic_heart_rate.get('status') == 'tachycardia':
            classic_prediction = 'ventricular_tachycardia'
        elif classic_rhythm.get('status') == 'irregular':
            classic_prediction = 'atrial_fibrillation'
        else:
            classic_prediction = 'normal'
        
        # Calculate agreement
        agreement = ai_type == classic_prediction
        
        return {
            'ai_prediction': ai_type,
            'classic_prediction': classic_prediction,
            'agreement': agreement,
            'agreement_description': 
                'AI and Classic ML agree' if agreement else 
                'AI and Classic ML disagree'
        }
    
    def get_abnormalities(self) -> List[str]:
        """Get list of detectable abnormalities."""
        return self.ABNORMALITY_TYPES
    
    def predict_from_wfdb(self, wfdb_path: str, temperature: float = 2.0) -> Dict:
        """
        Predict ECG abnormality from WFDB file (direct format without CSV conversion).
        
        This allows testing if the problem is in the CSV conversion or model itself.
        
        Args:
            wfdb_path: Path to WFDB file (without extension, e.g., './data/record')
            temperature: Temperature scaling for ECGNet confidence calibration
            
        Returns:
            Dictionary with predictions from both models
        """
        from ..models.model_loader import read_wfdb_file
        
        try:
            # Read WFDB file directly
            signal_data, fs, channel_names = read_wfdb_file(wfdb_path)
            
            results = {
                'source': 'wfdb_file',
                'wfdb_path': wfdb_path,
                'channels': len(channel_names),
                'channel_names': channel_names,
                'fs': fs,
                'samples': signal_data.shape[1],
                'data_range': [float(signal_data.min()), float(signal_data.max())]
            }
            
            # Get predictions from both models
            if self.ecgnet_model is not None:
                results['ecgnet'] = self.ecgnet_model.predict(signal_data, temperature=temperature)
            else:
                results['ecgnet'] = {'prediction': None, 'confidence': 0, 'error': 'Model not loaded'}
            
            if self.classical_ml_model is not None:
                results['classical_ml'] = self.classical_ml_model.predict(signal_data)
            else:
                results['classical_ml'] = {'prediction': None, 'confidence': 0, 'error': 'Model not loaded'}
            
            # Model comparison
            results['comparison'] = self._compare_models(results.get('ecgnet', {}), results.get('classical_ml', {}))
            
            return results
            
        except Exception as e:
            return {'error': str(e), 'type': 'wfdb_read_error'}


# Singleton instance
medical_service = MedicalService()


def get_medical_service() -> MedicalService:
    """Get the medical service instance."""
    return medical_service
