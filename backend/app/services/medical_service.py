"""
Medical Service for Signal Processing
Provides service layer for medical signal processing (ECG/EEG)
"""

import numpy as np
from typing import Dict, List, Optional
from ..models.ecg_model import (
    ECGClassifier,
    ClassicMLAnalyzer,
    predict_ecg,
    analyze_ecg_classic,
    get_available_abnormalities,
    ECGFeatureExtractor
)


class MedicalService:
    """Service for medical signal processing operations."""
    
    def __init__(self):
        self.ecg_classifier = ECGClassifier()
        self.classic_analyzer = ClassicMLAnalyzer()
        self.feature_extractor = ECGFeatureExtractor()
    
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
    
    def predict_abnormality(self, signal_data: Dict) -> Dict:
        """
        Predict abnormality from ECG signal data.
        
        Args:
            signal_data: Dictionary with ECG data
            
        Returns:
            Prediction results
        """
        # Convert data to numpy array
        data = signal_data.get('data', [])
        if isinstance(data, list):
            # If multi-channel, use first channel for prediction
            if isinstance(data[0], list):
                ecg_data = np.array(data[0])
            else:
                ecg_data = np.array(data)
        else:
            ecg_data = data
        
        fs = signal_data.get('fs', 250)
        
        # Get AI prediction
        ai_prediction = predict_ecg(ecg_data, fs)
        
        # Get classic ML analysis
        classic_analysis = analyze_ecg_classic(ecg_data, fs)
        
        return {
            'ai_prediction': ai_prediction,
            'classic_analysis': classic_analysis,
            'comparison': self._compare_ai_classic(ai_prediction, classic_analysis)
        }
    
    def _compare_ai_classic(self, ai_prediction: Dict, classic_analysis: Dict) -> Dict:
        """Compare AI and classic ML predictions."""
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
        return get_available_abnormalities()


# Singleton instance
medical_service = MedicalService()


def get_medical_service() -> MedicalService:
    """Get the medical service instance."""
    return medical_service
