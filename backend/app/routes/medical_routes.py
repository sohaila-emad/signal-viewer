from flask import Blueprint, jsonify, request
import numpy as np
import json

medical_bp = Blueprint('medical', __name__)

# Simulated ECG data with abnormalities
def generate_ecg_data(abnormality_type='normal', channels=12, duration=10, fs=250):
    """Generate simulated ECG data"""
    t = np.linspace(0, duration, duration * fs)
    data = []
    
    for ch in range(channels):
        # Base ECG signal (simplified)
        if abnormality_type == 'normal':
            signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
        elif abnormality_type == 'arrhythmia':
            # Irregular rhythm
            signal = np.sin(2 * np.pi * 1 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
        elif abnormality_type == 'tachycardia':
            # Fast heart rate
            signal = np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(2 * np.pi * 4 * t)
        elif abnormality_type == 'bradycardia':
            # Slow heart rate
            signal = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 1 * t)
        else:  # fibrillation
            # Chaotic signal
            signal = np.random.randn(len(t)) * 0.5 + np.sin(2 * np.pi * 1 * t)
        
        # Add channel-specific variation
        signal += 0.1 * np.random.randn(len(t))
        data.append(signal.tolist())
    
    return {
        'data': data,
        'time': t.tolist(),
        'channels': channels,
        'fs': fs,
        'abnormality': abnormality_type
    }

@medical_bp.route('/signals', methods=['GET'])
def get_signals():
    """Get list of available medical signals"""
    signals = [
        {'id': 1, 'name': 'Normal ECG', 'type': 'normal'},
        {'id': 2, 'name': 'Arrhythmia ECG', 'type': 'arrhythmia'},
        {'id': 3, 'name': 'Tachycardia ECG', 'type': 'tachycardia'},
        {'id': 4, 'name': 'Bradycardia ECG', 'type': 'bradycardia'},
        {'id': 5, 'name': 'Atrial Fibrillation', 'type': 'fibrillation'}
    ]
    return jsonify(signals)

@medical_bp.route('/signal/<signal_id>', methods=['GET'])
def get_signal(signal_id):
    """Get specific signal data"""
    abnormality_map = {
        '1': 'normal',
        '2': 'arrhythmia',
        '3': 'tachycardia',
        '4': 'bradycardia',
        '5': 'fibrillation'
    }
    
    abnormality = abnormality_map.get(signal_id, 'normal')
    data = generate_ecg_data(abnormality)
    return jsonify(data)

@medical_bp.route('/predict/<signal_id>', methods=['GET'])
def predict_abnormality(signal_id):
    """Simulate AI model prediction"""
    abnormality_map = {
        '1': {'prediction': 'normal', 'confidence': 0.95},
        '2': {'prediction': 'arrhythmia', 'confidence': 0.87},
        '3': {'prediction': 'tachycardia', 'confidence': 0.92},
        '4': {'prediction': 'bradycardia', 'confidence': 0.89},
        '5': {'prediction': 'fibrillation', 'confidence': 0.91}
    }
    
    return jsonify(abnormality_map.get(signal_id, {
        'prediction': 'unknown',
        'confidence': 0.5
    }))
