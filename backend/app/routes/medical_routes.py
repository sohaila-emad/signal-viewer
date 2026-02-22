from flask import Blueprint, jsonify, request
import numpy as np
import json
from ..services.medical_service import get_medical_service

medical_bp = Blueprint('medical', __name__)

# Get service instance
medical_service = get_medical_service()


@medical_bp.route('/signals', methods=['GET'])
def get_signals():
    """Get list of available medical signals"""
    signals = medical_service.get_signals()
    return jsonify(signals)


@medical_bp.route('/signal/<signal_id>', methods=['GET'])
def get_signal(signal_id):
    """Get specific signal data"""
    data = medical_service.get_signal_data(signal_id)
    return jsonify(data)


@medical_bp.route('/predict/<signal_id>', methods=['GET'])
def predict_abnormality(signal_id):
    """Get AI model prediction for a signal"""
    # Get the signal data
    signal_data = medical_service.get_signal_data(signal_id)
    
    # Get prediction using AI model
    prediction = medical_service.predict_abnormality(signal_data)
    
    return jsonify(prediction)


@medical_bp.route('/predict', methods=['POST'])
def predict_from_data():
    """Predict abnormality from uploaded signal data"""
    data = request.get_json()
    
    if not data or 'data' not in data:
        return jsonify({'error': 'Missing data field'}), 400
    
    try:
        temperature = data.get('temperature', 2.0)  # Default temperature = 2.0
        prediction = medical_service.predict_abnormality(data, temperature=temperature)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@medical_bp.route('/predict_wfdb', methods=['POST'])
def predict_from_wfdb():
    """
    Predict abnormality from WFDB file (direct format - no CSV conversion).
    
    Test endpoint to verify if the problem is in CSV conversion or model.
    
    Request body:
    {
        "wfdb_path": "./data/record",  # Path without extension
        "temperature": 2.0  # Optional temperature scaling (default: 2.0)
    }
    """
    data = request.get_json()
    
    if not data or 'wfdb_path' not in data:
        return jsonify({'error': 'Missing wfdb_path field'}), 400
    
    try:
        wfdb_path = data.get('wfdb_path')
        temperature = data.get('temperature', 2.0)
        
        prediction = medical_service.predict_from_wfdb(wfdb_path, temperature=temperature)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@medical_bp.route('/test_temperature', methods=['POST'])
def test_temperature_scaling():
    """
    Test how different temperature values affect ECGNet predictions.
    
    This helps find the best temperature value for confidence calibration.
    
    Request body:
    {
        "data": [...],  # ECG signal data
        "fs": 100  # Optional sampling rate
    }
    
    Response: 
    {
        "temperatures_tested": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "results": {...}  # Predictions for each temperature
    }
    """
    data = request.get_json()
    
    if not data or 'data' not in data:
        return jsonify({'error': 'Missing data field'}), 400
    
    try:
        temperatures = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        results = {
            'temperatures_tested': temperatures,
            'signal_shape': [len(data['data']), len(data['data'][0])] if isinstance(data['data'][0], list) else [1, len(data['data'])],
            'results': {}
        }
        
        for temp in temperatures:
            prediction = medical_service.predict_abnormality(data, temperature=temp)
            results['results'][str(temp)] = prediction
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@medical_bp.route('/abnormalities', methods=['GET'])
def get_abnormalities():
    """Get list of detectable abnormalities"""
    abnormalities = medical_service.get_abnormalities()
    return jsonify({'abnormalities': abnormalities})
