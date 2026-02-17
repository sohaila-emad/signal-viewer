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
        prediction = medical_service.predict_abnormality(data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@medical_bp.route('/abnormalities', methods=['GET'])
def get_abnormalities():
    """Get list of detectable abnormalities"""
    abnormalities = medical_service.get_abnormalities()
    return jsonify({'abnormalities': abnormalities})
