from flask import Blueprint, request, jsonify
import time
from ..services.acoustic_service import get_acoustic_service

acoustic_bp = Blueprint('acoustic', __name__)

# Sample acoustic measurements
acoustic_measurements = []

# Get service instance
acoustic_service = get_acoustic_service()


@acoustic_bp.route('/', methods=['GET'])
def get_measurements():
    """Get all acoustic measurements"""
    return jsonify({
        'measurements': acoustic_measurements,
        'count': len(acoustic_measurements)
    })


@acoustic_bp.route('/', methods=['POST'])
def create_measurement():
    """Create a new acoustic measurement"""
    data = request.get_json()
    
    if not data or 'frequency' not in data or 'amplitude' not in data:
        return jsonify({'error': 'Missing required fields: frequency, amplitude'}), 400
    
    new_measurement = {
        'id': len(acoustic_measurements),
        'frequency': data['frequency'],
        'amplitude': data['amplitude'],
        'duration': data.get('duration', 0),
        'timestamp': time.time(),
        'notes': data.get('notes', '')
    }
    
    acoustic_measurements.append(new_measurement)
    return jsonify(new_measurement), 201


@acoustic_bp.route('/analyze', methods=['POST'])
def analyze_sound():
    """Analyze sound data"""
    data = request.get_json()
    
    if not data or 'frequencies' not in data:
        return jsonify({'error': 'Missing frequencies data'}), 400
    
    frequencies = data['frequencies']
    
    # Simple analysis example
    analysis = {
        'mean_frequency': sum(frequencies) / len(frequencies) if frequencies else 0,
        'max_frequency': max(frequencies) if frequencies else 0,
        'min_frequency': min(frequencies) if frequencies else 0,
        'frequency_range': max(frequencies) - min(frequencies) if frequencies else 0
    }
    
    return jsonify(analysis)


@acoustic_bp.route('/<int:measurement_id>', methods=['DELETE'])
def delete_measurement(measurement_id):
    """Delete an acoustic measurement"""
    if 0 <= measurement_id < len(acoustic_measurements):
        deleted = acoustic_measurements.pop(measurement_id)
        return jsonify({'message': 'Measurement deleted', 'measurement': deleted})
    return jsonify({'error': 'Measurement not found'}), 404


# Doppler effect and vehicle analysis endpoints

@acoustic_bp.route('/doppler/generate', methods=['POST'])
def generate_doppler_sound():

    """
    Generate synthetic vehicle passing sound with Doppler effect.

    Request body:
    {
        "velocity": float,     # Vehicle velocity in m/s (default: 30.0)
        "frequency": float,    # Horn frequency in Hz (default: 440.0)
        "duration": float,     # Duration in seconds (default: 5.0)
        "sample_rate": int     # Sample rate in Hz (default: 44100)
    }
    """

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request body'}), 400

    velocity = data.get('velocity', 30.0)
    frequency = data.get('frequency', 440.0)
    duration = data.get('duration', 5.0)
    sample_rate = data.get('sample_rate', 44100)

    try:
        # Use the new function that returns base64
        result = acoustic_service.generate_doppler_sound_base64(
            velocity=velocity,
            frequency=frequency,
            duration=duration,
            sample_rate=sample_rate
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to generate sound: {str(e)}'}), 500


@acoustic_bp.route('/doppler/parameters', methods=['GET'])
def get_doppler_parameters():
    """
    Get Doppler effect parameters for given velocity and frequency.
    
    Query parameters:
    - velocity: Vehicle velocity in m/s
    - frequency: Source frequency in Hz
    """
    velocity = request.args.get('velocity', type=float, default=30.0)
    frequency = request.args.get('frequency', type=float, default=440.0)
    
    if velocity <= 0 or velocity >= 343:
        return jsonify({'error': 'Velocity must be between 0 and 343 m/s'}), 400
    
    if frequency <= 0:
        return jsonify({'error': 'Frequency must be positive'}), 400
    
    try:
        result = acoustic_service.get_doppler_parameters(velocity, frequency)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@acoustic_bp.route('/vehicle/analyze', methods=['POST'])
def analyze_vehicle_passing():
    """
    Analyze vehicle passing sound to estimate velocity and frequency.
    
    Request body:
    {
        "audio_data": list,  # List of audio samples (normalized to -1 to 1)
        "sample_rate": int   # Sample rate in Hz (default: 44100)
    }
    """
    data = request.get_json()
    
    if not data or 'audio_data' not in data:
        return jsonify({'error': 'Missing audio_data'}), 400
    
    audio_data = data['audio_data']
    sample_rate = data.get('sample_rate', 44100)
    
    try:
        result = acoustic_service.analyze_vehicle_passing(audio_data, sample_rate)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to analyze sound: {str(e)}'}), 500


@acoustic_bp.route('/vehicle/detect', methods=['POST'])
def detect_drone_submarine():
    """
    Detect drone or submarine sounds in audio.
    
    Request body:
    {
        "audio_data": list,      # List of audio samples
        "sample_rate": int,       # Sample rate in Hz (default: 44100)
        "vehicle_type": string   # 'auto', 'drone', or 'submarine' (default: 'auto')
    }
    """
    data = request.get_json()
    
    if not data or 'audio_data' not in data:
        return jsonify({'error': 'Missing audio_data'}), 400
    
    audio_data = data['audio_data']
    sample_rate = data.get('sample_rate', 44100)
    vehicle_type = data.get('vehicle_type', 'auto')
    
    try:
        result = acoustic_service.detect_unmanned_vehicle(
            audio_data, 
            sample_rate,
            vehicle_type
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to detect vehicle: {str(e)}'}), 500


@acoustic_bp.route('/spectrogram', methods=['POST'])
def compute_spectrogram():
    """
    Compute spectrogram of audio data.
    
    Request body:
    {
        "audio_data": list,  # List of audio samples
        "sample_rate": int,   # Sample rate in Hz (default: 44100)
        "nperseg": int       # FFT window size (default: 256)
    }
    """
    data = request.get_json()
    
    if not data or 'audio_data' not in data:
        return jsonify({'error': 'Missing audio_data'}), 400
    
    audio_data = data['audio_data']
    sample_rate = data.get('sample_rate', 44100)
    nperseg = data.get('nperseg', 256)
    
    try:
        result = acoustic_service.compute_spectrogram(
            audio_data, 
            sample_rate,
            nperseg
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Failed to compute spectrogram: {str(e)}'}), 500
