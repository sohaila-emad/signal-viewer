from flask import Blueprint, request, jsonify
import time
import random

from ..services.microbiome_service import get_microbiome_service
from ..models.microbiome_model import (
    train_ml_models, 
    predict_disease,
    MicrobiomeMLClassifier,
    MicrobiomeDataLoader
)

microbiome_bp = Blueprint('microbiome', __name__)

# Initialize service
microbiome_service = get_microbiome_service()

# ML Classifier singleton
_ml_classifier = None

def get_ml_classifier():
    """Get or initialize ML classifier."""
    global _ml_classifier
    if _ml_classifier is None:
        _ml_classifier = MicrobiomeMLClassifier()
        _ml_classifier.prepare_data(n_samples=500)
        _ml_classifier.train_models()
    return _ml_classifier

# Cache for generated data
_cached_data = None

def get_cached_data():
    """Get or generate cached microbiome data"""
    global _cached_data
    if _cached_data is None:
        _cached_data = microbiome_service.load_data(n_samples=200)
    return _cached_data

@microbiome_bp.route('/', methods=['GET'])
def get_samples():
    """Get all microbiome samples"""
    data = get_cached_data()
    samples = data.get('data', [])
    return jsonify({
        'samples': samples,
        'count': len(samples)
    })

@microbiome_bp.route('/', methods=['POST'])
def create_sample():
    """Create a new microbiome sample"""
    data = request.get_json()
    
    if not data or 'sample_id' not in data or 'bacteria_types' not in data:
        return jsonify({'error': 'Missing required fields: sample_id, bacteria_types'}), 400
    
    new_sample = {
        'id': random.randint(1000, 9999),
        'sample_id': data['sample_id'],
        'bacteria_types': data['bacteria_types'],
        'diversity_index': calculate_diversity_index(data['bacteria_types']),
        'collection_date': data.get('collection_date', time.time()),
        'location': data.get('location', ''),
        'notes': data.get('notes', '')
    }
    
    return jsonify(new_sample), 201

@microbiome_bp.route('/<string:sample_id>', methods=['GET'])
def get_sample(sample_id):
    """Get a specific microbiome sample by sample_id"""
    data = get_cached_data()
    samples = data.get('data', [])
    
    sample = next((s for s in samples if s.get('sample_id') == sample_id), None)
    
    if sample:
        return jsonify(sample)
    return jsonify({'error': 'Sample not found'}), 404

@microbiome_bp.route('/analyze-diversity', methods=['POST'])
def analyze_diversity():
    """Analyze microbiome diversity"""
    data = request.get_json()
    
    if not data or 'samples' not in data:
        return jsonify({'error': 'Missing samples data'}), 400
    
    sample_list = data['samples']
    
    analysis_results = []
    for sample in sample_list:
        if 'bacteria_types' in sample:
            analysis_results.append({
                'sample_id': sample.get('sample_id', 'unknown'),
                'diversity_index': calculate_diversity_index(sample['bacteria_types']),
                'bacteria_count': len(sample['bacteria_types'])
            })
    
    return jsonify({
        'analysis': analysis_results,
        'total_samples': len(analysis_results)
    })

def calculate_diversity_index(bacteria_types):
    """Simple Shannon diversity index calculation"""
    from collections import Counter
    import math
    
    if not bacteria_types:
        return 0.0
    
    if isinstance(bacteria_types, dict):
        total = sum(bacteria_types.values())
        if total == 0:
            return 0.0
        shannon_index = -sum(
            (v/total) * math.log(v/total) 
            for v in bacteria_types.values() if v > 0
        )
    else:
        counts = Counter(bacteria_types)
        total = sum(counts.values())
        shannon_index = -sum(
            (count/total) * math.log(count/total) 
            for count in counts.values()
        )
    
    return round(shannon_index, 3)

@microbiome_bp.route('/<string:sample_id>', methods=['DELETE'])
def delete_sample(sample_id):
    """Delete a microbiome sample"""
    return jsonify({'message': 'Sample deleted', 'sample_id': sample_id})

# NEW ENDPOINTS FOR MICROBIOME SIGNALS

@microbiome_bp.route('/summary', methods=['GET'])
def get_summary():
    """Get dataset summary"""
    data = get_cached_data()
    stats = microbiome_service.get_statistics()
    
    return jsonify({
        'total_samples': stats.get('n_samples', 0),
        'total_subjects': stats.get('n_subjects', 0),
        'disease_distribution': stats.get('disease_distribution', {}),
        'age_range': stats.get('age_range', [0, 0]),
        'mean_age': round(stats.get('age_mean', 0), 1),
        'bmi_range': stats.get('bmi_range', [0, 0]),
        'mean_bmi': round(stats.get('bmi_mean', 0), 1),
        'bacterial_taxa': len(microbiome_service.get_bacterial_taxonomy().get('genera', []))
    })

@microbiome_bp.route('/diseases', methods=['GET'])
def get_diseases():
    """Get available disease profiles"""
    profiles = microbiome_service.get_disease_profiles()
    return jsonify(profiles)

@microbiome_bp.route('/taxa', methods=['GET'])
def get_taxa():
    """Get bacterial taxonomy"""
    taxonomy = microbiome_service.get_bacterial_taxonomy()
    return jsonify(taxonomy)

@microbiome_bp.route('/analyze', methods=['POST'])
def analyze_sample():
    """Analyze a microbiome sample"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        result = microbiome_service.analyze_sample(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/estimate-patient', methods=['POST'])
def estimate_patient():
    """Estimate patient profile based on microbiome data"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        result = microbiome_service.estimate_patient(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Get detailed statistics"""
    try:
        stats = microbiome_service.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/diversity', methods=['GET'])
def get_diversity():
    """Get diversity indices for all samples"""
    data = get_cached_data()
    diversity = data.get('diversity', [])
    
    return jsonify({
        'samples': diversity,
        'count': len(diversity),
        'average_shannon': sum(d.get('shannon_index', 0) for d in diversity) / len(diversity) if diversity else 0,
        'average_simpson': sum(d.get('simpson_index', 0) for d in diversity) / len(diversity) if diversity else 0
    })

@microbiome_bp.route('/compare', methods=['POST'])
def compare_samples():
    """Compare two microbiome samples"""
    data = request.get_json()
    
    if not data or 'sample1_id' not in data or 'sample2_id' not in data:
        return jsonify({'error': 'Missing sample IDs'}), 400
    
    try:
        result = microbiome_service.compare_samples(data['sample1_id'], data['sample2_id'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/composition', methods=['GET'])
def get_composition():
    """Get average bacterial composition"""
    data = get_cached_data()
    samples = data.get('data', [])
    
    if not samples:
        return jsonify({'error': 'No data available'}), 404
    
    genera = microbiome_service.get_bacterial_taxonomy().get('genera', [])
    
    total_counts = {g: 0 for g in genera}
    
    for sample in samples:
        for g in genera:
            total_counts[g] += sample.get(g, 0)
    
    n_samples = len(samples)
    composition = [
        {'name': g, 'value': round(total_counts[g] / n_samples, 4)}
        for g in genera
    ]
    
    composition = sorted(composition, key=lambda x: x['value'], reverse=True)
    
    return jsonify({
        'composition': composition,
        'total_samples': n_samples
    })

@microbiome_bp.route('/disease/<string:disease_name>', methods=['GET'])
def get_disease_samples(disease_name):
    """Get samples for a specific disease"""
    data = get_cached_data()
    samples = data.get('data', [])
    
    filtered_samples = [
        s for s in samples 
        if s.get('disease_status', '').lower() == disease_name.lower()
    ]
    
    return jsonify({
        'disease': disease_name,
        'samples': filtered_samples,
        'count': len(filtered_samples)
    })

@microbiome_bp.route('/load-data', methods=['POST'])
def load_data():
    """Load microbiome data with specified number of samples"""
    data = request.get_json() or {}
    n_samples = data.get('n_samples', 100)
    
    try:
        result = microbiome_service.load_data(n_samples=n_samples)
        global _cached_data
        _cached_data = result
        return jsonify({
            'message': f'Loaded {n_samples} samples',
            'total_samples': result.get('n_samples', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ML ENDPOINTS

@microbiome_bp.route('/ml/train', methods=['POST'])
def ml_train():
    """Train ML models"""
    try:
        result = train_ml_models(n_samples=500)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/ml/predict', methods=['POST'])
def ml_predict():
    """Predict disease using ML"""
    data = request.get_json()
    
    if not data or 'abundances' not in data:
        return jsonify({'error': 'Missing abundances data'}), 400
    
    try:
        result = predict_disease(data['abundances'], model=data.get('model', 'rf'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/ml/feature-importance', methods=['GET'])
def ml_feature_importance():
    """Get feature importance from ML models"""
    try:
        classifier = get_ml_classifier()
        importance = classifier._get_feature_importance()
        return jsonify({'feature_importance': importance})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/ml/pca', methods=['GET'])
def ml_pca():
    """Get PCA visualization data"""
    try:
        classifier = get_ml_classifier()
        pca_data = classifier.get_pca_visualization_data()
        return jsonify(pca_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@microbiome_bp.route('/ml/models', methods=['GET'])
def ml_models_info():
    """Get information about available ML models"""
    return jsonify({
        'models': [
            {
                'name': 'Random Forest',
                'type': 'rf',
                'description': 'Ensemble learning method for classification'
            },
            {
                'name': 'Logistic Regression',
                'type': 'lr',
                'description': 'Linear classification model with probability output'
            },
            {
                'name': 'Support Vector Machine',
                'type': 'svm',
                'description': 'SVM with RBF kernel for complex boundaries'
            }
        ],
        'features': MicrobiomeDataLoader.COMMON_GENERA
    })
