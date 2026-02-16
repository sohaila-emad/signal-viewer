from flask import Blueprint, request, jsonify
import time

microbiome_bp = Blueprint('microbiome', __name__)

# Sample microbiome samples
samples = []

@microbiome_bp.route('/', methods=['GET'])
def get_samples():
    """Get all microbiome samples"""
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
        'id': len(samples),
        'sample_id': data['sample_id'],
        'bacteria_types': data['bacteria_types'],
        'diversity_index': calculate_diversity_index(data['bacteria_types']),
        'collection_date': data.get('collection_date', time.time()),
        'location': data.get('location', ''),
        'notes': data.get('notes', '')
    }
    
    samples.append(new_sample)
    return jsonify(new_sample), 201

@microbiome_bp.route('/<string:sample_id>', methods=['GET'])
def get_sample(sample_id):
    """Get a specific microbiome sample by sample_id"""
    sample = next((s for s in samples if s['sample_id'] == sample_id), None)
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
    """
    Simple Shannon diversity index calculation
    """
    from collections import Counter
    import math
    
    if not bacteria_types:
        return 0.0
    
    counts = Counter(bacteria_types)
    total = sum(counts.values())
    
    # Shannon index: -sum(p_i * ln(p_i))
    shannon_index = -sum(
        (count/total) * math.log(count/total) 
        for count in counts.values()
    )
    
    return round(shannon_index, 3)

@microbiome_bp.route('/<string:sample_id>', methods=['DELETE'])
def delete_sample(sample_id):
    """Delete a microbiome sample"""
    global samples
    sample = next((s for s in samples if s['sample_id'] == sample_id), None)
    if sample:
        samples = [s for s in samples if s['sample_id'] != sample_id]
        return jsonify({'message': 'Sample deleted', 'sample': sample})
    return jsonify({'error': 'Sample not found'}), 404
