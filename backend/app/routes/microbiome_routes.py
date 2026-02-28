from flask import Blueprint, jsonify, request
from ..services.microbiome_service import get_microbiome_service

microbiome_bp = Blueprint('microbiome', __name__)


@microbiome_bp.route('/summary', methods=['GET'])
def summary():
    """
    GET /api/microbiome/summary
    Returns metadata + abundance load status, genera list, matched participants.
    """
    try:
        return jsonify(get_microbiome_service().get_summary())
    except Exception as e:
        return jsonify({'error': str(e), 'metadata_loaded': False}), 500


@microbiome_bp.route('/participants', methods=['GET'])
def participants():
    """
    GET /api/microbiome/participants
    List of participant IDs that have data in the loaded abundance file.
    """
    try:
        ids = get_microbiome_service().get_participants()
        return jsonify({'participants': ids})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@microbiome_bp.route('/composition', methods=['GET'])
def composition():
    """
    GET /api/microbiome/composition
    Mean genus abundances across all samples in the loaded file.
    """
    return jsonify(get_microbiome_service().get_composition())


@microbiome_bp.route('/participant/<participant_id>/timeline', methods=['POST'])
def participant_timeline(participant_id):
    """
    POST /api/microbiome/participant/<id>/timeline
    Body JSON: { "genera": ["Bacteroides", "Prevotella", …] }

    Returns longitudinal time-series for the participant with:
      - One point per visit (sorted by visit_num)
      - Shannon diversity per visit
      - Requested genus abundances (%) per visit
      - Trend statistics (Shannon delta, F/B delta)
    """
    body   = request.get_json(silent=True) or {}
    genera = body.get('genera', [])
    if not isinstance(genera, list):
        return jsonify({'error': '"genera" must be a list of strings'}), 400

    result = get_microbiome_service().get_longitudinal_data(participant_id, genera)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


@microbiome_bp.route('/participant/<participant_id>/profile', methods=['GET'])
def participant_profile(participant_id):
    """
    GET /api/microbiome/participant/<id>/profile

    Returns:
      - Real clinical metadata (diagnosis, sex, age, BMI) from hmp2_metadata CSV
      - Formula-based gut health metrics (Shannon, F/B ratio, enterotype, etc.)
        computed from the participant's latest visit in the abundance file
      - Top-8 genera from the latest visit
    """
    result = get_microbiome_service().get_patient_profile(participant_id)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)
