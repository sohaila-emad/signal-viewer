"""
Microbiome Service — auto-loads the bundled iHMP longitudinal dataset.

No file upload required. Data is read from data/ihmp_longitudinal.csv at startup.
"""

from typing import Dict, List, Optional
from ..models.microbiome_model import (
    IHMPDataset,
    get_longitudinal_data,
    calculate_trends,
    estimate_patient_profile,
)


class MicrobiomeService:

    def __init__(self):
        self._dataset: IHMPDataset = IHMPDataset()

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        participants = self._dataset.participant_ids()
        genera       = self._dataset.all_genera()
        n_visits     = sum(
            len(self._dataset.visits_for_participant(p)) for p in participants
        )
        return {
            'metadata_loaded':      True,
            'abundance_loaded':     True,
            'filename':             'ihmp_longitudinal.csv',
            'n_participants':       len(participants),
            'n_samples':            n_visits,
            'n_genera':             len(genera),
            'genera':               genera,
            'n_matched':            len(participants),
            'n_unmatched':          0,
            'matched_participants': participants,
        }

    # ── Participant list ──────────────────────────────────────────────────────

    def get_participants(self) -> List[str]:
        return self._dataset.participant_ids()

    # ── Longitudinal timeline ─────────────────────────────────────────────────

    def get_longitudinal_data(self, participant_id: str, genera: List[str]) -> Dict:
        try:
            visits = get_longitudinal_data(self._dataset, participant_id)
        except Exception as e:
            print(f"Error fetching longitudinal data for {participant_id}: {e}")
            return {'error': f'Error fetching data for participant {participant_id}: {str(e)}'}
        if not visits:
            return {'error': f'No data for participant {participant_id!r}.'}

        timeline = []
        for v in visits:
            point: Dict = {
                'external_id': v['external_id'],
                'visit_num':   v['visit_num'],
                'week_num':    v['week_num'],
                'shannon':     v['shannon'],
                'label':       f"V{v['visit_num']}",
            }
            for g in genera:
                point[g] = round(v['genera'].get(g, 0.0) * 100, 4)
            timeline.append(point)

        trends = calculate_trends(visits)
        return {
            'participant_id': participant_id,
            'timeline':       timeline,
            'trends':         trends,
            'n_visits':       len(visits),
        }

    # ── Patient profile ───────────────────────────────────────────────────────

    def get_patient_profile(self, participant_id: str) -> Dict:
        try:
            visits = get_longitudinal_data(self._dataset, participant_id)
        except Exception as e:
            print(f"Error fetching patient profile for {participant_id}: {e}")
            return {'error': f'Error fetching data for participant {participant_id}: {str(e)}'}
        if not visits:
            return {'error': f'No data for participant {participant_id!r}.'}

        info       = self._dataset.participant_info(participant_id)
        latest     = visits[-1]
        formula    = estimate_patient_profile(latest['genera'])
        top_genera = sorted(latest['genera'].items(), key=lambda x: x[1], reverse=True)[:8]

        return {
            **info,
            'sex':                None,
            'consent_age':        None,
            'bmi':                None,
            'site_name':          None,
            'latest_external_id': latest['external_id'],
            'latest_visit_num':   latest['visit_num'],
            'latest_week_num':    latest['week_num'],
            'n_visits_in_file':   len(visits),
            'top_genera': [
                {'genus': g, 'abundance_pct': round(a * 100, 3)}
                for g, a in top_genera
            ],
            **formula,
        }

    # ── Mean composition ──────────────────────────────────────────────────────

    def get_composition(self) -> List[Dict]:
        return self._dataset.mean_composition()


# ── Singleton ─────────────────────────────────────────────────────────────────

_service: Optional[MicrobiomeService] = None


def get_microbiome_service() -> MicrobiomeService:
    global _service
    if _service is None:
        _service = MicrobiomeService()
    return _service
