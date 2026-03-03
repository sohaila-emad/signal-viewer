"""
Microbiome Model — iHMP longitudinal dataset loader.

Single-file architecture
─────────────────────────
Loads data/ihmp_longitudinal.csv automatically on startup.
No user upload required.

CSV columns:
  subject_id, visit_number, date, disease_status,
  Bacteroides, Prevotella, Faecalibacterium, Bifidobacterium,
  Lactobacillus, Escherichia, Streptococcus, Clostridium,
  Ruminococcus, Akkermansia, Blautia, Roseburia
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# ── Path to bundled dataset ────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]          # …/signal-viewer
_DATA_CSV     = _PROJECT_ROOT / 'data' / 'ihmp_longitudinal.csv'

GENERA_COLS = [
    'Bacteroides', 'Prevotella', 'Faecalibacterium', 'Bifidobacterium',
    'Lactobacillus', 'Escherichia', 'Streptococcus', 'Clostridium',
    'Ruminococcus', 'Akkermansia', 'Blautia', 'Roseburia',
]


# ══════════════════════════════════════════════════════════════════════════════
#  Alpha diversity
# ══════════════════════════════════════════════════════════════════════════════

def shannon_index(abundances) -> float:
    """Shannon H' = −Σ p·ln(p)."""
    vals = [float(a) for a in abundances if float(a) > 0]
    return -sum(v * math.log(v) for v in vals) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loader
# ══════════════════════════════════════════════════════════════════════════════

class IHMPDataset:
    """
    Wraps the iHMP longitudinal CSV.
    Each row is one visit for one subject.
    """

    def __init__(self, csv_path: Path = _DATA_CSV):
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f'iHMP dataset not found: {csv_path}\n'
                'Place ihmp_longitudinal.csv in the project data/ directory.'
            )
        self._df = pd.read_csv(csv_path, dtype={'subject_id': str})
        self._df['subject_id'] = self._df['subject_id'].str.strip()

        # Ensure all genus columns are present (fill missing with 0)
        for g in GENERA_COLS:
            if g not in self._df.columns:
                self._df[g] = 0.0
            else:
                self._df[g] = pd.to_numeric(self._df[g], errors='coerce').fillna(0.0)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def participant_ids(self) -> List[str]:
        return sorted(self._df['subject_id'].dropna().unique().tolist())

    def visits_for_participant(self, subject_id: str) -> pd.DataFrame:
        mask = self._df['subject_id'] == subject_id
        return self._df[mask].sort_values('visit_number').reset_index(drop=True)

    def all_genera(self) -> List[str]:
        return [g for g in GENERA_COLS if g in self._df.columns]

    def participant_info(self, subject_id: str) -> Dict:
        rows = self.visits_for_participant(subject_id)
        if rows.empty:
            return {}
        first = rows.iloc[0]
        return {
            'participant_id': subject_id,
            'diagnosis':      first.get('disease_status', None),
            'n_metadata_rows': len(rows),
        }

    def mean_composition(self) -> List[Dict]:
        """Mean relative abundance per genus across all visits."""
        result = []
        for g in self.all_genera():
            mean_val = float(self._df[g].mean())
            result.append({'genus': g, 'mean_abundance': round(mean_val, 6)})
        return sorted(result, key=lambda x: x['mean_abundance'], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Longitudinal data
# ══════════════════════════════════════════════════════════════════════════════

def get_longitudinal_data(dataset: IHMPDataset, subject_id: str) -> List[Dict]:
    """
    Return sorted visit list for a subject.
    Each entry has: visit_num, date, shannon, genera {genus: float}.
    """
    rows = dataset.visits_for_participant(subject_id)
    if rows.empty:
        return []

    genera_cols = dataset.all_genera()
    result = []
    for _, row in rows.iterrows():
        genera_abund = {g: round(float(row[g]), 6) for g in genera_cols}
        shannon      = round(shannon_index(list(genera_abund.values())), 4)
        result.append({
            'external_id':     f"{subject_id}_v{int(row['visit_number'])}",
            'visit_num':       int(row['visit_number']),
            'week_num':        None,
            'date_of_receipt': str(row.get('date', '')),
            'data_type':       'metagenomics',
            'shannon':         shannon,
            'genera':          genera_abund,
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Trend calculations
# ══════════════════════════════════════════════════════════════════════════════

def _fb_ratio(genera: Dict[str, float]) -> Optional[float]:
    def g(n): return genera.get(n, 0.0)
    firm = g('Faecalibacterium') + g('Ruminococcus') + g('Blautia') + g('Roseburia') + g('Lactobacillus') + g('Clostridium')
    bact = g('Bacteroides') + g('Prevotella')
    return round(firm / bact, 4) if bact > 0.001 else None


def calculate_trends(longitudinal: List[Dict]) -> Dict:
    """
    Compute Shannon diversity and F/B ratio trends across visits.
    """
    shannon_series, fb_series = [], []
    for v in longitudinal:
        visit_label = v.get('visit_num') or v['external_id']
        shannon_series.append({'visit': visit_label, 'shannon': v['shannon']})
        fb = _fb_ratio(v['genera'])
        if fb is not None:
            fb_series.append({'visit': visit_label, 'fb_ratio': fb})

    def _delta(series, key):
        vals = [p[key] for p in series if p[key] is not None]
        return round(vals[-1] - vals[-2], 4) if len(vals) >= 2 else None

    return {
        'shannon_trend': shannon_series,
        'fb_trend':      fb_series,
        'shannon_delta': _delta(shannon_series, 'shannon'),
        'fb_delta':      _delta(fb_series, 'fb_ratio'),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Formula-based gut health profile
# ══════════════════════════════════════════════════════════════════════════════

def estimate_patient_profile(abundances: Dict[str, float]) -> Dict:
    """
    Derive health indicators from genus-level relative abundances.
    """
    def g(n): return abundances.get(n, 0.0)

    bacteroides      = g('Bacteroides')
    prevotella       = g('Prevotella')
    faecalibacterium = g('Faecalibacterium')
    akkermansia      = g('Akkermansia')
    bifidobacterium  = g('Bifidobacterium')
    lactobacillus    = g('Lactobacillus')
    escherichia      = g('Escherichia')
    clostridium      = g('Clostridium')
    ruminococcus     = g('Ruminococcus')
    blautia          = g('Blautia')
    roseburia        = g('Roseburia')
    streptococcus    = g('Streptococcus')

    H = shannon_index(list(abundances.values()))

    firm     = faecalibacterium + ruminococcus + blautia + roseburia + lactobacillus + clostridium
    bact     = bacteroides + prevotella
    fb_ratio = round(firm / bact, 4) if bact > 0.001 else None
    bp_ratio = round(bacteroides / prevotella, 4) if prevotella > 0.001 else None

    beneficial = faecalibacterium + akkermansia + bifidobacterium + roseburia
    gut_score  = round(min(beneficial * 200, 100))

    pro  = escherichia + clostridium + streptococcus
    anti = faecalibacterium + akkermansia + roseburia + bifidobacterium
    inflammation_index = round(pro / anti, 4) if anti > 0.001 else None
    dysbiosis_index    = round(escherichia / (faecalibacterium + akkermansia + 0.001), 4)

    if bacteroides >= prevotella and bacteroides >= ruminococcus:
        enterotype = 'ET-1 (Bacteroides-driven) — Western / animal-protein diet'
    elif prevotella >= bacteroides and prevotella >= ruminococcus:
        enterotype = 'ET-2 (Prevotella-driven) — Plant-rich / carbohydrate diet'
    else:
        enterotype = 'ET-3 (Ruminococcus/mixed) — Diverse diet'




    return {
        'shannon_diversity':  round(H, 4),
        'gut_health_score':   gut_score,
        'gut_health_label':   'Good' if gut_score >= 25 else 'Moderate' if gut_score >= 10 else 'Poor',
        'enterotype':         enterotype,
        'fb_ratio':           fb_ratio,
        'bp_ratio':           bp_ratio,
        'beneficial_pct':     round(beneficial * 100, 2),
        'inflammation_index': inflammation_index,
        'dysbiosis_index':    dysbiosis_index,
        'dominant_genus':     max(abundances, key=abundances.get) if abundances else 'Unknown',
    }
