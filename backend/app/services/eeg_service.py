"""
EEG Service for handling EEG signal processing with BIOT and Classical ML.
"""
import numpy as np
from pathlib import Path
from typing import Dict

from ..models.eeg_loader import BIOTLoader, EEGRandomForestLoader

class EEGService:
    """Service for EEG abnormality detection using trained models."""

    # Mapping of class indices to abnormality names (same as in loaders)
    ABNORMALITY_TYPES = ['normal', 'seizure', 'alcoholism', 'motor_abnormality', 'mental_stress', 'epileptic_interictal']

    def __init__(self, models_dir: str = None):
        if models_dir is None:
        # Go up four levels: from app/services/ -> app/ -> backend/ -> project root
            models_dir = Path(__file__).parent.parent.parent.parent / 'models'
        self.models_dir = Path(models_dir)

        # Load models
        self.biot_loader = None
        self.rf_loader = None
        self._initialize_models()

    def _initialize_models(self):
        """Load BIOT and Random Forest models from the models directory."""
        print("\n" + "="*60)
        print("Initializing EEG Analysis Service")
        print("="*60)

        # BIOT model
        biot_path = self.models_dir / 'eeg_biot_best.pt'
        if biot_path.exists():
            self.biot_loader = BIOTLoader(str(biot_path))
            print("  [OK] BIOT model loaded")
        else:
            print(f"  [ERROR] BIOT model not found at {biot_path}")

        # Random Forest model
        rf_path = self.models_dir / 'eeg_rf_model.joblib'
        if rf_path.exists():
            self.rf_loader = EEGRandomForestLoader(str(rf_path))
            print("  [OK] Random Forest model loaded")
        else:
            print(f"  [ERROR] Random Forest model not found at {rf_path}")

        print("="*60 + "\n")

    def predict(self, signal_data: Dict, temperature: float = 1.0) -> Dict:
        """
        Predict abnormality from EEG signal data.
        Expects signal_data with 'data' (list of lists) and 'fs' (sampling frequency).
        """
        # Convert to numpy array
        data = np.array(signal_data.get('data', []), dtype=np.float32)
        fs = signal_data.get('fs', 250)  # fallback, but should be provided

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 2 and data.shape[0] > data.shape[1]:
            # Probably time x channels -> convert to channels x time
            data = data.T

        # Warn if more than 18 channels — models use only 18
        n_leads = data.shape[0]
        warnings_list = []
        if n_leads > 18:
            msg = (f"Input has {n_leads} leads/channels. Models require exactly 18 — "
                   f"{n_leads - 18} extra lead(s) will be dropped in AI analysis.")
            warnings_list.append(msg)
            print(f"[WARNING] {msg}")

        results = {}
        if warnings_list:
            results['warnings'] = warnings_list

        if self.biot_loader is not None:
            results['biot'] = self.biot_loader.predict(data, fs, temperature)
        if self.rf_loader is not None:
            results['random_forest'] = self.rf_loader.predict(data, fs)

        # Add comparison if both models exist
        if 'biot' in results and 'random_forest' in results:
            results['comparison'] = self._compare_models(results['biot'], results['random_forest'])

        return results

    def _compare_models(self, biot_result: Dict, rf_result: Dict) -> Dict:
        """Compare predictions from BIOT and Random Forest."""
        biot_pred = biot_result.get('prediction', 'unknown')
        biot_conf = biot_result.get('confidence', 0)
        rf_pred = rf_result.get('prediction', 'unknown')
        rf_conf = rf_result.get('confidence', 0)

        agreement = (biot_pred == rf_pred)

        if agreement:
            ensemble_pred = biot_pred
            ensemble_conf = (biot_conf + rf_conf) / 2
            source = 'Both models agree'
        else:
            if biot_conf > rf_conf:
                ensemble_pred = biot_pred
                ensemble_conf = biot_conf
                source = f'BIOT ({biot_conf:.2%}) > RF ({rf_conf:.2%})'
            else:
                ensemble_pred = rf_pred
                ensemble_conf = rf_conf
                source = f'RF ({rf_conf:.2%}) > BIOT ({biot_conf:.2%})'

        return {
            'biot_prediction': biot_pred,
            'biot_confidence': biot_conf,
            'rf_prediction': rf_pred,
            'rf_confidence': rf_conf,
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': ensemble_conf,
            'agreement': agreement,
            'confidence_source': source,
        }


# Singleton instance
_eeg_service = None

def get_eeg_service() -> EEGService:
    global _eeg_service
    if _eeg_service is None:
        _eeg_service = EEGService()
    return _eeg_service