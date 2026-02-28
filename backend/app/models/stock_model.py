"""
stock_model.py
--------------
Model loader for the Flask API.
Does NOT train — only loads pre-trained models and cached prediction data
produced by train_all.py.

Folder it reads from:
    saved_models/
        model_{ASSET}.pth
        scaler_{ASSET}.pkl
        cache_{ASSET}.json
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy as dc
from typing import Optional

MODELS_DIR = r"D:\Git\bin\signal-viewer\models\saved_models"

# ── asset catalogue ────────────────────────────────────────────────────────────
# These are the only assets the app supports.
# Keys are what the frontend sends, values are the display labels.

ASSET_CATALOGUE = {
    "stocks": {
        "AAPL":  "Apple",
        "GOOGL": "Google",
        "MSFT":  "Microsoft",
    },
    "commodities": {
        "GOLD":   "Gold",
        "SILVER": "Silver",
    },
    "currencies": {
        "USD": "US Dollar",
        "EUR": "Euro",
    },
}

# Flat set of all valid symbols for quick validation
ALL_SYMBOLS = {s for cat in ASSET_CATALOGUE.values() for s in cat}


# ── LSTM definition (must match train_all.py exactly) ─────────────────────────

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size        = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        device     = next(self.parameters()).device
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ── model loader ──────────────────────────────────────────────────────────────

def load_model(symbol: str) -> Optional[LSTM]:
    """Load a saved model from disk. Returns None if file not found."""
    path = os.path.join(MODELS_DIR, f"model_{symbol}.pth")
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location="cpu")
    model = LSTM(
        input_size          = 1,
        hidden_size         = checkpoint["hidden_size"],
        num_stacked_layers  = checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_scaler(symbol: str):
    """Load a saved MinMaxScaler from disk. Returns None if file not found."""
    path = os.path.join(MODELS_DIR, f"scaler_{symbol}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── cache reader (primary path — used by every API request) ──────────────────

def load_cache(symbol: str) -> Optional[dict]:
    """
    Load the pre-computed prediction cache for a symbol.
    This is what the API serves — no model inference needed at request time.

    Cache structure (written by train_all.py):
    {
        "train_dates":           [...],   # date strings for train period
        "test_dates":            [...],   # date strings for test period
        "actual_train":          [...],   # actual close prices (train)
        "actual_test":           [...],   # actual close prices (test)
        "train_predictions":     [...],   # model 1-step predictions (train)
        "test_predictions":      [...],   # model 1-step predictions (test)
        "recursive_predictions": [...],   # recursive predictions (test period)
        "split_index":           int,
        "forecast_7d": {
            "dates":  [...],              # next 7 calendar days
            "prices": [...]               # recursive forecast prices
        },
        "forecast_30d": {
            "dates":  [...],
            "prices": [...]
        }
    }
    """
    path = os.path.join(MODELS_DIR, f"cache_{symbol}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ── response builders ─────────────────────────────────────────────────────────

def build_graph1_response(cache: dict) -> dict:
    """
    Graph 1 — full history view.
    Returns everything the frontend needs to draw:
        black line  = actual close (train + test combined)
        blue line   = train predictions
        green line  = test predictions (1-step)
        red dashed  = recursive predictions (same test period)
        vertical line at split_index
    """
    return {
        "train_dates":            cache["train_dates"],
        "test_dates":             cache["test_dates"],
        "actual_train":           cache["actual_train"],
        "actual_test":            cache["actual_test"],
        "train_predictions":      cache["train_predictions"],
        "test_predictions":       cache["test_predictions"],
        "recursive_predictions":  cache["recursive_predictions"],
        "split_index":            cache["split_index"],
    }


def build_graph2_response(cache: dict, horizon: str) -> dict:
    """
    Graph 2 — future forecast only.
    horizon: '7d' or '30d'
    Returns dates + prices for the red dashed line only.
    """
    key = f"forecast_{horizon}"
    if key not in cache:
        return {"error": f"Horizon '{horizon}' not found in cache. "
                         f"Re-run train_all.py to generate it."}
    return {
        "horizon":       horizon,
        "forecast_dates":  cache[key]["dates"],
        "forecast_prices": cache[key]["prices"],
        # include last known price as anchor point for the chart
        "last_actual_date":  cache["test_dates"][-1],
        "last_actual_price": cache["actual_test"][-1],
    }