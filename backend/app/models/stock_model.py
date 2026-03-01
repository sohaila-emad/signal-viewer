"""
stock_model.py
--------------
Serves prediction data to the Flask API.

Priority order when a request comes in:
    1. cache JSON exists              → serve both Graph 1 + Graph 2 instantly
    2. CSV exists, no cache           → live inference → both graphs → save cache
    3. No CSV, model+scaler exist     → fetch from yfinance → Graph 2 only
                                        (Graph 1 skipped — model wasn't trained
                                         on this data so history view is meaningless)
    4. Nothing exists                 → return descriptive error
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy as dc
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

print(f"[stock_model] BASE_DIR   = {BASE_DIR}")
print(f"[stock_model] MODELS_DIR = {MODELS_DIR}")
print(f"[stock_model] DATA_DIR   = {DATA_DIR}")
print(f"[stock_model] MODELS_DIR exists: {os.path.exists(MODELS_DIR)}")
print(f"[stock_model] DATA_DIR   exists: {os.path.exists(DATA_DIR)}")

# Training constants — must match train_all.py exactly
LOOKBACK    = 7
TRAIN_SPLIT = 0.80
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── asset catalogue ────────────────────────────────────────────────────────────
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

# yfinance ticker symbols — may differ from our internal names
YFINANCE_TICKERS = {
    "AAPL":   "AAPL",
    "GOOGL":  "GOOGL",
    "MSFT":   "MSFT",
    "GOLD":   "GC=F",
    "SILVER": "SI=F",
    "USD":    "DX-Y.NYB",
    "EUR":    "EURUSD=X",
}

ASSET_FILES = {
    "AAPL":   "AAPL.csv",
    "GOOGL":  "GOOGL.csv",
    "MSFT":   "MSFT.csv",
    "GOLD":   "GOLD.csv",
    "SILVER": "SILVER.csv",
    "USD":    "USD.csv",
    "EUR":    "EUR.csv",
}

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
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ── loaders ────────────────────────────────────────────────────────────────────

def load_cache(symbol: str) -> Optional[dict]:
    path = os.path.join(MODELS_DIR, f"cache_{symbol}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_model(symbol: str) -> Optional[LSTM]:
    path = os.path.join(MODELS_DIR, f"model_{symbol}.pth")
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=DEVICE)
    model = LSTM(
        input_size         = 1,
        hidden_size        = checkpoint["hidden_size"],
        num_stacked_layers = checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(DEVICE)
    return model


def load_scaler(symbol: str) -> Optional[MinMaxScaler]:
    path = os.path.join(MODELS_DIR, f"scaler_{symbol}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(symbol: str) -> Optional[pd.DataFrame]:
    filename = ASSET_FILES.get(symbol)
    if not filename:
        return None
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return _parse_dataframe(pd.read_csv(path))


def fetch_from_yfinance(symbol: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        print("[stock_model] yfinance not installed. Run: pip install yfinance")
        return None

    ticker = YFINANCE_TICKERS.get(symbol)
    if not ticker:
        return None

    try:
        print(f"[stock_model] Fetching {symbol} ({ticker}) from yfinance...")
        df = yf.download(ticker, period="2y", progress=False)

        if df.empty:
            print(f"[stock_model] yfinance returned no data for {ticker}")
            return None
        
        # flatten to single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].copy()
        df.index.name = "Date"
        df.reset_index(inplace=True)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(inplace=True)
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"[stock_model] Fetched {len(df)} rows for {symbol}")
        return df

    except Exception as e:
        print(f"[stock_model] yfinance fetch failed for {symbol}: {e}")
        return None


def _parse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw CSV dataframe into a clean Date + Close dataframe."""
    if df.columns[0] == "Price":
        df = df.drop(index=[0, 1]).reset_index(drop=True)
        df.rename(columns={"Price": "Date"}, inplace=True)
    df = df[["Date", "Close"]].copy()
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── data helpers ───────────────────────────────────────────────────────────────

def prepare_dataframe_for_lstm(df: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    df = dc(df)
    df.set_index("Date", inplace=True)
    for i in range(1, n_steps + 1):
        df[f"Close(t-{i})"] = df["Close"].shift(i)
    df.dropna(inplace=True)
    return df


def build_tensors(df: pd.DataFrame, scaler: MinMaxScaler):
    """Build tensors using an already-fitted scaler (transform only, not fit)."""
    shifted   = prepare_dataframe_for_lstm(df, LOOKBACK)
    dates     = shifted.index.to_numpy()
    raw       = shifted.to_numpy().astype(float)
    scaled    = scaler.transform(raw)

    X         = dc(np.flip(scaled[:, 1:], axis=1))
    y         = scaled[:, 0]
    split_idx = int(len(X) * TRAIN_SPLIT)

    X_train = torch.tensor(X[:split_idx].reshape(-1, LOOKBACK, 1)).float()
    X_test  = torch.tensor(X[split_idx:].reshape(-1, LOOKBACK, 1)).float()
    y_train = torch.tensor(y[:split_idx].reshape(-1, 1)).float()
    y_test  = torch.tensor(y[split_idx:].reshape(-1, 1)).float()

    return X_train, X_test, y_train, y_test, split_idx, dates


def inverse_transform(scaled_preds: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    dummies       = np.zeros((len(scaled_preds), LOOKBACK + 1))
    dummies[:, 0] = scaled_preds
    return scaler.inverse_transform(dummies)[:, 0]


def compute_future_forecast(model: LSTM, last_window: np.ndarray,
                             scaler: MinMaxScaler, last_date: str,
                             n_days: int) -> dict:
    """
    Run recursive forecast n_days into the future from the given last_window.
    last_window shape: (LOOKBACK, 1) in scaled space.
    """
    window     = dc(last_window)
    fut_scaled = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_days):
            inp  = torch.tensor(window).float().unsqueeze(0).to(DEVICE)
            pred = model(inp).cpu().numpy().flatten()[0]
            fut_scaled.append(pred)
            window        = np.roll(window, -1, axis=0)
            window[-1, 0] = pred

    prices       = inverse_transform(np.array(fut_scaled), scaler)
    future_dates = pd.date_range(
        start=pd.Timestamp(last_date) + pd.Timedelta(days=1),
        periods=n_days,
        freq="D",
    )
    return {
        "dates":  [d.strftime("%Y-%m-%d") for d in future_dates],
        "prices": prices.tolist(),
    }


# ── path 2: full live inference (CSV exists, no cache) ────────────────────────

def run_live_inference(symbol: str, model: LSTM,
                       scaler: MinMaxScaler, df: pd.DataFrame) -> dict:
    """
    Full pipeline when CSV exists but cache doesn't.
    Produces both Graph 1 + Graph 2 data and saves to cache.
    """
    print(f"[{symbol}] Running live inference from CSV...")

    X_train, X_test, y_train, y_test, split_idx, dates = build_tensors(df, scaler)

    model.eval()
    with torch.no_grad():
        train_pred        = model(X_train.to(DEVICE)).cpu().numpy().flatten()
        train_predictions = inverse_transform(train_pred, scaler)
        actual_train      = inverse_transform(y_train.numpy().flatten(), scaler)

        test_pred         = model(X_test.to(DEVICE)).cpu().numpy().flatten()
        test_predictions  = inverse_transform(test_pred, scaler)
        actual_test       = inverse_transform(y_test.numpy().flatten(), scaler)

    # recursive predictions over test period
    last_window      = dc(X_train[-1].numpy())
    recursive_scaled = []

    with torch.no_grad():
        for _ in range(len(X_test)):
            inp  = torch.tensor(last_window).float().unsqueeze(0).to(DEVICE)
            pred = model(inp).cpu().numpy().flatten()[0]
            recursive_scaled.append(pred)
            last_window        = np.roll(last_window, -1, axis=0)
            last_window[-1, 0] = pred

    recursive_predictions = inverse_transform(np.array(recursive_scaled), scaler)

    train_dates = [str(d)[:10] for d in dates[:split_idx]]
    test_dates  = [str(d)[:10] for d in dates[split_idx:]]

    cache = {
        "graph1_available":       True,
        "train_dates":            train_dates,
        "test_dates":             test_dates,
        "actual_train":           actual_train.tolist(),
        "actual_test":            actual_test.tolist(),
        "train_predictions":      train_predictions.tolist(),
        "test_predictions":       test_predictions.tolist(),
        "recursive_predictions":  recursive_predictions.tolist(),
        "split_index":            split_idx,
        "forecast_7d":  compute_future_forecast(
            model, dc(X_test[-1].numpy()), scaler, test_dates[-1], 7
        ),
        "forecast_30d": compute_future_forecast(
            model, dc(X_test[-1].numpy()), scaler, test_dates[-1], 30
        ),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, f"cache_{symbol}.json"), "w") as f:
        json.dump(cache, f)
    print(f"[{symbol}] Cache saved.")

    return cache


# ── path 3: yfinance fallback (no CSV — Graph 2 only) ─────────────────────────

def run_yfinance_forecast(symbol: str, model: LSTM,
                          scaler: MinMaxScaler) -> dict:
    """
    Called when the CSV doesn't exist.
    Fetches recent data from yfinance, uses it ONLY to extract the last
    LOOKBACK window as the starting point for Graph 2 forecast.
    Graph 1 is marked unavailable since the model wasn't trained on this data.
    Does NOT cache — yfinance data changes daily so we always re-fetch.
    """
    df = fetch_from_yfinance(symbol)
    if df is None:
        return {"error": f"Could not fetch data for {symbol} from yfinance."}

    if len(df) < LOOKBACK:
        return {"error": f"Not enough data returned by yfinance for {symbol}."}

    # scale the recent prices using the existing scaler
    # we only need the last LOOKBACK closing prices as the seed window
    recent_close = df["Close"].values[-LOOKBACK:].reshape(-1, 1)

    # build a dummy shifted df just to get the right scaled format
    # we scale a (LOOKBACK,1) array — use scaler's known range
    min_val = scaler.data_min_[0]
    max_val = scaler.data_max_[0]
    feature_range = scaler.feature_range  # (-1, 1)

    scale    = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    scaled   = (recent_close - min_val) * scale + feature_range[0]
    last_window = scaled.reshape(LOOKBACK, 1)  # shape (LOOKBACK, 1)

    last_date = str(df["Date"].iloc[-1])[:10]

    return {
        "graph1_available": False,
        "graph1_message":   (
            "Full history graph is only available for assets you have trained on. "
            "Download the CSV and run train_all.py to enable it."
        ),
        "forecast_7d":  compute_future_forecast(model, last_window, scaler, last_date, 7),
        "forecast_30d": compute_future_forecast(model, last_window, scaler, last_date, 30),
        # pass last actual price + date so Graph 2 can anchor correctly
        "last_actual_date":  last_date,
        "last_actual_price": float(df["Close"].iloc[-1]),
    }


# ── main entry point ───────────────────────────────────────────────────────────

def get_prediction_data(symbol: str) -> dict:
    """
    Single function called by all routes.

    Flow:
        1. cache JSON      → serve instantly (both graphs)
        2. CSV + pth + pkl → live inference  (both graphs) → cache
        3. pth + pkl only  → yfinance fetch  (Graph 2 only, no cache)
        4. nothing         → error
    """

    # ── 1. cache hit — only use if CSV still exists ────────────────
    # If the CSV was deleted we ignore the cache too, because the
    # cached graph1 data would show a history the user no longer has.
    csv_exists = load_csv(symbol) is not None
    cache = load_cache(symbol)
    if cache is not None and csv_exists:
        return cache

    # ── load model and scaler (needed for paths 2 and 3) ──────────
    model  = load_model(symbol)
    scaler = load_scaler(symbol)

    if model is None or scaler is None:
        missing = []
        if model  is None: missing.append(f"model_{symbol}.pth")
        if scaler is None: missing.append(f"scaler_{symbol}.pkl")
        return {
            "error": (
                f"Missing trained model files: {missing}. "
                f"Please run: python train_all.py --assets {symbol}"
            )
        }

    # ── 2. CSV exists → full inference → both graphs ──────────────
    df = load_csv(symbol)
    if df is not None:
        return run_live_inference(symbol, model, scaler, df)

    # ── 3. No CSV → yfinance → Graph 2 only ───────────────────────
    return run_yfinance_forecast(symbol, model, scaler)


# ── response builders ─────────────────────────────────────────────────────────

def build_graph1_response(cache: dict) -> dict:
    """
    Returns graph1_available flag so the frontend knows
    whether to show the full history or the unavailable message.
    """
    if not cache.get("graph1_available", True):
        return {
            "graph1_available": False,
            "graph1_message":   cache.get("graph1_message", ""),
        }
    return {
        "graph1_available":       True,
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
    key = f"forecast_{horizon}"
    if key not in cache:
        return {
            "error": f"Horizon '{horizon}' not found. "
                     f"Delete cache JSON and re-request to regenerate."
        }

    # last_actual comes from cache (CSV path) or directly (yfinance path)
    last_date  = cache.get("last_actual_date")  or cache.get("test_dates",  [""]  )[-1]
    last_price = cache.get("last_actual_price") or cache.get("actual_test", [None])[-1]

    return {
        "horizon":           horizon,
        "graph1_available":  cache.get("graph1_available", True),
        "forecast_dates":    cache[key]["dates"],
        "forecast_prices":   cache[key]["prices"],
        "last_actual_date":  last_date,
        "last_actual_price": last_price,
    }