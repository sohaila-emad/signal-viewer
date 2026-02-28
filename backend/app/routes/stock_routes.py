"""
stock_routes.py
---------------
Flask Blueprint for all stock-related API endpoints.

Endpoints:
    GET  /api/stocks/assets              → returns the full asset catalogue
    GET  /api/stocks/graph1/<symbol>     → full history graph data
    GET  /api/stocks/graph2/<symbol>     → future forecast data
                                           ?horizon=7d  (default)
                                           ?horizon=30d
"""

from flask import Blueprint, jsonify, request
from ..models.stock_model import (
    ASSET_CATALOGUE,
    ALL_SYMBOLS,
    load_cache,
    build_graph1_response,
    build_graph2_response,
)

stock_bp = Blueprint("stock", __name__)


# ── helper ────────────────────────────────────────────────────────────────────

def symbol_error(symbol: str):
    """Return a 404 JSON error for an unknown or untrained symbol."""
    return jsonify({
        "error": f"Symbol '{symbol}' not found. "
                 f"Valid symbols: {sorted(ALL_SYMBOLS)}. "
                 f"Make sure you have run train_all.py for this asset."
    }), 404


def cache_missing_error(symbol: str):
    """Return a 503 JSON error when the cache file hasn't been generated yet."""
    return jsonify({
        "error": f"No cache found for '{symbol}'. "
                 f"Please run: python train_all.py --assets {symbol}"
    }), 503


# ── endpoints ─────────────────────────────────────────────────────────────────

@stock_bp.route("/assets", methods=["GET"])
def get_assets():
    """
    Returns the full asset catalogue so the frontend can build its menus.

    Response shape:
    {
        "stocks":      {"AAPL": "Apple", "GOOGL": "Google", "MSFT": "Microsoft"},
        "commodities": {"GOLD": "Gold",  "SILVER": "Silver"},
        "currencies":  {"USD": "US Dollar", "EUR": "Euro"}
    }
    """
    return jsonify(ASSET_CATALOGUE)


@stock_bp.route("/graph1/<string:symbol>", methods=["GET"])
def get_graph1(symbol):
    """
    Full history graph data for a symbol.

    Returns everything needed to draw:
        - Actual close price (entire dataset)
        - Training predictions
        - Test predictions (1-step)
        - Recursive predictions (same test period)
        - Split index (where training ends)

    Response shape:
    {
        "symbol":                 "AAPL",
        "train_dates":            ["2020-01-02", ...],
        "test_dates":             ["2023-06-01", ...],
        "actual_train":           [300.1, 301.5, ...],
        "actual_test":            [185.2, 186.0, ...],
        "train_predictions":      [299.8, 301.0, ...],
        "test_predictions":       [184.9, 185.8, ...],
        "recursive_predictions":  [184.5, 183.2, ...],
        "split_index":            1200
    }
    """
    symbol = symbol.upper()

    if symbol not in ALL_SYMBOLS:
        return symbol_error(symbol)

    cache = load_cache(symbol)
    if cache is None:
        return cache_missing_error(symbol)

    response = build_graph1_response(cache)
    response["symbol"] = symbol
    return jsonify(response)


@stock_bp.route("/graph2/<string:symbol>", methods=["GET"])
def get_graph2(symbol):
    """
    Future forecast graph data for a symbol.

    Query params:
        horizon: '7d' (default) or '30d'

    Response shape:
    {
        "symbol":             "AAPL",
        "horizon":            "7d",
        "last_actual_date":   "2024-12-31",
        "last_actual_price":  250.10,
        "forecast_dates":     ["2025-01-01", "2025-01-02", ...],
        "forecast_prices":    [251.3, 252.0, ...]
    }
    """
    symbol  = symbol.upper()
    horizon = request.args.get("horizon", "7d").lower()

    if symbol not in ALL_SYMBOLS:
        return symbol_error(symbol)

    if horizon not in ("7d", "30d"):
        return jsonify({
            "error": "Invalid horizon. Use '7d' or '30d'."
        }), 400

    cache = load_cache(symbol)
    if cache is None:
        return cache_missing_error(symbol)

    response = build_graph2_response(cache, horizon)
    if "error" in response:
        return jsonify(response), 500

    response["symbol"] = symbol
    return jsonify(response)