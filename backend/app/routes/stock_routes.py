"""
stock_routes.py
---------------
Flask Blueprint for stock prediction endpoints.

Endpoints:
    GET /api/stocks/assets           → asset catalogue for frontend menus
    GET /api/stocks/graph1/<symbol>  → full history graph data
    GET /api/stocks/graph2/<symbol>  → future forecast
                                       ?horizon=7d (default) or ?horizon=30d
"""

import traceback
from flask import Blueprint, jsonify, request
from ..models.stock_model import (
    ASSET_CATALOGUE,
    ALL_SYMBOLS,
    get_prediction_data,
    build_graph1_response,
    build_graph2_response,
)

stock_bp = Blueprint("stock", __name__)


def symbol_error(symbol: str):
    return jsonify({
        "error": f"'{symbol}' is not a valid symbol. "
                 f"Valid options: {sorted(ALL_SYMBOLS)}"
    }), 404


@stock_bp.route("/assets", methods=["GET"])
def get_assets():
    """Returns the asset catalogue so the frontend can build its menus."""
    return jsonify(ASSET_CATALOGUE)


@stock_bp.route("/graph1/<string:symbol>", methods=["GET"])
def get_graph1(symbol):
    symbol = symbol.upper()

    if symbol not in ALL_SYMBOLS:
        return symbol_error(symbol)

    try:
        data = get_prediction_data(symbol)
    except Exception as e:
        # print full traceback to Flask terminal so you can see exactly what crashed
        traceback.print_exc()
        return jsonify({
            "error": f"Server error while loading {symbol}: {str(e)}. "
                     f"Check the Flask terminal for the full traceback."
        }), 500

    if "error" in data:
        return jsonify(data), 503

    response = build_graph1_response(data)
    response["symbol"] = symbol
    return jsonify(response)


@stock_bp.route("/graph2/<string:symbol>", methods=["GET"])
def get_graph2(symbol):
    symbol  = symbol.upper()
    horizon = request.args.get("horizon", "7d").lower()

    if symbol not in ALL_SYMBOLS:
        return symbol_error(symbol)

    if horizon not in ("7d", "30d"):
        return jsonify({"error": "Invalid horizon. Use '7d' or '30d'."}), 400

    try:
        data = get_prediction_data(symbol)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": f"Server error while loading {symbol}: {str(e)}. "
                     f"Check the Flask terminal for the full traceback."
        }), 500

    if "error" in data:
        return jsonify(data), 503

    response = build_graph2_response(data, horizon)
    if "error" in response:
        return jsonify(response), 500

    response["symbol"] = symbol
    return jsonify(response)