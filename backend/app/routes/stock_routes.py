from flask import Blueprint, request, jsonify
from ..services.stock_service import get_stock_service

stock_bp = Blueprint('stock', __name__)

# Get service instance
stock_service = get_stock_service()


# =====================================================
# STOCK MARKET DATA ENDPOINTS
# =====================================================

@stock_bp.route('/data/<string:symbol>', methods=['GET'])
def get_stock_data(symbol):
    """
    Get stock data for a symbol.
    
    Query parameters:
    - period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    - interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
    """
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    try:
        result = stock_service.get_stock_data(symbol.upper(), period, interval)
        if 'error' in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/info/<string:symbol>', methods=['GET'])
def get_info(symbol):
    """Get current stock information."""
    try:
        result = stock_service.get_stock_info(symbol.upper())
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/list/<string:category>', methods=['GET'])
def get_stock_list(category):
    """Get list of stocks for a category (tech, finance, energy, healthcare, industrial)."""
    valid_categories = ['tech', 'finance', 'energy', 'healthcare', 'industrial']
    
    if category not in valid_categories:
        return jsonify({'error': f'Invalid category. Choose from: {valid_categories}'}), 400
    
    try:
        stocks = stock_service.get_stock_list(category)
        return jsonify({'category': category, 'stocks': stocks})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# CURRENCY DATA ENDPOINTS
# =====================================================

@stock_bp.route('/currency/list/<string:category>', methods=['GET'])
def get_currency_list(category):
    """Get list of currency pairs (major, emerging)."""
    valid_categories = ['major', 'emerging']
    
    if category not in valid_categories:
        return jsonify({'error': f'Invalid category. Choose from: {valid_categories}'}), 400
    
    try:
        currencies = stock_service.get_currency_list(category)
        return jsonify({'category': category, 'currencies': currencies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/currency/data/<string:symbol>', methods=['GET'])
def get_currency_data(symbol):
    """Get currency data."""
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    try:
        # Currency symbols typically use =X suffix in yfinance
        if not symbol.endswith('=X'):
            symbol = symbol.replace('/', '=') + '=X'
        
        result = stock_service.get_stock_data(symbol, period, interval)
        if 'error' in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# MINERALS/COMMODITIES ENDPOINTS
# =====================================================

@stock_bp.route('/mineral/list/<string:category>', methods=['GET'])
def get_mineral_list(category):
    """Get list of minerals/commodities (precious, industrial, base)."""
    valid_categories = ['precious', 'industrial', 'base']
    
    if category not in valid_categories:
        return jsonify({'error': f'Invalid category. Choose from: {valid_categories}'}), 400
    
    try:
        minerals = stock_service.get_mineral_list(category)
        return jsonify({'category': category, 'minerals': minerals})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/mineral/data/<string:symbol>', methods=['GET'])
def get_mineral_data(symbol):
    """Get mineral/commodity data."""
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    try:
        # Add =F suffix for futures
        if not symbol.endswith('=F'):
            symbol = symbol + '=F'
        
        result = stock_service.get_stock_data(symbol, period, interval)
        if 'error' in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# PREDICTION ENDPOINTS
# =====================================================

@stock_bp.route('/predict/<string:symbol>', methods=['GET'])
def predict_stock(symbol):
    """
    Predict stock prices.
    
    Query parameters:
    - method: Prediction method ('sma', 'exp', 'lr', 'lstm')
    - n_days: Number of days to predict (default: 7)
    - period: Prediction period ('7d', '1mo', '3mo', '6mo', '1y')
               - '7d': 7 days prediction
               - '1mo': 1 month prediction (~30 days)
               - '3mo': 3 months prediction (~90 days)
               - '6mo': 6 months prediction (~180 days)
    """
    method = request.args.get('method', 'sma')
    n_days = request.args.get('n_days', type=int, default=7)
    period = request.args.get('period', '1y')
    
    if method not in ['sma', 'exp', 'lr', 'lstm']:
        return jsonify({'error': 'Invalid method. Choose from: sma, exp, lr, lstm'}), 400
    
    # Map period to n_days
    period_to_days = {'7d': 7, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
    if period in period_to_days and n_days == 7:
        n_days = period_to_days[period]
    
    if n_days < 1 or n_days > 365:
        return jsonify({'error': 'n_days must be between 1 and 365'}), 400
    
    valid_periods = ['7d', '1mo', '3mo', '6mo', '1y', '2y']
    if period not in valid_periods:
        return jsonify({'error': f'Invalid period. Choose from: {valid_periods}'}), 400
    
    try:
        result = stock_service.predict_stock_price(symbol.upper(), method, n_days, period)
        if 'error' in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# TECHNICAL ANALYSIS ENDPOINTS
# =====================================================

@stock_bp.route('/analysis/<string:symbol>', methods=['GET'])
def get_technical_analysis(symbol):
    """Get technical indicators for a stock."""
    try:
        result = stock_service.get_technical_analysis(symbol.upper())
        if 'error' in result:
            return jsonify(result), 404
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/compare', methods=['POST'])
def compare_stocks():
    """Compare multiple stocks."""
    data = request.get_json()
    
    if not data or 'symbols' not in data:
        return jsonify({'error': 'Missing symbols list'}), 400
    
    symbols = data['symbols']
    period = data.get('period', '1y')
    
    if not isinstance(symbols, list) or len(symbols) < 2:
        return jsonify({'error': 'Provide at least 2 symbols to compare'}), 400
    
    try:
        result = stock_service.compare_stocks(symbols, period)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# MARKET SUMMARY ENDPOINTS
# =====================================================

@stock_bp.route('/market/summary', methods=['GET'])
def get_market_summary():
    """Get overall market summary."""
    try:
        result = stock_service.get_market_summary()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# LEGACY INVENTORY ENDPOINTS (kept for backward compatibility)
# =====================================================

# Sample stock inventory (kept for backward compatibility)
inventory = []

@stock_bp.route('/inventory', methods=['GET'])
def get_inventory():
    """Get all stock items (inventory)"""
    return jsonify({
        'inventory': inventory,
        'count': len(inventory)
    })


@stock_bp.route('/inventory', methods=['POST'])
def add_stock_item():
    """Add a new stock item to inventory"""
    data = request.get_json()
    
    if not data or 'name' not in data or 'quantity' not in data:
        return jsonify({'error': 'Missing required fields: name, quantity'}), 400
    
    new_item = {
        'id': len(inventory),
        'name': data['name'],
        'quantity': data['quantity'],
        'price': data.get('price', 0.0),
        'category': data.get('category', 'general'),
        'supplier': data.get('supplier', '')
    }
    
    inventory.append(new_item)
    return jsonify(new_item), 201


@stock_bp.route('/inventory/<int:item_id>', methods=['GET'])
def get_stock_item(item_id):
    """Get a specific stock item from inventory"""
    if 0 <= item_id < len(inventory):
        return jsonify(inventory[item_id])
    return jsonify({'error': 'Item not found'}), 404
