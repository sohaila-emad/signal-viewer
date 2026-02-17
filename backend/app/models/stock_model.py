"""
Stock Market Models for Signal Processing
- Stock data fetching (yfinance)
- Currency data
- Minerals/commodities data
- Time series prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Try to import yfinance, handle if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class StockDataFetcher:
    """Fetch stock market data using yfinance."""
    
    # Default stock symbols for different categories
    DEFAULT_STOCKS = {
        'tech': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'finance': ['JPM', 'BAC', 'GS', 'MS', 'C'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK'],
        'industrial': ['BA', 'CAT', 'HON', 'GE', 'MMM']
    }
    
    DEFAULT_CURRENCIES = {
        'major': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD'],
        'emerging': ['USD/MXN', 'USD/BRL', 'USD/INR', 'USD/ZAR', 'USD/RUB']
    }
    
    DEFAULT_MINERALS = {
        'precious': ['GC=F', 'SI=F', 'PL=F', 'PA=F'],  # Gold, Silver, Platinum, Palladium
        'industrial': ['CL=F', 'NG=F', 'HG=F', 'ZC=F'],  # Oil, Natural Gas, Copper, Corn
        'base': ['HG=F', 'ALU=F', 'ZINC=F', 'NICKEL=F']  # Copper, Aluminum, Zinc, Nickel
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def fetch_stock_data(self, symbol: str, period: str = '1y', 
                        interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        if not YFINANCE_AVAILABLE:
            return self._generate_sample_data(symbol, period, interval)
        
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return self._generate_sample_data(symbol, period, interval)
            
            # Cache the data
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return self._generate_sample_data(symbol, period, interval)
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = '1y',
                             interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks."""
        result = {}
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, period, interval)
            if df is not None:
                result[symbol] = df
        return result
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get current stock information."""
        if not YFINANCE_AVAILABLE:
            return {'symbol': symbol, 'name': f'Sample {symbol}', 'price': 100.0}
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except:
            return {'symbol': symbol, 'name': f'Sample {symbol}', 'price': 100.0}
    
    def _generate_sample_data(self, symbol: str, period: str, 
                             interval: str) -> pd.DataFrame:
        """Generate sample data for testing when yfinance is not available."""
        # Parse period
        if period == '1y':
            days = 365
        elif period == '6mo':
            days = 180
        elif period == '3mo':
            days = 90
        elif period == '1mo':
            days = 30
        else:
            days = 365
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random price data with trend
        np.random.seed(hash(symbol) % 10000)
        base_price = 100 + np.random.rand() * 100
        trend = np.linspace(0, 10, len(dates))
        noise = np.random.randn(len(dates)) * 2
        
        close = base_price + trend + noise
        open_prices = close + np.random.randn(len(dates)) * 0.5
        high = np.maximum(open_prices, close) + np.random.rand(len(dates)) * 0.5
        low = np.minimum(open_prices, close) - np.random.rand(len(dates)) * 0.5
        volume = np.random.randint(1000000, 10000000, len(dates))
        
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        return df


class StockPredictor:
    """Predict stock prices using various methods."""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
    
    def prepare_data(self, data: pd.DataFrame, lookback: int = 30) -> Tuple:
        """
        Prepare data for prediction.
        
        Args:
            data: DataFrame with stock data
            lookback: Number of days to use for prediction
            
        Returns:
            Tuple of (X, y) arrays
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close_prices = data['Close'].values
        
        X, y = [], []
        for i in range(lookback, len(close_prices)):
            X.append(close_prices[i-lookback:i])
            y.append(close_prices[i])
        
        return np.array(X), np.array(y)
    
    def predict_simple_moving_average(self, data: pd.DataFrame, 
                                     window: int = 20) -> np.ndarray:
        """
        Simple moving average prediction.
        
        Args:
            data: DataFrame with stock data
            window: Moving average window size
            
        Returns:
            Array of predicted prices
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close_prices = data['Close'].values
        sma = pd.Series(close_prices).rolling(window=window).mean().values
        
        # Predict next day as the last SMA value
        predictions = np.full(len(close_prices), sma[-1])
        
        return predictions
    
    def predict_exponential_smoothing(self, data: pd.DataFrame, 
                                     alpha: float = 0.3) -> np.ndarray:
        """
        Exponential smoothing prediction.
        
        Args:
            data: DataFrame with stock data
            alpha: Smoothing factor (0-1)
            
        Returns:
            Array of predicted prices
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close_prices = data['Close'].values
        n = len(close_prices)
        
        # Calculate exponential smoothing
        smoothed = np.zeros(n)
        smoothed[0] = close_prices[0]
        
        for i in range(1, n):
            smoothed[i] = alpha * close_prices[i] + (1 - alpha) * smoothed[i-1]
        
        # Predict next value
        predictions = np.full(n, smoothed[-1])
        
        return predictions
    
    def predict_linear_regression(self, data: pd.DataFrame, 
                                 lookback: int = 30) -> np.ndarray:
        """
        Linear regression prediction.
        
        Args:
            data: DataFrame with stock data
            lookback: Number of days to use for features
            
        Returns:
            Array of predicted prices
        """
        from sklearn.linear_model import LinearRegression
        
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close_prices = data['Close'].values
        n = len(close_prices)
        
        if n < lookback + 1:
            return np.full(n, close_prices[-1])
        
        # Prepare data
        X, y = self.prepare_data(data, lookback)
        
        # Fit model
        model = LinearRegression()
        model.fit(X[:-1], y[:-1])
        
        # Predict
        predictions = model.predict(X)
        
        # Pad the beginning
        full_predictions = np.zeros(n)
        full_predictions[:] = np.nan
        full_predictions[lookback:] = predictions
        
        # Fill NaN with last known value
        full_predictions = pd.Series(full_predictions).fillna(method='bfill').values
        
        return full_predictions
    
    def predict_next_days(self, data: pd.DataFrame, 
                         method: str = 'sma',
                         lookback: int = 30,
                         n_days: int = 7) -> Dict:
        """
        Predict next n days of stock prices.
        
        Args:
            data: DataFrame with stock data
            method: Prediction method ('sma', 'exp', 'lr')
            lookback: Number of days to use for features
            n_days: Number of days to predict
            
        Returns:
            Dictionary with predictions
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close_prices = data['Close'].values
        
        if method == 'sma':
            # Simple moving average
            window = min(lookback, len(close_prices) // 2)
            last_sma = pd.Series(close_prices[-window:]).mean()
            predictions = np.full(n_days, last_sma)
        elif method == 'exp':
            # Exponential smoothing
            alpha = 0.3
            last_smoothed = close_prices[-1]
            for i in range(1, len(close_prices)):
                last_smoothed = alpha * close_prices[i] + (1 - alpha) * last_smoothed
            predictions = np.full(n_days, last_smoothed)
        elif method == 'lr':
            # Linear regression
            try:
                X, y = self.prepare_data(data, lookback)
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X[:-1], y[:-1])
                
                # Use last window for prediction
                last_features = close_prices[-lookback:].reshape(1, -1)
                predictions = []
                current = last_features.copy()
                
                for _ in range(n_days):
                    pred = model.predict(current)[0]
                    predictions.append(pred)
                    current = np.roll(current, -1)
                    current[0, -1] = pred
                
                predictions = np.array(predictions)
            except:
                predictions = np.full(n_days, close_prices[-1])
        else:
            predictions = np.full(n_days, close_prices[-1])
        
        # Generate future dates
        last_date = data.index[-1] if hasattr(data.index, '__getitem__') else datetime.now()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=n_days, freq='D')
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predictions': predictions.tolist(),
            'method': method,
            'last_known_price': float(close_prices[-1])
        }
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate technical indicators.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Dictionary with technical indicators
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close = data['Close']
        high = data['High'] if 'High' in data.columns else close
        low = data['Low'] if 'Low' in data.columns else close
        volume = data['Volume'] if 'Volume' in data.columns else None
        
        # Moving averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        
        # Exponential moving averages
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20_val = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        upper_band = sma_20_val + (std_20 * 2)
        lower_band = sma_20_val - (std_20 * 2)
        
        # Average True Range (ATR)
        if 'High' in data.columns and 'Low' in data.columns:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close)
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
        else:
            atr = None
        
        return {
            'sma_20': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None,
            'sma_50': sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else None,
            'sma_200': sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else None,
            'ema_12': ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else None,
            'ema_26': ema_26.iloc[-1] if not pd.isna(ema_26.iloc[-1]) else None,
            'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else None,
            'macd_signal': signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else None,
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
            'bollinger_upper': upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else None,
            'bollinger_lower': lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else None,
            'atr': atr.iloc[-1] if atr is not None and not pd.isna(atr.iloc[-1]) else None
        }


def fetch_stock_data(symbol: str, period: str = '1y', 
                    interval: str = '1d') -> dict:
    """Fetch stock data and return as dictionary."""
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(symbol, period, interval)
    
    if df is None or df.empty:
        return {'error': 'Failed to fetch data'}
    
    return {
        'symbol': symbol,
        'data': df.to_dict(orient='records'),
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'columns': df.columns.tolist()
    }


def get_stock_info(symbol: str) -> dict:
    """Get stock information."""
    fetcher = StockDataFetcher()
    return fetcher.get_stock_info(symbol)


def predict_stock(symbol: str, method: str = 'sma', 
                n_days: int = 7) -> dict:
    """Predict stock prices."""
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(symbol)
    
    if df is None or df.empty:
        return {'error': 'Failed to fetch data'}
    
    predictor = StockPredictor()
    return predictor.predict_next_days(df, method=method, n_days=n_days)


def get_technical_indicators(symbol: str) -> dict:
    """Get technical indicators for a stock."""
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(symbol)
    
    if df is None or df.empty:
        return {'error': 'Failed to fetch data'}
    
    predictor = StockPredictor()
    return predictor.calculate_technical_indicators(df)
