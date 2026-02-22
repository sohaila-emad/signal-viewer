"""
Stock Market Models for Signal Processing
- Stock data fetching (yfinance)
- Currency data
- Minerals/commodities data
- Time series prediction
- AI/LSTM prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

warnings.filterwarnings('ignore')

# Try to import yfinance, handle if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Try to import TensorFlow/Keras for LSTM, handle if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. AI predictions will use fallback methods.")


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
        'major': ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'CHF=X', 'AUDUSD=X'],
        'emerging': ['MXN=X', 'BRL=X', 'INR=X', 'ZAR=X', 'RUB=X']
    }
    
    DEFAULT_MINERALS = {
        'precious': ['GC=F', 'SI=F', 'PL=F', 'PA=F'],  # Gold, Silver, Platinum, Palladium
        'energy': ['CL=F', 'NG=F', 'RB=F', 'HO=F'],  # Oil, Natural Gas, Gasoline, Heating Oil
        'agriculture': ['ZC=F', 'ZW=F', 'ZS=F', 'KC=F']  # Corn, Wheat, Soybeans, Coffee
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def fetch_data(self, symbol: str, period: str = '6mo', 
                        interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch data for a given symbol.
        
        Args:
            symbol: Ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with data or None if failed
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
            print(f"Error fetching data for {symbol}: {e}")
            return self._generate_sample_data(symbol, period, interval)
    
    def fetch_multiple(self, symbols: List[str], period: str = '6mo',
                             interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        result = {}
        for symbol in symbols:
            df = self.fetch_data(symbol, period, interval)
            if df is not None:
                result[symbol] = df
        return result
    
    def get_info(self, symbol: str) -> Dict:
        """Get current symbol information."""
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
                'market_cap': info.get('marketCap', 0)
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


class MarketVisualizer:
    """Create interactive visualizations for market data."""
    
    @staticmethod
    def create_price_chart(data: Dict[str, pd.DataFrame], 
                           title: str = "Market Data") -> go.Figure:
        """Create price chart for multiple symbols."""
        fig = go.Figure()
        
        for symbol, df in data.items():
            if df is not None and not df.empty:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, 
                                 symbol: str) -> go.Figure:
        """Create candlestick chart for a single symbol."""
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol
        )])
        
        fig.update_layout(
            title=f'{symbol} - Candlestick Chart',
            xaxis_title="Date",
            yaxis_title="Price",
            template='plotly_dark',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @staticmethod
    def create_technical_chart(df: pd.DataFrame, 
                               indicators: Dict) -> go.Figure:
        """Create chart with technical indicators."""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & Indicators', 'Volume', 'RSI')
        )
        
        # Price and moving averages
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=False
        ), row=1, col=1)
        
        # Add moving averages
        if 'sma_20' in indicators and indicators['sma_20']:
            fig.add_trace(go.Scatter(
                x=df.index[-len(indicators['sma_20']):],
                y=indicators['sma_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        
        if 'sma_50' in indicators and indicators['sma_50']:
            fig.add_trace(go.Scatter(
                x=df.index[-len(indicators['sma_50']):],
                y=indicators['sma_50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)
        
        # RSI
        if 'rsi' in indicators and indicators['rsi']:
            rsi_data = indicators['rsi']
            rsi_dates = df.index[-len(rsi_data):]
            
            fig.add_trace(go.Scatter(
                x=rsi_dates,
                y=rsi_data,
                name='RSI',
                line=dict(color='purple', width=2)
            ), row=3, col=1)
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title='Technical Analysis Dashboard',
            template='plotly_dark',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(data: Dict[str, pd.DataFrame],
                               benchmark: str = 'SPY') -> go.Figure:
        """Create normalized comparison chart."""
        fig = go.Figure()
        
        # Normalize all series to 100 at start
        for symbol, df in data.items():
            if df is not None and not df.empty:
                normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Performance Comparison (Normalized to 100)',
            xaxis_title="Date",
            yaxis_title="Relative Performance",
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_prediction_chart(historical: pd.DataFrame,
                               predictions: Dict) -> go.Figure:
        """Create chart with historical data and predictions."""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        pred_dates = pd.to_datetime(predictions['dates'])
        
        if 'upper_bound' in predictions and 'lower_bound' in predictions:
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions['lower_bound'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=False
            ))
        
        # Prediction line
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions['predictions'],
            mode='lines+markers',
            name='Predictions',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Price Predictions - {predictions['method'].upper()}",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig


class LSTMPredictor:
    """AI-based prediction using LSTM neural network."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.lookback = 60
        self.trained = False
        self.min_val = None
        self.max_val = None
    
    def _prepare_data(self, data: np.ndarray, lookback: int = 60) -> Tuple:
        """Prepare data for LSTM model."""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _normalize_data(self, data: np.ndarray) -> Tuple:
        """Normalize data using min-max scaling."""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return data, lambda x: x, lambda x: x
        scaled = (data - min_val) / (max_val - min_val)
        return scaled, min_val, max_val
    
    def _build_model(self, lookback: int = 60):
        """Build LSTM model architecture."""
        if not TF_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, data: pd.DataFrame, epochs: int = 5, 
              batch_size: int = 64, lookback: int = 20) -> Dict:
        """
        Train LSTM model on data - OPTIMIZED FOR SPEED.
        
        Args:
            data: DataFrame with data
            epochs: Number of training epochs (default 5 for speed)
            batch_size: Batch size (default 64 for faster processing)
            lookback: Number of time steps to look back (default 20)
            
        Returns:
            Training history
        """
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        if 'Close' not in data.columns:
            return {'error': 'Data must contain Close column'}
        
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # Normalize data
        scaled_data, self.min_val, self.max_val = self._normalize_data(close_prices)
        
        # Prepare training data with reduced lookback
        X, y = self._prepare_data(scaled_data, lookback)
        self.lookback = lookback
        
        if len(X) < 20:
            return {'error': 'Insufficient data for training'}
        
        # Build ultra-fast LSTM model with minimal architecture
        model = Sequential([
            LSTM(16, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.1),
            LSTM(8, return_sequences=False),
            Dropout(0.1),
            Dense(4),
            Dense(1)
        ])
        # Use SGD with momentum for faster convergence
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        
        # Split data for validation
        split = int(len(X) * 0.85)  # More training data
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Early stopping with minimal patience for speed
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        # Train with minimal epochs
        history = self.model.fit(
            X_train, y_train,
            epochs=min(epochs, 5),  # Cap at 5 epochs for speed
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        self.trained = True
        
        # Calculate training metrics
        train_loss = float(np.min(history.history['loss']))
        val_loss = float(np.min(history.history['val_loss'])) if history.history['val_loss'] else train_loss
        
        return {
            'trained': True,
            'epochs_trained': len(history.history['loss']),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lookback': lookback
        }
    
    def predict(self, data: pd.DataFrame, n_days: int = 7) -> Dict:
        """
        Predict next n days using trained LSTM model.
        
        Args:
            data: DataFrame with data
            n_days: Number of days to predict
            
        Returns:
            Predictions dictionary
        """
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        if 'Close' not in data.columns:
            return {'error': 'Data must contain Close column'}
        
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # If not trained, train first
        if not self.trained:
            train_result = self.train(data)
            if 'error' in train_result:
                return train_result
        
        # Normalize data
        scaled_data = (close_prices - self.min_val) / (self.max_val - self.min_val)
        
        # Use last lookback days to start prediction
        last_sequence = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_days):
            # Predict next day
            pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = predictions * (self.max_val - self.min_val) + self.min_val
        predictions = predictions.flatten().tolist()
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=n_days, freq='D')
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predictions': predictions,
            'method': 'lstm',
            'last_known_price': float(close_prices[-1]),
            'trained': self.trained
        }
    
    def predict_with_confidence(self, data: pd.DataFrame, n_days: int = 7) -> Dict:
        """
        Predict with confidence intervals using Monte Carlo dropout.
        
        Args:
            data: DataFrame with data
            n_days: Number of days to predict
            
        Returns:
            Predictions with confidence intervals
        """
        if not TF_AVAILABLE:
            # Fallback to simple prediction if TensorFlow not available
            return self._simple_prediction(data, n_days)
        
        try:
            if 'Close' not in data.columns:
                return {'error': 'Data must contain Close column'}
            
            # Check minimum data requirements
            if len(data) < 50:
                return {'error': 'Insufficient data for LSTM. Need at least 50 data points.'}
            
            close_prices = data['Close'].values.reshape(-1, 1)
            
            # If not trained, train first
            if not self.trained:
                train_result = self.train(data)
                if 'error' in train_result:
                    return self._simple_prediction(data, n_days)
            
            # Check if model was trained successfully
            if self.model is None or self.min_val is None or self.max_val is None:
                return self._simple_prediction(data, n_days)
            
            # Normalize data
            scaled_data = (close_prices - self.min_val) / (self.max_val - self.min_val)
            
            # Use last lookback days
            last_sequence = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
            
            # Multiple predictions with dropout (Monte Carlo)
            n_samples = 10
            all_predictions = []
            
            for _ in range(n_samples):
                current_seq = last_sequence.copy()
                preds = []
                
                for _ in range(n_days):
                    try:
                        pred = self.model.predict(current_seq, verbose=0)[0, 0]
                    except Exception:
                        pred = current_seq[0, -1, 0]  # Use last value as fallback
                    preds.append(pred)
                    current_seq = np.roll(current_seq, -1, axis=1)
                    current_seq[0, -1, 0] = pred
                
                # Inverse transform
                preds = np.array(preds).reshape(-1, 1)
                preds = preds * (self.max_val - self.min_val) + self.min_val
                preds = preds.flatten()
                all_predictions.append(preds)
            
            all_predictions = np.array(all_predictions)
            
            # Calculate mean and std
            mean_predictions = np.mean(all_predictions, axis=0)
            std_predictions = np.std(all_predictions, axis=0)
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=n_days, freq='D')
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'predictions': mean_predictions.tolist(),
                'upper_bound': (mean_predictions + 1.96 * std_predictions).tolist(),
                'lower_bound': (mean_predictions - 1.96 * std_predictions).tolist(),
                'method': 'lstm',
                'last_known_price': float(close_prices[-1]),
                'confidence': '95%'
            }
        except Exception as e:
            # Fallback to simple prediction on any error
            print(f"LSTM prediction error: {str(e)}")
            return self._simple_prediction(data, n_days)
    
    def _simple_prediction(self, data: pd.DataFrame, n_days: int = 7) -> Dict:
        """
        Simple fallback prediction when LSTM fails.
        
        Args:
            data: DataFrame with data
            n_days: Number of days to predict
            
        Returns:
            Simple predictions with confidence intervals
        """
        if 'Close' not in data.columns:
            return {'error': 'Data must contain Close column'}
        
        close_prices = data['Close'].values
        
        if len(close_prices) < 2:
            return {'error': 'Insufficient data for prediction'}
        
        # Calculate trend using linear regression on recent data
        try:
            X = np.arange(len(close_prices)).reshape(-1, 1)
            y = close_prices
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Predict next n days
            future_X = np.arange(len(close_prices), len(close_prices) + n_days).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Calculate confidence intervals based on residuals
            residuals = y - model.predict(X)
            std_error = np.std(residuals) if len(residuals) > 0 else close_prices.std() * 0.1
            
            # Ensure predictions are positive
            predictions = np.maximum(predictions, close_prices[-1] * 0.5)
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=n_days, freq='D')
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'predictions': predictions.tolist(),
                'upper_bound': (predictions + 1.96 * std_error).tolist(),
                'lower_bound': (predictions - 1.96 * std_error).tolist(),
                'method': 'lr_fallback',
                'last_known_price': float(close_prices[-1]),
                'confidence': '95%',
                'note': 'Used linear regression fallback due to LSTM unavailable'
            }
        except Exception as e:
            # Ultimate fallback - use last price with small variation
            last_price = float(close_prices[-1])
            predictions = [last_price] * n_days
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=n_days, freq='D')
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'predictions': predictions,
                'upper_bound': [last_price * 1.05] * n_days,
                'lower_bound': [last_price * 0.95] * n_days,
                'method': 'simple_fallback',
                'last_known_price': last_price,
                'confidence': '90%',
                'note': 'Used simple fallback due to error: ' + str(e)
            }


class TechnicalIndicatorCalculator:
    """Calculate comprehensive technical indicators."""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators."""
        if 'Close' not in df.columns:
            return {'error': 'Data must contain Close column'}
        
        close = df['Close']
        high = df['High'] if 'High' in df.columns else close
        low = df['Low'] if 'Low' in df.columns else close
        volume = df['Volume'] if 'Volume' in df.columns else None
        
        # Moving Averages
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20_val = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        bb_upper = sma_20_val + (std_20 * 2)
        bb_middle = sma_20_val
        bb_lower = sma_20_val - (std_20 * 2)
        
        # Stochastic
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        stochastic = 100 * (close - low_14) / (high_14 - low_14)
        
        # Average True Range
        if 'High' in df.columns and 'Low' in df.columns:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close)
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
        else:
            atr = None
        
        # On-Balance Volume
        if volume is not None:
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        else:
            obv = None
        
        # Return all indicators
        return {
            'sma_20': sma_20.dropna().tolist(),
            'sma_50': sma_50.dropna().tolist(),
            'sma_200': sma_200.dropna().tolist(),
            'ema_12': ema_12.dropna().tolist(),
            'ema_26': ema_26.dropna().tolist(),
            'macd': macd.dropna().tolist(),
            'macd_signal': macd_signal.dropna().tolist(),
            'macd_hist': macd_hist.dropna().tolist(),
            'rsi': rsi.dropna().tolist(),
            'bb_upper': bb_upper.dropna().tolist(),
            'bb_middle': bb_middle.dropna().tolist(),
            'bb_lower': bb_lower.dropna().tolist(),
            'stochastic': stochastic.dropna().tolist(),
            'atr': atr.dropna().tolist() if atr is not None else [],
            'obv': obv.dropna().tolist() if obv is not None else [],
            'latest': {
                'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
                'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
                'sma_200': float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None,
                'ema_12': float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else None,
                'ema_26': float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else None,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None,
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                'bb_upper': float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
                'bb_middle': float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else None,
                'bb_lower': float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
                'stochastic': float(stochastic.iloc[-1]) if not pd.isna(stochastic.iloc[-1]) else None,
                'atr': float(atr.iloc[-1]) if atr is not None and not pd.isna(atr.iloc[-1]) else None,
            }
        }


class MarketAnalyzer:
    """Main class for market analysis and visualization."""
    
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.visualizer = MarketVisualizer()
        self.lstm_predictor = LSTMPredictor()
        self.technical_calculator = TechnicalIndicatorCalculator()
    
    def analyze_stocks(self, symbols: List[str] = None, 
                      category: str = 'tech') -> Dict:
        """Analyze stocks."""
        if symbols is None:
            symbols = StockDataFetcher.DEFAULT_STOCKS.get(category, 
                        StockDataFetcher.DEFAULT_STOCKS['tech'])
        
        print(f"Fetching stock data for: {symbols}")
        data = self.fetcher.fetch_multiple(symbols)
        
        # Create visualizations
        charts = {
            'price_chart': self.visualizer.create_price_chart(data, f"Stock Prices - {category.title()}"),
            'comparison_chart': self.visualizer.create_comparison_chart(data)
        }
        
        # Get current info
        info = {symbol: self.fetcher.get_info(symbol) for symbol in data.keys()}
        
        return {
            'type': 'stocks',
            'category': category,
            'symbols': list(data.keys()),
            'data': {k: v.to_dict(orient='records') for k, v in data.items()},
            'info': info,
            'charts': charts
        }
    
    def analyze_currencies(self, symbols: List[str] = None,
                          category: str = 'major') -> Dict:
        """Analyze currencies."""
        if symbols is None:
            symbols = StockDataFetcher.DEFAULT_CURRENCIES.get(category,
                        StockDataFetcher.DEFAULT_CURRENCIES['major'])
        
        print(f"Fetching currency data for: {symbols}")
        data = self.fetcher.fetch_multiple(symbols)
        
        # Create visualizations
        charts = {
            'price_chart': self.visualizer.create_price_chart(data, f"Currency Rates - {category.title()}"),
            'comparison_chart': self.visualizer.create_comparison_chart(data)
        }
        
        # Get current info
        info = {}
        for symbol, df in data.items():
            if df is not None and not df.empty:
                info[symbol] = {
                    'symbol': symbol,
                    'price': float(df['Close'].iloc[-1]),
                    'change': float(df['Close'].iloc[-1] - df['Close'].iloc[-2]) if len(df) > 1 else 0,
                    'high': float(df['High'].max()),
                    'low': float(df['Low'].min())
                }
        
        return {
            'type': 'currencies',
            'category': category,
            'symbols': list(data.keys()),
            'data': {k: v.to_dict(orient='records') for k, v in data.items()},
            'info': info,
            'charts': charts
        }
    
    def analyze_minerals(self, symbols: List[str] = None,
                        category: str = 'precious') -> Dict:
        """Analyze minerals/commodities."""
        if symbols is None:
            symbols = StockDataFetcher.DEFAULT_MINERALS.get(category,
                        StockDataFetcher.DEFAULT_MINERALS['precious'])
        
        print(f"Fetching minerals data for: {symbols}")
        data = self.fetcher.fetch_multiple(symbols)
        
        # Create visualizations
        charts = {
            'price_chart': self.visualizer.create_price_chart(data, f"Commodities - {category.title()}"),
            'comparison_chart': self.visualizer.create_comparison_chart(data)
        }
        
        # Get current info
        info = {}
        for symbol, df in data.items():
            if df is not None and not df.empty:
                name_map = {
                    'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum', 'PA=F': 'Palladium',
                    'CL=F': 'Crude Oil', 'NG=F': 'Natural Gas', 'RB=F': 'Gasoline', 'HO=F': 'Heating Oil',
                    'ZC=F': 'Corn', 'ZW=F': 'Wheat', 'ZS=F': 'Soybeans', 'KC=F': 'Coffee'
                }
                info[symbol] = {
                    'symbol': symbol,
                    'name': name_map.get(symbol, symbol),
                    'price': float(df['Close'].iloc[-1]),
                    'change': float(df['Close'].iloc[-1] - df['Close'].iloc[-2]) if len(df) > 1 else 0,
                    'high': float(df['High'].max()),
                    'low': float(df['Low'].min())
                }
        
        return {
            'type': 'minerals',
            'category': category,
            'symbols': list(data.keys()),
            'data': {k: v.to_dict(orient='records') for k, v in data.items()},
            'info': info,
            'charts': charts
        }
    
    def predict_future(self, symbol: str, n_days: int = 7,
                      use_lstm: bool = True) -> Dict:
        """Predict future prices for a symbol."""
        print(f"Fetching data for {symbol}...")
        data = self.fetcher.fetch_data(symbol, period='1y')
        
        if data is None or data.empty:
            return {'error': f'Failed to fetch data for {symbol}'}
        
        print(f"Making predictions for next {n_days} days...")
        
        if use_lstm and TF_AVAILABLE:
            predictions = self.lstm_predictor.predict_with_confidence(data, n_days)
        else:
            # Simple prediction methods
            predictor = StockPredictor()
            predictions = predictor.predict_next_days(data, method='lr', n_days=n_days)
        
        # Create visualization
        chart = self.visualizer.create_prediction_chart(data, predictions)
        
        # Calculate technical indicators
        indicators = self.technical_calculator.calculate_all(data)
        
        return {
            'symbol': symbol,
            'predictions': predictions,
            'technical_indicators': indicators['latest'],
            'chart': chart,
            'current_price': float(data['Close'].iloc[-1])
        }
    
    def generate_report(self, symbols: List[str] = None) -> Dict:
        """Generate comprehensive market report."""
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'GC=F', 'EURUSD=X']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'stocks': {},
            'currencies': {},
            'minerals': {},
            'predictions': {}
        }
        
        # Analyze each category
        for symbol in symbols:
            data = self.fetcher.fetch_data(symbol)
            if data is not None and not data.empty:
                # Determine type
                if '=X' in symbol:
                    category = 'currencies'
                elif '=F' in symbol:
                    category = 'minerals'
                else:
                    category = 'stocks'
                
                # Get info and indicators
                info = self.fetcher.get_info(symbol)
                indicators = self.technical_calculator.calculate_all(data)
                
                # Make prediction
                predictor = LSTMPredictor()
                prediction = predictor.predict(data, n_days=5)
                
                report[category][symbol] = {
                    'info': info,
                    'current_price': float(data['Close'].iloc[-1]),
                    'change_1d': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]) if len(data) > 1 else 0,
                    'change_1w': float(data['Close'].iloc[-1] - data['Close'].iloc[-6]) if len(data) > 5 else 0,
                    'change_1m': float(data['Close'].iloc[-1] - data['Close'].iloc[-21]) if len(data) > 20 else 0,
                    'volume': float(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0,
                    'indicators': indicators['latest'],
                    'prediction': prediction
                }
        
        return report


class StockPredictor:
    """Simple stock predictor for fallback methods."""
    
    def predict_next_days(self, data: pd.DataFrame, method: str = 'lr',
                         lookback: int = 30, n_days: int = 7) -> Dict:
        """Simple prediction method."""
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        close_prices = data['Close'].values
        
        if method == 'lr':
            # Simple linear regression
            try:
                from sklearn.linear_model import LinearRegression
                X = np.arange(len(close_prices)).reshape(-1, 1)
                y = close_prices
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict next n days
                future_X = np.arange(len(close_prices), len(close_prices) + n_days).reshape(-1, 1)
                predictions = model.predict(future_X)
            except:
                predictions = np.full(n_days, close_prices[-1])
        else:
            predictions = np.full(n_days, close_prices[-1])
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=n_days, freq='D')
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'predictions': predictions.tolist(),
            'method': method,
            'last_known_price': float(close_prices[-1])
        }


def main():
    """Main function to run the analysis."""
    print("=" * 60)
    print("STOCK MARKET ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Create analyzer
    analyzer = MarketAnalyzer()
    
    # Check if yfinance is available
    if not YFINANCE_AVAILABLE:
        print("\n⚠️  yfinance not installed. Using sample data.")
        print("Install with: pip install yfinance")
    
    if not TF_AVAILABLE:
        print("\n⚠️  TensorFlow not installed. LSTM predictions will use fallback methods.")
        print("Install with: pip install tensorflow")
    
    print("\n" + "=" * 60)
    print("1. STOCK MARKET ANALYSIS")
    print("=" * 60)
    
    # Analyze tech stocks
    print("\n📈 Analyzing Tech Stocks...")
    stocks_analysis = analyzer.analyze_stocks(category='tech')
    print(f"✅ Found {len(stocks_analysis['symbols'])} stocks")
    
    # Display stock info
    print("\nCurrent Stock Prices:")
    for symbol, info in stocks_analysis['info'].items():
        print(f"  {symbol}: ${info['price']:.2f} ({info.get('change_percent', 0):+.2f}%)")
    
    print("\n" + "=" * 60)
    print("2. CURRENCY MARKET ANALYSIS")
    print("=" * 60)
    
    # Analyze major currencies
    print("\n💱 Analyzing Major Currencies...")
    currency_analysis = analyzer.analyze_currencies(category='major')
    print(f"✅ Found {len(currency_analysis['symbols'])} currency pairs")
    
    # Display currency info
    print("\nCurrent Exchange Rates:")
    for symbol, info in currency_analysis['info'].items():
        name = symbol.replace('=X', '')
        print(f"  {name}: {info['price']:.4f}")
    
    print("\n" + "=" * 60)
    print("3. COMMODITIES ANALYSIS")
    print("=" * 60)
    
    # Analyze precious metals
    print("\n⛏️  Analyzing Precious Metals...")
    minerals_analysis = analyzer.analyze_minerals(category='precious')
    print(f"✅ Found {len(minerals_analysis['symbols'])} commodities")
    
    # Display commodity info
    print("\nCurrent Commodity Prices:")
    for symbol, info in minerals_analysis['info'].items():
        print(f"  {info['name']}: ${info['price']:.2f}")
    
    print("\n" + "=" * 60)
    print("4. PRICE PREDICTIONS")
    print("=" * 60)
    
    # Predict for a major stock
    print("\n🔮 Predicting AAPL stock price for next 7 days...")
    prediction = analyzer.predict_future('AAPL', n_days=7, use_lstm=True)
    
    if 'error' not in prediction:
        print(f"\nCurrent Price: ${prediction['current_price']:.2f}")
        print("\nPredicted Prices:")
        for i, (date, price) in enumerate(zip(prediction['predictions']['dates'], 
                                              prediction['predictions']['predictions'])):
            print(f"  Day {i+1} ({date}): ${price:.2f}")
        
        if 'rsi' in prediction['technical_indicators']:
            print(f"\nTechnical Indicators:")
            print(f"  RSI: {prediction['technical_indicators']['rsi']:.2f}")
            print(f"  MACD: {prediction['technical_indicators']['macd']:.4f}")
            print(f"  SMA 20: ${prediction['technical_indicators']['sma_20']:.2f}")
            print(f"  SMA 50: ${prediction['technical_indicators']['sma_50']:.2f}")
    else:
        print(f"⚠️  {prediction['error']}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\n📊 Visualizations have been created and can be displayed")
    print("Run in Jupyter notebook to see interactive charts")
    
    return {
        'stocks': stocks_analysis,
        'currencies': currency_analysis,
        'minerals': minerals_analysis,
        'predictions': prediction if 'error' not in prediction else None
    }


