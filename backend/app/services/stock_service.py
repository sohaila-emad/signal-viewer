"""
Stock Market Service for Signal Processing
Provides service layer for stock market data and prediction
"""

import pandas as pd
from typing import Optional, Dict, List
from ..models.stock_model import (
    StockDataFetcher,
    StockPredictor,
    fetch_stock_data,
    get_stock_info,
    predict_stock,
    get_technical_indicators
)


class StockService:
    """Service for stock market operations."""
    
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.predictor = StockPredictor()
    
    def get_stock_data(self, symbol: str, period: str = '1y', 
                      interval: str = '1d') -> Dict:
        """
        Get stock data for a symbol.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL', 'GOOGL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            
        Returns:
            Dictionary with stock data
        """
        try:
            df = self.fetcher.fetch_stock_data(symbol, period, interval)
            
            if df is None or df.empty:
                return {'error': f'No data found for symbol: {symbol}'}
            
            # Convert to JSON-serializable format
            return {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'data': df.to_dict(orient='records'),
                'dates': df.index.strftime('%Y-%m-%d').tolist() if hasattr(df.index, 'strftime') else [str(d) for d in df.index],
                'columns': df.columns.tolist(),
                'latest_price': float(df['Close'].iloc[-1]) if 'Close' in df.columns else None,
                'latest_volume': int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else None
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_multiple_stocks(self, symbols: List[str], period: str = '1y',
                           interval: str = '1d') -> Dict:
        """Get data for multiple stocks."""
        result = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, period, interval)
            if 'error' not in data:
                result[symbol] = data
        return result
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get current stock information."""
        try:
            return self.fetcher.get_stock_info(symbol)
        except Exception as e:
            return {'error': str(e)}
    
    def get_stock_list(self, category: str = 'tech') -> List[str]:
        """Get list of stock symbols for a category."""
        return StockDataFetcher.DEFAULT_STOCKS.get(category, [])
    
    def get_currency_list(self, category: str = 'major') -> List[str]:
        """Get list of currency pairs for a category."""
        return StockDataFetcher.DEFAULT_CURRENCIES.get(category, [])
    
    def get_mineral_list(self, category: str = 'precious') -> List[str]:
        """Get list of mineral/commodity symbols for a category."""
        return StockDataFetcher.DEFAULT_MINERALS.get(category, [])
    
    def predict_stock_price(self, symbol: str, method: str = 'sma',
                          n_days: int = 7) -> Dict:
        """
        Predict stock prices.
        
        Args:
            symbol: Stock ticker
            method: Prediction method ('sma', 'exp', 'lr')
            n_days: Number of days to predict
            
        Returns:
            Dictionary with predictions
        """
        try:
            df = self.fetcher.fetch_stock_data(symbol)
            
            if df is None or df.empty:
                return {'error': f'No data found for symbol: {symbol}'}
            
            return self.predictor.predict_next_days(
                df, 
                method=method, 
                n_days=n_days
            )
        except Exception as e:
            return {'error': str(e)}
    
    def get_technical_analysis(self, symbol: str) -> Dict:
        """Get technical indicators for a stock."""
        try:
            df = self.fetcher.fetch_stock_data(symbol)
            
            if df is None or df.empty:
                return {'error': f'No data found for symbol: {symbol}'}
            
            indicators = self.predictor.calculate_technical_indicators(df)
            
            # Get recent price data
            recent_data = df.tail(30).to_dict(orient='records')
            
            return {
                'symbol': symbol,
                'indicators': indicators,
                'recent_prices': recent_data
            }
        except Exception as e:
            return {'error': str(e)}
    
    def compare_stocks(self, symbols: List[str], period: str = '1y') -> Dict:
        """Compare multiple stocks."""
        comparison = {}
        
        for symbol in symbols:
            df = self.fetcher.fetch_stock_data(symbol, period)
            if df is not None and not df.empty:
                close_prices = df['Close']
                comparison[symbol] = {
                    'latest_price': float(close_prices.iloc[-1]),
                    'price_change': float(close_prices.iloc[-1] - close_prices.iloc[0]),
                    'price_change_percent': float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100),
                    'high': float(close_prices.max()),
                    'low': float(close_prices.min()),
                    'average': float(close_prices.mean())
                }
        
        return comparison
    
    def get_market_summary(self) -> Dict:
        """Get overall market summary."""
        # Get data for major indices
        major_indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
        
        summary = {}
        for index in major_indices:
            df = self.fetcher.fetch_stock_data(index)
            if df is not None and not df.empty:
                close = df['Close']
                summary[index] = {
                    'latest': float(close.iloc[-1]),
                    'change': float(close.iloc[-1] - close.iloc[-2]) if len(close) > 1 else 0,
                    'change_percent': float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0
                }
        
        return summary


# Singleton instance
stock_service = StockService()


def get_stock_service() -> StockService:
    """Get the stock service instance."""
    return stock_service
