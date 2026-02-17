import React, { useState, useEffect } from 'react';
import { stockAPI } from '../../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

const StockPage = () => {
  const [activeCategory, setActiveCategory] = useState('stocks'); // 'stocks', 'currencies', 'minerals'
  const [symbol, setSymbol] = useState('AAPL');
  const [period, setPeriod] = useState('1y');
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  
  // Available options
  const stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD'];
  const currencies = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'];
  const minerals = ['GOLD', 'SILVER', 'PLATINUM', 'PALLADIUM', 'COPPER'];
  
  const periods = [
    { value: '1mo', label: '1 Month' },
    { value: '3mo', label: '3 Months' },
    { value: '6mo', label: '6 Months' },
    { value: '1y', label: '1 Year' },
    { value: '2y', label: '2 Years' },
    { value: '5y', label: '5 Years' },
  ];

  useEffect(() => {
    loadData();
  }, [symbol, period, activeCategory]);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    setStockData(null);
    setPrediction(null);
    
    try {
      let data;
      if (activeCategory === 'stocks') {
        const response = await stockAPI.getStockData(symbol, period);
        data = response.data;
      } else if (activeCategory === 'currencies') {
        const response = await stockAPI.getCurrencyData(symbol, period);
        data = response.data;
      } else if (activeCategory === 'minerals') {
        const response = await stockAPI.getMineralData(symbol, period);
        data = response.data;
      }
      
      if (data && data.data) {
        // Transform data for chart
        const chartData = data.data.map((item, index) => ({
          date: data.dates?.[index] || `Day ${index + 1}`,
          open: item.Open || item.open || item.Close || 0,
          high: item.High || item.high || item.Close || 0,
          low: item.Low || item.low || item.Close || 0,
          close: item.Close || item.close || 0,
          volume: item.Volume || item.volume || 0,
        }));
        
        setStockData({
          ...data,
          chartData
        });
      } else {
        setError('No data available');
      }
    } catch (err) {
      console.error('Error loading data:', err);
      setError('Failed to load data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!stockData) return;
    
    try {
      const response = await stockAPI.predictStock(symbol, 'sma', 7);
      setPrediction(response.data);
    } catch (err) {
      console.error('Error predicting:', err);
    }
  };

  const getCurrentSymbolList = () => {
    switch (activeCategory) {
      case 'stocks': return stocks;
      case 'currencies': return currencies;
      case 'minerals': return minerals;
      default: return stocks;
    }
  };

  return (
    <div className="stock-page" style={{ padding: '20px', height: '100%', overflow: 'auto' }}>
      <h1 style={{ marginBottom: '20px' }}>📈 Stock Market Analysis</h1>
      
      {/* Category Selector */}
      <div style={{ 
        display: 'flex', 
        gap: '10px', 
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px'
      }}>
        <button
          onClick={() => { setActiveCategory('stocks'); setSymbol(stocks[0]); }}
          style={{
            padding: '10px 20px',
            backgroundColor: activeCategory === 'stocks' ? '#45b7d1' : '#ddd',
            color: activeCategory === 'stocks' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          📊 Stocks
        </button>
        <button
          onClick={() => { setActiveCategory('currencies'); setSymbol(currencies[0]); }}
          style={{
            padding: '10px 20px',
            backgroundColor: activeCategory === 'currencies' ? '#45b7d1' : '#ddd',
            color: activeCategory === 'currencies' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          💱 Currencies
        </button>
        <button
          onClick={() => { setActiveCategory('minerals'); setSymbol(minerals[0]); }}
          style={{
            padding: '10px 20px',
            backgroundColor: activeCategory === 'minerals' ? '#45b7d1' : '#ddd',
            color: activeCategory === 'minerals' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          💎 Minerals
        </button>
      </div>

      {/* Controls */}
      <div style={{ 
        display: 'flex', 
        gap: '15px', 
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: 'white',
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        flexWrap: 'wrap'
      }}>
        <div>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Symbol:
          </label>
          <select 
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', minWidth: '150px' }}
          >
            {getCurrentSymbolList().map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Period:
          </label>
          <select 
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', minWidth: '150px' }}
          >
            {periods.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button
            onClick={loadData}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: loading ? '#ccc' : '#4ecdc4',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: 'bold'
            }}
          >
            {loading ? '⏳ Loading...' : '🔄 Refresh'}
          </button>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button
            onClick={handlePredict}
            disabled={!stockData || loading}
            style={{
              padding: '10px 20px',
              backgroundColor: !stockData || loading ? '#ccc' : '#ff6b6b',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: !stockData || loading ? 'not-allowed' : 'pointer',
              fontWeight: 'bold'
            }}
          >
            🔮 Predict
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#ffebee', 
          color: '#c62828',
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          {error}
        </div>
      )}

      {/* Stock Data Display */}
      {stockData && stockData.chartData && (
        <>
          {/* Stats Bar */}
          <div style={{ 
            display: 'flex', 
            gap: '20px', 
            marginBottom: '20px',
            padding: '15px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            flexWrap: 'wrap'
          }}>
            <div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Latest Price</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#45b7d1' }}>
                ${stockData.latest_price?.toFixed(2)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Volume</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
                {(stockData.latest_volume / 1000000).toFixed(2)}M
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Data Points</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
                {stockData.chartData.length}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Period</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
                {period}
              </div>
            </div>
          </div>

          {/* Price Chart */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            marginBottom: '20px'
          }}>
            <h3>{symbol} Price Chart</h3>
            <div style={{ height: '400px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stockData.chartData}>
                  <defs>
                    <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#45b7d1" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#45b7d1" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 10 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis 
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 10 }}
                    tickFormatter={(value) => `$${value.toFixed(0)}`}
                  />
                  <Tooltip 
                    formatter={(value) => [`$${value.toFixed(2)}`, 'Price']}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Legend />
                  <Area 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#45b7d1" 
                    fillOpacity={1} 
                    fill="url(#colorClose)" 
                    name="Close Price"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* High/Low Chart */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            marginBottom: '20px'
          }}>
            <h3>High/Low Range</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stockData.chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 10 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis 
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 10 }}
                    tickFormatter={(value) => `$${value.toFixed(0)}`}
                  />
                  <Tooltip 
                    formatter={(value) => [`$${value.toFixed(2)}`, 'Price']}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="high" 
                    stroke="#4ecdc4" 
                    dot={false}
                    name="High"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="low" 
                    stroke="#ff6b6b" 
                    dot={false}
                    name="Low"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Prediction Results */}
          {prediction && (
            <div style={{ 
              padding: '20px', 
              backgroundColor: 'white', 
              borderRadius: '8px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              marginBottom: '20px'
            }}>
              <h3>🔮 Price Prediction (Next 7 Days)</h3>
              <div style={{ height: '300px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={prediction.predictions || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                    <YAxis 
                      tickFormatter={(value) => `$${value.toFixed(0)}`}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Predicted Price']} />
                    <Line 
                      type="monotone" 
                      dataKey="predicted" 
                      stroke="#ff6b6b" 
                      strokeWidth={2}
                      dot={{ fill: '#ff6b6b' }}
                      name="Predicted Price"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {prediction.confidence && (
                <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#e3f2fd', borderRadius: '4px' }}>
                  <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(1)}%
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Loading State */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <div style={{ fontSize: '2rem' }}>⏳</div>
          <p>Loading {symbol} data...</p>
        </div>
      )}

      {/* No Data State */}
      {!loading && !stockData && !error && (
        <div style={{ textAlign: 'center', padding: '50px', color: '#666' }}>
          <div style={{ fontSize: '3rem' }}>📈</div>
          <h3>Select a symbol to view data</h3>
          <p>Choose a stock, currency, or mineral from the dropdown above</p>
        </div>
      )}
    </div>
  );
};

export default StockPage;
