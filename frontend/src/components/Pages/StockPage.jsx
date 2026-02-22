import React, { useState, useEffect } from 'react';
import { stockAPI } from '../../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ComposedChart } from 'recharts';

const StockPage = () => {
  const [activeCategory, setActiveCategory] = useState('stocks');
  const [activeTab, setActiveTab] = useState('chart');
  const [chartType, setChartType] = useState('area');
  const [symbol, setSymbol] = useState('AAPL');
  const [period, setPeriod] = useState('1y');
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [technicalAnalysis, setTechnicalAnalysis] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [compareSymbols, setCompareSymbols] = useState([]);
  const [predictionMethod, setPredictionMethod] = useState('lstm');
  const [predictionDays, setPredictionDays] = useState(7);
  const [predictionPeriod, setPredictionPeriod] = useState('1mo');
  const [error, setError] = useState(null);

  // Expanded options from backend
  const stocks = {
    tech: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'IBM', 'CRM', 'ADBE', 'NFLX', 'PYPL'],
    finance: ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'AXP', 'V', 'MA', 'SCHW', 'COF', 'USB', 'PNC', 'TFC'],
    energy: ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'DVN', 'HES', 'FANG', 'PXD', 'APC'],
    healthcare: ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'ISRG', 'MDT', 'SYK'],
    industrial: ['BA', 'CAT', 'HON', 'GE', 'MMM', 'UPS', 'RTX', 'LMT', 'DE', 'FDX', 'EMR', 'ITW', 'ETN', 'CMI', 'ROK']
  };

  const currencies = {
    major: ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'CHF=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'],
    emerging: ['MXN=X', 'BRL=X', 'INR=X', 'ZAR=X', 'RUB=X', 'TRY=X', 'PLN=X', 'THB=X', 'IDR=X', 'MYR=X']
  };

  const minerals = {
    precious: ['GC=F', 'SI=F', 'PL=F', 'PA=F'],
    energy: ['CL=F', 'NG=F', 'RB=F', 'HO=F'],
    agriculture: ['ZC=F', 'ZW=F', 'ZS=F', 'KC=F', 'LE=F', 'HE=F']
  };

  const stockCategories = Object.keys(stocks);
  const currencyCategories = Object.keys(currencies);
  const mineralCategories = Object.keys(minerals);
  const [stockCategory, setStockCategory] = useState('tech');
  const [currencyCategory, setCurrencyCategory] = useState('major');
  const [mineralCategory, setMineralCategory] = useState('precious');

  const periods = [
    { value: '1mo', label: '1 Month' },
    { value: '3mo', label: '3 Months' },
    { value: '6mo', label: '6 Months' },
    { value: '1y', label: '1 Year' },
    { value: '2y', label: '2 Years' },
    { value: '5y', label: '5 Years' },
  ];

  const predictionMethods = [
    { value: 'lr', label: 'Linear Regression' },
    { value: 'sma', label: 'Simple Moving Avg' },
    { value: 'lstm', label: 'LSTM AI (with confidence)' }
  ];

  // Prediction periods for LSTM: 7 days, 1 month, 6 months
  const predictionPeriods = [
    { value: '7d', label: '7 Days' },
    { value: '1mo', label: '1 Month' },
    { value: '6mo', label: '6 Months' }
  ];

  useEffect(() => {
    loadData();
  }, [symbol, period, activeCategory]);

  useEffect(() => {
    if (activeTab === 'compare' && compareSymbols.length > 0) {
      loadComparisonData();
    }
  }, [activeTab, compareSymbols, period]);

  useEffect(() => {
    if (activeTab === 'predict' && stockData) {
      loadTechnicalAnalysis();
    }
  }, [activeTab, stockData, symbol]);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    setStockData(null);
    setPrediction(null);
    setTechnicalAnalysis(null);

    try {
      let data;
      if (activeCategory === 'stocks') {
        const response = await stockAPI.getStockData(symbol, period);
        data = response.data;
      } else if (activeCategory === 'currencies') {
        const response = await stockAPI.getCurrencyData(symbol.replace('=X', ''), period);
        data = response.data;
      } else if (activeCategory === 'minerals') {
        const response = await stockAPI.getMineralData(symbol.replace('=F', ''), period);
        data = response.data;
      }

      if (data && data.data) {
        const chartData = data.data.map((item, index) => ({
          date: data.dates?.[index] || `Day ${index + 1}`,
          open: parseFloat(item.Open || item.open || item.Close || 0),
          high: parseFloat(item.High || item.high || item.Close || 0),
          low: parseFloat(item.Low || item.low || item.Close || 0),
          close: parseFloat(item.Close || item.close || 0),
          volume: parseInt(item.Volume || item.volume || 0),
        }));

        setStockData({ ...data, chartData });
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

  const loadTechnicalAnalysis = async () => {
    try {
      const response = await stockAPI.getTechnicalAnalysis(symbol);
      if (response.data && !response.data.error) {
        setTechnicalAnalysis(response.data);
      }
    } catch (err) {
      console.error('Error loading technical analysis:', err);
    }
  };

  const loadComparisonData = async () => {
    try {
      const response = await stockAPI.compareStocks(compareSymbols, period);
      if (response.data) {
        setComparisonData(response.data);
      }
    } catch (err) {
      console.error('Error loading comparison:', err);
    }
  };

  const handlePredict = async () => {
    if (!stockData) return;
    try {
      setLoading(true);
      // Convert prediction period to n_days for API
      const nDays = predictionPeriod === '6mo' ? 180 : predictionPeriod === '1mo' ? 30 : 7;
      const response = await stockAPI.predictStock(symbol, predictionMethod, nDays, predictionPeriod);
      setPrediction(response.data);
    } catch (err) {
      console.error('Error predicting:', err);
      setError('Prediction failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getCurrentSymbolList = () => {
    switch (activeCategory) {
      case 'stocks': return stocks[stockCategory] || stocks.tech;
      case 'currencies': return currencies[currencyCategory] || currencies.major;
      case 'minerals': return minerals[mineralCategory] || minerals.precious;
      default: return stocks.tech;
    }
  };

  const handleCategoryChange = (category) => {
    setActiveCategory(category);
    if (category === 'stocks') setSymbol(stocks[stockCategory][0]);
    else if (category === 'currencies') setSymbol(currencies[currencyCategory][0]);
    else setSymbol(minerals[mineralCategory][0]);
  };

  const handleSymbolChange = (newSymbol) => {
    setSymbol(newSymbol);
    setPrediction(null);
  };

  const toggleCompareSymbol = (sym) => {
    if (compareSymbols.includes(sym)) {
      setCompareSymbols(compareSymbols.filter(s => s !== sym));
    } else if (compareSymbols.length < 5) {
      setCompareSymbols([...compareSymbols, sym]);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
  };

  const renderCandlestickChart = () => {
    if (!stockData?.chartData) return null;
    return (
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={stockData.chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} tickFormatter={(value) => `$${value.toFixed(0)}`} />
          <Tooltip formatter={(value, name) => [`$${value?.toFixed(2)}`, name]} labelFormatter={(label) => `Date: ${label}`} />
          <Legend />
          <Bar dataKey="high" fill="#4ecdc4" name="High" />
          <Bar dataKey="low" fill="#ff6b6b" name="Low" />
          <Line type="monotone" dataKey="close" stroke="#45b7d1" strokeWidth={2} dot={false} name="Close" />
        </ComposedChart>
      </ResponsiveContainer>
    );
  };

  const renderVolumeChart = () => {
    if (!stockData?.chartData) return null;
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={stockData.chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10 }} tickFormatter={formatNumber} />
          <Tooltip formatter={(value) => [formatNumber(value), 'Volume']} />
          <Bar dataKey="volume" fill="#45b7d1" name="Volume" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderPredictionChart = () => {
    if (!prediction?.predictions) return null;
    const predData = prediction.predictions.map((pred, idx) => ({
      date: prediction.dates?.[idx] || `Day ${idx + 1}`,
      predicted: pred,
      upper: prediction.upper_bound?.[idx],
      lower: prediction.lower_bound?.[idx]
    }));

    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={predData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} tickFormatter={(value) => `$${value.toFixed(0)}`} />
          <Tooltip formatter={(value) => [`$${value?.toFixed(2)}`, 'Price']} />
          <Legend />
          {prediction.upper_bound && prediction.lower_bound && (
            <>
              <Line type="monotone" dataKey="upper" stroke="#4ecdc4" strokeDasharray="5 5" strokeWidth={1} dot={false} name="Upper Bound (95%)" />
              <Line type="monotone" dataKey="lower" stroke="#4ecdc4" strokeDasharray="5 5" strokeWidth={1} dot={false} name="Lower Bound (95%)" />
            </>
          )}
          <Line type="monotone" dataKey="predicted" stroke="#ff6b6b" strokeWidth={2} dot={{ fill: '#ff6b6b' }} name="Predicted Price" />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="stock-page" style={{ padding: '20px', height: '100%', overflow: 'auto', backgroundColor: '#f5f7fa' }}>
      <h1 style={{ marginBottom: '20px', color: '#2c3e50' }}>📈 Stock Market Analysis</h1>

      {/* Category Selector */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', padding: '15px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', flexWrap: 'wrap' }}>
        <button onClick={() => handleCategoryChange('stocks')} style={{ padding: '10px 20px', backgroundColor: activeCategory === 'stocks' ? '#45b7d1' : '#e0e0e0', color: activeCategory === 'stocks' ? 'white' : '#333', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>📊 Stocks</button>
        <button onClick={() => handleCategoryChange('currencies')} style={{ padding: '10px 20px', backgroundColor: activeCategory === 'currencies' ? '#45b7d1' : '#e0e0e0', color: activeCategory === 'currencies' ? 'white' : '#333', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>💱 Currencies</button>
        <button onClick={() => handleCategoryChange('minerals')} style={{ padding: '10px 20px', backgroundColor: activeCategory === 'minerals' ? '#45b7d1' : '#e0e0e0', color: activeCategory === 'minerals' ? 'white' : '#333', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>💎 Minerals</button>

        {activeCategory === 'stocks' && (
          <select value={stockCategory} onChange={(e) => { setStockCategory(e.target.value); setSymbol(stocks[e.target.value][0]); }} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', marginLeft: '10px' }}>
            {stockCategories.map(cat => <option key={cat} value={cat}>{cat.charAt(0).toUpperCase() + cat.slice(1)}</option>)}
          </select>
        )}
        {activeCategory === 'currencies' && (
          <select value={currencyCategory} onChange={(e) => { setCurrencyCategory(e.target.value); setSymbol(currencies[e.target.value][0]); }} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', marginLeft: '10px' }}>
            {currencyCategories.map(cat => <option key={cat} value={cat}>{cat.charAt(0).toUpperCase() + cat.slice(1)}</option>)}
          </select>
        )}
        {activeCategory === 'minerals' && (
          <select value={mineralCategory} onChange={(e) => { setMineralCategory(e.target.value); setSymbol(minerals[e.target.value][0]); }} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', marginLeft: '10px' }}>
            {mineralCategories.map(cat => <option key={cat} value={cat}>{cat.charAt(0).toUpperCase() + cat.slice(1)}</option>)}
          </select>
        )}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: '15px', marginBottom: '20px', padding: '15px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '0.9rem' }}>Symbol:</label>
          <select value={symbol} onChange={(e) => handleSymbolChange(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', minWidth: '150px' }}>
            {getCurrentSymbolList().map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '0.9rem' }}>Period:</label>
          <select value={period} onChange={(e) => setPeriod(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', minWidth: '150px' }}>
            {periods.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select>
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '0.9rem' }}>Chart Type:</label>
          <select value={chartType} onChange={(e) => setChartType(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', minWidth: '150px' }}>
            <option value="area">Area Chart</option>
            <option value="candlestick">High/Low Range</option>
            <option value="line">Line Chart</option>
          </select>
        </div>
        <button onClick={loadData} disabled={loading} style={{ padding: '10px 20px', backgroundColor: loading ? '#ccc' : '#4ecdc4', color: 'white', border: 'none', borderRadius: '4px', cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 'bold' }}>
          {loading ? '⏳ Loading...' : '🔄 Refresh'}
        </button>
        <div style={{ display: 'flex', gap: '5px', marginLeft: 'auto' }}>
          <button onClick={() => setActiveTab('chart')} style={{ padding: '10px 15px', backgroundColor: activeTab === 'chart' ? '#45b7d1' : '#e0e0e0', color: activeTab === 'chart' ? 'white' : '#333', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>📈 Charts</button>
          <button onClick={() => setActiveTab('compare')} style={{ padding: '10px 15px', backgroundColor: activeTab === 'compare' ? '#45b7d1' : '#e0e0e0', color: activeTab === 'compare' ? 'white' : '#333', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>⚖️ Compare</button>
          <button onClick={() => setActiveTab('predict')} style={{ padding: '10px 15px', backgroundColor: activeTab === 'predict' ? '#45b7d1' : '#e0e0e0', color: activeTab === 'predict' ? 'white' : '#333', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>🔮 Predict</button>
        </div>
      </div>

      {/* Error Display */}
      {error && <div style={{ padding: '15px', backgroundColor: '#ffebee', color: '#c62828', borderRadius: '4px', marginBottom: '20px' }}>{error}</div>}

      {/* Stock Data Display */}
      {stockData && stockData.chartData && (
        <>
          {/* Stats Bar */}
          <div style={{ display: 'flex', gap: '20px', marginBottom: '20px', padding: '15px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: '0.8rem', color: '#666' }}>Latest Price</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#45b7d1' }}>${stockData.latest_price?.toFixed(2)}</div>
              {stockData.change_percent !== undefined && stockData.change_percent !== null && (
                <div style={{ fontSize: '0.9rem', color: stockData.change_percent >= 0 ? '#4ecdc4' : '#ff6b6b' }}>{stockData.change >= 0 ? '▲' : '▼'} {stockData.change_percent?.toFixed(2)}%</div>
              )}
            </div>
            <div><div style={{ fontSize: '0.8rem', color: '#666' }}>Open</div><div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>${stockData.open?.toFixed(2)}</div></div>
            <div><div style={{ fontSize: '0.8rem', color: '#666' }}>High</div><div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#4ecdc4' }}>${stockData.high?.toFixed(2)}</div></div>
            <div><div style={{ fontSize: '0.8rem', color: '#666' }}>Low</div><div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#ff6b6b' }}>${stockData.low?.toFixed(2)}</div></div>
            <div><div style={{ fontSize: '0.8rem', color: '#666' }}>Volume</div><div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{formatNumber(stockData.latest_volume)}</div></div>
            <div><div style={{ fontSize: '0.8rem', color: '#666' }}>Data Points</div><div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{stockData.chartData.length}</div></div>
          </div>

          {/* Chart Tab */}
          {activeTab === 'chart' && (
            <>
              <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
                <h3>{symbol} Price Chart ({chartType === 'area' ? 'Area' : chartType === 'candlestick' ? 'High/Low Range' : 'Line'})</h3>
                <div style={{ height: '400px' }}>
                  {chartType === 'area' && (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={stockData.chartData}>
                        <defs>
                          <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#45b7d1" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="#45b7d1" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                        <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} tickFormatter={(value) => `$${value.toFixed(0)}`} />
                        <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Price']} labelFormatter={(label) => `Date: ${label}`} />
                        <Legend />
                        <Area type="monotone" dataKey="close" stroke="#45b7d1" fillOpacity={1} fill="url(#colorClose)" name="Close Price" />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                  {chartType === 'candlestick' && renderCandlestickChart()}
                  {chartType === 'line' && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={stockData.chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                        <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} tickFormatter={(value) => `$${value.toFixed(0)}`} />
                        <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Price']} />
                        <Legend />
                        <Line type="monotone" dataKey="close" stroke="#45b7d1" strokeWidth={2} dot={false} name="Close Price" />
                        <Line type="monotone" dataKey="open" stroke="#9b59b6" strokeWidth={1} dot={false} name="Open Price" />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
                <h3>📊 Volume</h3>
                <div style={{ height: '200px' }}>{renderVolumeChart()}</div>
              </div>
            </>
          )}

          {/* Compare Tab */}
          {activeTab === 'compare' && (
            <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
              <h3>⚖️ Stock Comparison</h3>
              <p style={{ color: '#666', marginBottom: '15px' }}>Select up to 5 stocks to compare:</p>
              <div style={{ marginBottom: '20px' }}>
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  {stocks.tech.slice(0, 8).map(s => (
                    <button key={s} onClick={() => toggleCompareSymbol(s)} style={{ padding: '8px 15px', backgroundColor: compareSymbols.includes(s) ? '#45b7d1' : '#e0e0e0', color: compareSymbols.includes(s) ? 'white' : '#333', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '0.85rem' }}>
                      {s} {compareSymbols.includes(s) && '✓'}
                    </button>
                  ))}
                </div>
              </div>

              {comparisonData && Object.keys(comparisonData).length > 0 && (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f5f5f5' }}>
                      <th style={{ padding: '10px', textAlign: 'left' }}>Symbol</th>
                      <th style={{ padding: '10px', textAlign: 'right' }}>Price</th>
                      <th style={{ padding: '10px', textAlign: 'right' }}>Change</th>
                      <th style={{ padding: '10px', textAlign: 'right' }}>Change %</th>
                      <th style={{ padding: '10px', textAlign: 'right' }}>High</th>
                      <th style={{ padding: '10px', textAlign: 'right' }}>Low</th>
                      <th style={{ padding: '10px', textAlign: 'right' }}>Average</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(comparisonData).map(([sym, data]) => (
                      <tr key={sym} style={{ borderBottom: '1px solid #eee' }}>
                        <td style={{ padding: '10px', fontWeight: 'bold' }}>{sym}</td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>${data.latest_price?.toFixed(2)}</td>
                        <td style={{ padding: '10px', textAlign: 'right', color: data.price_change >= 0 ? '#4ecdc4' : '#ff6b6b' }}>{data.price_change >= 0 ? '+' : ''}{data.price_change?.toFixed(2)}</td>
                        <td style={{ padding: '10px', textAlign: 'right', color: data.price_change_percent >= 0 ? '#4ecdc4' : '#ff6b6b' }}>{data.price_change_percent >= 0 ? '+' : ''}{data.price_change_percent?.toFixed(2)}%</td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>${data.high?.toFixed(2)}</td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>${data.low?.toFixed(2)}</td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>${data.average?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {!comparisonData && compareSymbols.length > 0 && <div style={{ textAlign: 'center', padding: '30px', color: '#666' }}>Click "Refresh" to load comparison data</div>}
            </div>
          )}

          {/* Predict Tab */}
          {activeTab === 'predict' && (
            <>
              {technicalAnalysis?.indicators?.latest && (
                <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
                  <h3>📊 Technical Indicators</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px', marginTop: '15px' }}>
                    {Object.entries(technicalAnalysis.indicators.latest).map(([key, value]) => (
                      <div key={key} style={{ padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#666', textTransform: 'uppercase' }}>{key.replace('_', ' ')}</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#2c3e50' }}>{value !== null ? (typeof value === 'number' ? value.toFixed(2) : value) : 'N/A'}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
                <h3>🔮 Price Prediction</h3>
                <div style={{ display: 'flex', gap: '15px', alignItems: 'flex-end', marginTop: '15px', flexWrap: 'wrap' }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '0.9rem' }}>Prediction Method:</label>
                    <select value={predictionMethod} onChange={(e) => setPredictionMethod(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', minWidth: '180px' }}>
                      {predictionMethods.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', fontSize: '0.9rem' }}>Prediction Period:</label>
                    <select value={predictionPeriod} onChange={(e) => setPredictionPeriod(e.target.value)} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ddd', minWidth: '150px' }}>
                      {predictionPeriods.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                    </select>
                  </div>
                  <button onClick={handlePredict} disabled={!stockData || loading} style={{ padding: '10px 25px', backgroundColor: !stockData || loading ? '#ccc' : '#ff6b6b', color: 'white', border: 'none', borderRadius: '4px', cursor: !stockData || loading ? 'not-allowed' : 'pointer', fontWeight: 'bold', fontSize: '1rem' }}>
                    {loading ? '⏳ Predicting...' : `🔮 Predict ${predictionPeriods.find(p => p.value === predictionPeriod)?.label || '7 Days'}`}
                  </button>
                </div>
              </div>

              {prediction && (
                <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
                  <h3>📈 Prediction Results ({prediction.method?.toUpperCase()})</h3>
                  <div style={{ height: '350px', marginTop: '15px' }}>{renderPredictionChart()}</div>
                  
                  <div style={{ marginTop: '20px' }}>
                    <h4>Daily Predictions</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px' }}>
                      <thead>
                        <tr style={{ backgroundColor: '#f5f5f5' }}>
                          <th style={{ padding: '10px', textAlign: 'left' }}>Date</th>
                          <th style={{ padding: '10px', textAlign: 'right' }}>Predicted Price</th>
                          {prediction.upper_bound && <th style={{ padding: '10px', textAlign: 'right' }}>Lower Bound</th>}
                          {prediction.upper_bound && <th style={{ padding: '10px', textAlign: 'right' }}>Upper Bound</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {prediction.predictions?.map((pred, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                            <td style={{ padding: '10px' }}>{prediction.dates?.[idx]}</td>
                            <td style={{ padding: '10px', textAlign: 'right', fontWeight: 'bold' }}>${pred?.toFixed(2)}</td>
                            {prediction.upper_bound && <td style={{ padding: '10px', textAlign: 'right', color: '#ff6b6b' }}>${prediction.lower_bound?.[idx]?.toFixed(2)}</td>}
                            {prediction.upper_bound && <td style={{ padding: '10px', textAlign: 'right', color: '#4ecdc4' }}>${prediction.upper_bound?.[idx]?.toFixed(2)}</td>}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* Loading State */}
      {loading && <div style={{ textAlign: 'center', padding: '50px' }}><div style={{ fontSize: '2rem' }}>⏳</div><p>Loading {symbol} data...</p></div>}

      {/* No Data State */}
      {!loading && !stockData && !error && <div style={{ textAlign: 'center', padding: '50px', color: '#666' }}><div style={{ fontSize: '3rem' }}>📈</div><h3>Select a symbol to view data</h3><p>Choose a stock, currency, or mineral from the dropdown above</p></div>}
    </div>
  );
};

export default StockPage;
