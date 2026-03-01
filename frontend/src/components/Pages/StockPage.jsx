import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';

const API_BASE = 'http://localhost:5000/api/stocks';

// ── colour palette (matches existing app theme) ───────────────────────────────
const COLORS = {
  actual:    '#000000',
  train:     '#45b7d1',
  test:      '#4ecdc4',
  recursive: '#ff6b6b',
  forecast:  '#ff6b6b',
  split:     '#aaaaaa',
};

// ── tiny helpers ──────────────────────────────────────────────────────────────
const fmt = (v) => (v !== undefined && v !== null ? `$${Number(v).toFixed(2)}` : '');

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      backgroundColor: 'white', border: '1px solid #e0e0e0',
      borderRadius: 6, padding: '10px 14px', fontSize: 12,
    }}>
      <p style={{ margin: '0 0 6px', fontWeight: 'bold', color: '#2c3e50' }}>{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ margin: '2px 0', color: p.color }}>
          {p.name}: {fmt(p.value)}
        </p>
      ))}
    </div>
  );
};

// ── main component ────────────────────────────────────────────────────────────
const StockPage = () => {
  const [catalogue, setCatalogue]   = useState(null);   // asset menu from API
  const [category, setCategory]     = useState('stocks');
  const [symbol, setSymbol]         = useState('AAPL');
  const [horizon, setHorizon]       = useState('7d');

  const [graph1, setGraph1]         = useState(null);
  const [graph2, setGraph2]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);

  // ── fetch asset catalogue once on mount ────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/assets`)
      .then((r) => r.json())
      .then((data) => {
        setCatalogue(data);
        // set initial symbol to first in stocks
        const firstSymbol = Object.keys(data?.stocks ?? {})[0];
        if (firstSymbol) setSymbol(firstSymbol);
      })
      .catch(() => setError('Could not load asset list from API.'));
  }, []);

  // ── fetch graph data whenever symbol changes ────────────────────────────────
  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    setError(null);
    setGraph1(null);
    setGraph2(null);

    Promise.all([
      fetch(`${API_BASE}/graph1/${symbol}`).then((r) => r.json()),
      fetch(`${API_BASE}/graph2/${symbol}?horizon=${horizon}`).then((r) => r.json()),
    ])
      .then(([g1, g2]) => {
        if (g1.error) throw new Error(g1.error);
        if (g2.error) throw new Error(g2.error);
        setGraph1(g1);
        setGraph2(g2);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [symbol]);

  // ── re-fetch graph2 when horizon changes (graph1 stays the same) ───────────
  useEffect(() => {
    if (!symbol || !graph1) return;
    fetch(`${API_BASE}/graph2/${symbol}?horizon=${horizon}`)
      .then((r) => r.json())
      .then((g2) => {
        if (g2.error) throw new Error(g2.error);
        setGraph2(g2);
      })
      .catch((e) => setError(e.message));
  }, [horizon]);

  // ── build recharts data arrays ─────────────────────────────────────────────

  const buildGraph1Data = () => {
    if (!graph1 || !graph1.graph1_available) return [];

    const trainLen = graph1.train_dates.length;
    const testLen  = graph1.test_dates.length;

    const trainRows = graph1.train_dates.map((date, i) => ({
      date,
      actual:    graph1.actual_train[i],
      train:     graph1.train_predictions[i],
      test:      null,
      recursive: null,
    }));

    const testRows = graph1.test_dates.map((date, i) => ({
      date,
      actual:    graph1.actual_test[i],
      train:     null,
      test:      graph1.test_predictions[i],
      recursive: graph1.recursive_predictions[i],
    }));

    return [...trainRows, ...testRows];
  };

  const buildGraph2Data = () => {
    if (!graph2) return [];

    // anchor point — last known price so the line starts connected
    const anchor = {
      date:     graph2.last_actual_date,
      forecast: graph2.last_actual_price,
    };

    const forecastRows = graph2.forecast_dates.map((date, i) => ({
      date,
      forecast: graph2.forecast_prices[i],
    }));

    return [anchor, ...forecastRows];
  };

  // ── category change ────────────────────────────────────────────────────────
  const handleCategoryChange = (cat) => {
    setCategory(cat);
    if (!catalogue) return;
    const firstSymbol = Object.keys(catalogue[cat] ?? {})[0];
    if (firstSymbol) setSymbol(firstSymbol);
  };

  const currentSymbols = catalogue?.[category] ?? {};

  // ── tick reducer — show ~6 evenly-spaced labels ────────────────────────────
  const tickInterval = (dataLen) => Math.max(1, Math.floor(dataLen / 6));

  const g1Data = buildGraph1Data();
  const g2Data = buildGraph2Data();
  const splitDate = graph1?.test_dates?.[0];

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{
      padding: 24, minHeight: '100%',
      backgroundColor: '#f5f7fa', fontFamily: 'sans-serif',
    }}>
      <h1 style={{ marginBottom: 24, color: '#2c3e50' }}>📈 Stock Market Analysis</h1>

      {/* ── CONTROLS ── */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end',
        padding: 16, backgroundColor: 'white', borderRadius: 8,
        boxShadow: '0 2px 4px rgba(0,0,0,0.08)', marginBottom: 24,
      }}>

        {/* category buttons */}
        <div style={{ display: 'flex', gap: 8 }}>
          {['stocks', 'commodities', 'currencies'].map((cat) => (
            <button
              key={cat}
              onClick={() => handleCategoryChange(cat)}
              style={{
                padding: '9px 18px', border: 'none', borderRadius: 4,
                cursor: 'pointer', fontWeight: 'bold', fontSize: 13,
                backgroundColor: category === cat ? '#45b7d1' : '#e0e0e0',
                color:           category === cat ? 'white'   : '#333',
              }}
            >
              { cat === 'stocks'      ? '📊 Stocks'
              : cat === 'commodities' ? '🪙 Commodities'
              :                         '💱 Currencies' }
            </button>
          ))}
        </div>

        {/* symbol selector */}
        <div>
          <label style={{ display: 'block', fontSize: 12, fontWeight: 'bold', marginBottom: 4 }}>
            Asset
          </label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: 4, border: '1px solid #ddd', minWidth: 160 }}
          >
            {Object.entries(currentSymbols).map(([sym, label]) => (
              <option key={sym} value={sym}>{sym} — {label}</option>
            ))}
          </select>
        </div>

        {/* horizon selector */}
        <div>
          <label style={{ display: 'block', fontSize: 12, fontWeight: 'bold', marginBottom: 4 }}>
            Forecast Horizon
          </label>
          <div style={{ display: 'flex', gap: 6 }}>
            {[['7d', 'Next Week'], ['30d', 'Next Month']].map(([val, lbl]) => (
              <button
                key={val}
                onClick={() => setHorizon(val)}
                style={{
                  padding: '8px 16px', border: 'none', borderRadius: 4,
                  cursor: 'pointer', fontWeight: 'bold', fontSize: 13,
                  backgroundColor: horizon === val ? '#ff6b6b' : '#e0e0e0',
                  color:           horizon === val ? 'white'   : '#333',
                }}
              >
                {lbl}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── ERROR ── */}
      {error && (
        <div style={{
          padding: 14, marginBottom: 20, borderRadius: 6,
          backgroundColor: '#ffebee', color: '#c62828', fontSize: 14,
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* ── LOADING ── */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 60, color: '#666' }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>⏳</div>
          <p>Loading {symbol} data…</p>
        </div>
      )}

      

      {/* ── GRAPH 1 — full history ── */}
      {!loading && graph1 && graph1.graph1_available && (
        <div style={{
          backgroundColor: 'white', borderRadius: 8, padding: 24,
          boxShadow: '0 2px 4px rgba(0,0,0,0.08)', marginBottom: 24,
        }}>
          <h3 style={{ margin: '0 0 6px', color: '#2c3e50' }}>
            {symbol} — Full History
          </h3>
          <p style={{ margin: '0 0 20px', fontSize: 13, color: '#888' }}>
            Training predictions (blue) · Test 1-step (teal) · Recursive (red) · Actual (black)
          </p>

          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={g1Data} margin={{ top: 4, right: 20, left: 10, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11 }}
                interval={tickInterval(g1Data.length)}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
                domain={['auto', 'auto']}
                width={70}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 13 }} />

              {/* vertical line at train/test boundary */}
              {splitDate && (
                <ReferenceLine
                  x={splitDate}
                  stroke={COLORS.split}
                  strokeDasharray="4 4"
                  label={{ value: 'Train / Test', fill: '#999', fontSize: 11, position: 'insideTopRight' }}
                />
              )}

              <Line dataKey="actual"    name="Actual"              stroke={COLORS.actual}    strokeWidth={1.5} dot={false} connectNulls={false} />
              <Line dataKey="train"     name="Train Prediction"    stroke={COLORS.train}     strokeWidth={1.5} dot={false} connectNulls={false} />
              <Line dataKey="test"      name="Test Prediction"     stroke={COLORS.test}      strokeWidth={1.5} dot={false} connectNulls={false} />
              <Line dataKey="recursive" name="Recursive Prediction" stroke={COLORS.recursive} strokeWidth={1.5} dot={false} strokeDasharray="5 3" connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── GRAPH 2 — future forecast ── */}
      {!loading && graph2 && (
        <div style={{
          backgroundColor: 'white', borderRadius: 8, padding: 24,
          boxShadow: '0 2px 4px rgba(0,0,0,0.08)', marginBottom: 24,
        }}>
          <h3 style={{ margin: '0 0 6px', color: '#2c3e50' }}>
            {symbol} — {horizon === '7d' ? 'Next Week' : 'Next Month'} Forecast
          </h3>
          <p style={{ margin: '0 0 20px', fontSize: 13, color: '#888' }}>
            Recursive prediction starting from the last known price on {graph2.last_actual_date}
          </p>

          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={g2Data} margin={{ top: 4, right: 20, left: 10, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
                domain={['auto', 'auto']}
                width={70}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 13 }} />

              {/* reference line at the last known price anchor */}
              <ReferenceLine
                x={graph2.last_actual_date}
                stroke={COLORS.split}
                strokeDasharray="4 4"
                label={{ value: 'Data ends', fill: '#999', fontSize: 11, position: 'insideTopRight' }}
              />

              <Line
                dataKey="forecast"
                name="Forecast"
                stroke={COLORS.forecast}
                strokeWidth={2}
                strokeDasharray="5 3"
                dot={{ fill: COLORS.forecast, r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>

          {/* forecast price table */}
          <div style={{ marginTop: 20, overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ backgroundColor: '#f8f9fa' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left',  borderBottom: '1px solid #eee' }}>Date</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #eee' }}>Forecast Price</th>
                  <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #eee' }}>Change vs Last</th>
                </tr>
              </thead>
              <tbody>
                {graph2.forecast_dates.map((date, i) => {
                  const price  = graph2.forecast_prices[i];
                  const change = price - graph2.last_actual_price;
                  const pct    = (change / graph2.last_actual_price) * 100;
                  return (
                    <tr key={date} style={{ borderBottom: '1px solid #f0f0f0' }}>
                      <td style={{ padding: '8px 12px' }}>{date}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 'bold' }}>
                        {fmt(price)}
                      </td>
                      <td style={{
                        padding: '8px 12px', textAlign: 'right',
                        color: change >= 0 ? '#4ecdc4' : '#ff6b6b',
                      }}>
                        {change >= 0 ? '+' : ''}{fmt(change)} ({pct >= 0 ? '+' : ''}{pct.toFixed(2)}%)
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default StockPage;