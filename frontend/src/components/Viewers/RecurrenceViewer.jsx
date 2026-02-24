import React, { useState, useEffect, useMemo, useRef } from 'react';

const RecurrenceViewer = ({ data }) => {
  const [channelX, setChannelX] = useState(0);
  const [channelY, setChannelY] = useState(0); // Default to 0 for self-recurrence
  const [timeStart, setTimeStart] = useState(0);
  const [timeEnd, setTimeEnd] = useState(200);
  
  const [mDim, setMDim] = useState(1);
  const [tau, setTau] = useState(1);
  const [epsilon, setEpsilon] = useState(0.1);
  
  const [colorMap, setColorMap] = useState('binary');
  const [dimensions] = useState({ width: 500, height: 500 });
  const canvasRef = useRef(null);

  const normalizedData = useMemo(() => {
    if (!data || !data.data || data.data.length === 0) return { channels: [], names: [] };
    return {
      channels: data.data,
      names: data.channel_names || data.data.map((_, i) => `Ch ${i}`),
    };
  }, [data]);

  const { channels, names } = normalizedData;

  const recurrenceMatrix = useMemo(() => {
    if (channels.length === 0 || !channels[channelX]) return null;
    const ch1 = channels[channelX];
    const ch2 = channels[channelY] !== undefined ? channels[channelY] : channels[channelX];

    const getNorm = (arr) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      const range = max - min || 1;
      return arr.map(v => (v - min) / range);
    };

    const normCh1 = getNorm(ch1);
    const normCh2 = getNorm(ch2);
    
    const start = Math.max(0, timeStart);
    const end = Math.min(normCh1.length, normCh2.length, timeEnd);
    const N = end - start;

    const validN = N - (mDim - 1) * tau;
    if (validN <= 0) return null;

    const matrix = [];
    for (let i = 0; i < validN; i++) {
      const row = [];
      for (let j = 0; j < validN; j++) {
        let sumSq = 0;
        for (let k = 0; k < mDim; k++) {
          const valX = normCh1[start + i + k * tau];
          const valY = normCh2[start + j + k * tau];
          sumSq += Math.pow(valX - valY, 2);
        }
        row.push(Math.sqrt(sumSq));
      }
      matrix.push(row);
    }
    return { matrix, size: validN };
  }, [channels, channelX, channelY, timeStart, timeEnd, mDim, tau]);

  const getColor = (dist) => {
    if (colorMap === 'binary') return dist < epsilon ? '#000000' : '#FFFFFF';
    if (dist > epsilon) return '#FFFFFF';
    const val = Math.max(0, Math.min(1, 1 - (dist / epsilon)));
    if (colorMap === 'heat') return `rgb(${val * 255}, ${val * 100}, 0)`;
    return `hsl(${280 - val * 280}, 70%, 50%)`;
  };

  useEffect(() => {
    if (!recurrenceMatrix || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { matrix, size } = recurrenceMatrix;
    const cellW = dimensions.width / size;
    const cellH = dimensions.height / size;

    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    matrix.forEach((row, i) => {
      row.forEach((dist, j) => {
        const color = getColor(dist);
        if (color !== '#FFFFFF') {
            ctx.fillStyle = color;
            ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
        }
      });
    });
  }, [recurrenceMatrix, colorMap, epsilon]);

  const renderColorLegend = () => {
    if (colorMap === 'binary') {
      return (
        <div style={{ display: 'flex', gap: '20px', fontSize: '0.85rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ width: 12, height: 12, backgroundColor: '#000' }} />
            <span>Match (&lt; {epsilon.toFixed(3)})</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{ width: 12, height: 12, backgroundColor: '#fff', border: '1px solid #ccc' }} />
            <span>Different</span>
          </div>
        </div>
      );
    }

    const grad = colorMap === 'heat' 
      ? 'linear-gradient(to right, #000, #f60, #ff0)'
      : 'linear-gradient(to right, #60c, #0c6, #f00)';

    return (
      <div style={{ width: '100%', maxWidth: '300px' }}>
        <div style={{ height: '10px', width: '100%', background: grad, borderRadius: '5px' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', marginTop: '5px' }}>
          <span>Limit ({epsilon.toFixed(2)})</span>
          <span>Identical (0.0)</span>
        </div>
      </div>
    );
  };

  if (channels.length === 0) return <div>No data provided.</div>;

  return (
    <div style={{ display: 'flex', gap: '20px', padding: '20px', backgroundColor: '#f8f9fa', minHeight: '100vh', fontFamily: 'sans-serif' }}>
      
      {/* Sidebar */}
      <div style={{ width: '300px', background: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
        <h3 style={{ margin: '0 0 20px 0', color: '#333' }}>Settings</h3>
        
        {/* CHANNEL SELECTORS - BACK IN ACTION */}
        <div style={{ marginBottom: '15px' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>Horizontal Channel (X)</label>
          <select value={channelX} onChange={e => setChannelX(Number(e.target.value))} style={{ width: '100%', padding: '8px', marginTop: '5px' }}>
            {names.map((n, i) => <option key={i} value={i}>{n}</option>)}
          </select>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>Vertical Channel (Y)</label>
          <select value={channelY} onChange={e => setChannelY(Number(e.target.value))} style={{ width: '100%', padding: '8px', marginTop: '5px' }}>
            {names.map((n, i) => <option key={i} value={i}>{n}</option>)}
          </select>
        </div>

        <hr />

        <div style={{ margin: '20px 0' }}>
            <label style={{ fontSize: '0.85rem' }}>Threshold ($\epsilon$): {epsilon.toFixed(3)}</label>
            <input type="range" min="0.001" max="0.5" step="0.001" value={epsilon} onChange={e => setEpsilon(Number(e.target.value))} style={{ width: '100%' }} />
        </div>

        <div style={{ marginBottom: '20px' }}>
            <label style={{ fontSize: '0.85rem' }}>Embedding ($m$): {mDim}</label>
            <input type="range" min="1" max="5" value={mDim} onChange={e => setMDim(Number(e.target.value))} style={{ width: '100%' }} />
        </div>

        <select value={colorMap} onChange={e => setColorMap(e.target.value)} style={{ width: '100%', padding: '8px' }}>
          <option value="binary">Binary (Dots)</option>
          <option value="viridis">Plasma (Distance)</option>
          <option value="heat">Thermal (Heat)</option>
        </select>
      </div>

      {/* Main Plot */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div style={{ marginBottom: '10px', color: '#666' }}>{names[channelX]} vs {names[channelY]}</div>
        
        <div style={{ position: 'relative', border: '1px solid #ddd', background: '#fff' }}>
          <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} />
        </div>

        <div style={{ marginTop: '20px', padding: '15px', background: 'white', borderRadius: '10px' }}>
          {renderColorLegend()}
        </div>
      </div>
    </div>
  );
};

export default RecurrenceViewer;