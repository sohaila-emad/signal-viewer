import React, { useState, useEffect, useMemo, useRef } from 'react';

const RecurrenceViewer = ({ data }) => {
  // State management for user-controlled parameters 
  const [channelX, setChannelX] = useState(0);
  const [channelY, setChannelY] = useState(1);
  const [timeEnd, setTimeEnd] = useState(500); 
  const [colorMap, setColorMap] = useState('thermal'); // Default thermal intensity map 
  
  const canvasRef = useRef(null);
  const dimensions = { width: 500, height: 500 };

  // Normalize data and names from the input source [cite: 3, 9]
  const { channels, names } = useMemo(() => {
    if (!data || !data.data || data.data.length === 0) return { channels: [], names: [] };
    return {
      channels: data.data,
      names: data.channel_names || data.data.map((_, i) => `Ch ${i}`),
    };
  }, [data]);

  // Color Mapping Logic: Satisfies the requirement for 2D map intensity representation 
  const getIntensityColor = (factor, map) => {
    if (map === 'spectral') {
      return `hsl(${240 - factor * 240}, 100%, 45%)`; // Blue to Red transition
    }
    if (map === 'thermal') {
      // Classic medical thermal map: Black -> Orange -> Yellow
      return `rgb(${factor * 255}, ${Math.max(0, factor * 255 - 128) * 2}, 0)`;
    }
    return '#2c3e50'; // Standard clinical dark blue/gray
  };

  // Main Drawing Effect: Renders the cumulative scatter plot 
  useEffect(() => {
    if (channels.length === 0 || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const chX = channels[channelX];
    const chY = channels[channelY] !== undefined ? channels[channelY] : chX;

    // Background matching standard website light-mode
    ctx.fillStyle = '#ffffff'; 
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    // Dynamic scaling for ECG/EEG signals [cite: 3]
    const getBounds = (arr) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return { min, range: (max - min) || 1 };
    };

    const boundsX = getBounds(chX);
    const boundsY = getBounds(chY);

    // Cumulative plotting loop: draws every point up to the specified time [cite: 13, 14]
    for (let t = 0; t < timeEnd; t++) {
      if (t >= chX.length || t >= chY.length) break;

      const x = ((chX[t] - boundsX.min) / boundsX.range) * dimensions.width;
      const y = dimensions.height - (((chY[t] - boundsY.min) / boundsY.range) * dimensions.height);

      const intensity = t / timeEnd;
      ctx.fillStyle = getIntensityColor(intensity, colorMap);
      
      // Points plotted as cumulative scatter 
      ctx.fillRect(x, y, 1.8, 1.8);
    }
  }, [channelX, channelY, timeEnd, colorMap, channels]);

  // Legend component explaining color intensity 
  const renderLegend = () => {
    const gradients = {
      spectral: 'linear-gradient(to right, #0000ff, #00ff00, #ff0000)',
      thermal: 'linear-gradient(to right, #000, #f60, #ff0)',
      monochrome: 'linear-gradient(to right, #2c3e50, #2c3e50)'
    };

    return (
      <div style={{ marginTop: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #dee2e6' }}>
        <div style={{ fontSize: '0.85rem', marginBottom: '8px', fontWeight: '600', color: '#495057' }}>
          Intensity Map: {colorMap.toUpperCase()}
        </div>
        <div style={{ height: '10px', width: '100%', background: gradients[colorMap], borderRadius: '5px' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', marginTop: '6px', color: '#6c757d' }}>
          <span>Past Data (t=0)</span>
          <span>Current (t={timeEnd})</span>
        </div>
      </div>
    );
  };

  if (channels.length === 0) return <div style={{padding: '20px'}}>No signal data loaded.</div>;

  return (
    <div style={{ 
      display: 'flex', 
      gap: '20px', 
      padding: '20px', 
      backgroundColor: '#f1f3f5', // Standard website background
      minHeight: '100vh', 
      fontFamily: 'system-ui, -apple-system, sans-serif' 
    }}>
      
      {/* Sidebar Controls [cite: 8, 9, 15] */}
      <div style={{ width: '280px', background: '#ffffff', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', height: 'fit-content' }}>
        <h3 style={{ marginTop: 0, color: '#212529', borderBottom: '2px solid #e9ecef', paddingBottom: '10px' }}>Settings</h3>

        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 'bold', color: '#495057' }}>Channel X (Horizontal)</label>
          <select value={channelX} onChange={e => setChannelX(Number(e.target.value))} style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ced4da' }}>
            {names.map((n, i) => <option key={i} value={i}>{n}</option>)}
          </select>
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 'bold', color: '#495057' }}>Channel Y (Vertical)</label>
          <select value={channelY} onChange={e => setChannelY(Number(e.target.value))} style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ced4da' }}>
            {names.map((n, i) => <option key={i} value={i}>{n}</option>)}
          </select>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 'bold', color: '#495057' }}>Accumulated Time: {timeEnd}</label>
          <input type="range" min="10" max={channels[0]?.length || 1000} value={timeEnd} onChange={e => setTimeEnd(Number(e.target.value))} style={{ width: '100%' }} />
        </div>

        <div style={{ marginBottom: '10px' }}>
          <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 'bold', color: '#495057' }}>Color Map Selection</label>
          <select value={colorMap} onChange={e => setColorMap(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ced4da' }}>
            <option value="thermal">Thermal (Recommended)</option>
            <option value="spectral">Spectral (Time Flow)</option>
            <option value="monochrome">Clinical (Static)</option>
          </select>
        </div>

        {renderLegend()}
      </div>

      {/* Main Plot Area [cite: 9] */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', background: '#ffffff', borderRadius: '12px', padding: '30px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
        <div style={{ marginBottom: '20px', fontSize: '1.1rem', fontWeight: '600', color: '#212529' }}>
          Reoccurrence: {names[channelX]} vs {names[channelY]}
        </div>
        
        <div style={{ border: '1px solid #dee2e6', borderRadius: '4px', background: '#fff' }}>
          <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} />
        </div>
        
        <div style={{ marginTop: '20px', color: '#adb5bd', fontSize: '0.75rem' }}>
          Visualizing relationship between {names[channelX]} and {names[channelY]} 
        </div>
      </div>
    </div>
  );
};

export default RecurrenceViewer;
