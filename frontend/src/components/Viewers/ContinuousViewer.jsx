import React, { useState, useEffect, useRef, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ContinuousViewer = ({ data }) => {
  const [viewport, setViewport] = useState({ start: 0, end: 500 });
  const [visibleChannels, setVisibleChannels] = useState({});
  const [playback, setPlayback] = useState({ playing: false, speed: 1 });
  const [channelColors, setChannelColors] = useState({});
  const [displayMode, setDisplayMode] = useState('overlay');
  const [offsetAmount, setOffsetAmount] = useState(3);
  const animationRef = useRef();

  // 1. DATA NORMALIZATION
  const normalized = useMemo(() => {
    if (!data || !data.data) return null;
    const signalData = data.data;
    const fs = data.fs || 250;
    const samplesCount = signalData[0]?.length || 0;
    const channelNames = data.channel_names || signalData.map((_, i) => `Channel ${i + 1}`);
    
    return { channels: signalData, totalSamples: samplesCount, channelNames, fs };
  }, [data]);

  // 2. INITIALIZATION
  const isInitialized = useRef(false);
  useEffect(() => {
    if (normalized && !isInitialized.current) {
      const initVis = {};
      const initCols = {};
      normalized.channels.forEach((_, i) => {
        initVis[i] = true;
        initCols[i] = `hsl(${(i * 137) % 360}, 70%, 50%)`;
      });
      setVisibleChannels(initVis);
      setChannelColors(initCols);
      setViewport({ start: 0, end: Math.min(normalized.fs * 2, normalized.totalSamples) });
      isInitialized.current = true;
    }
  }, [normalized]);

  useEffect(() => { isInitialized.current = false; }, [data]);

  // 3. PLAYBACK
  useEffect(() => {
    if (playback.playing && normalized) {
      let lastTimestamp = performance.now();

      const animate = (now) => {
        const deltaTime = now - lastTimestamp;
        lastTimestamp = now;
        const samplesToMove = (deltaTime / 1000) * normalized.fs * playback.speed;

        setViewport(prev => {
          const range = prev.end - prev.start;
          const nextEnd = prev.end + samplesToMove;
          if (nextEnd >= normalized.totalSamples) {
            setPlayback(p => ({ ...p, playing: false }));
            return prev;
          }
          return { start: nextEnd - range, end: nextEnd };
        });
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    }
    return () => cancelAnimationFrame(animationRef.current);
  }, [playback.playing, playback.speed, normalized]);

  // --- NAVIGATION HANDLERS ---
  
  const handleZoom = (factor) => {
    setViewport(prev => {
      const range = prev.end - prev.start;
      const center = (prev.start + prev.end) / 2;
      const newRange = Math.max(normalized.fs * 0.1, Math.min(range * factor, normalized.totalSamples));
      
      let start = center - newRange / 2;
      let end = center + newRange / 2;

      // Bound checks
      if (start < 0) { start = 0; end = newRange; }
      if (end > normalized.totalSamples) { end = normalized.totalSamples; start = end - newRange; }
      
      return { start, end };
    });
  };

  const handlePan = (direction) => {
    setViewport(prev => {
      const range = prev.end - prev.start;
      const shift = range * 0.25 * direction; // Move by 25% of current view
      let start = prev.start + shift;
      let end = prev.end + shift;

      if (start < 0) { start = 0; end = range; }
      if (end > normalized.totalSamples) { end = normalized.totalSamples; start = end - range; }

      return { start, end };
    });
  };

  // 4. CHART DATA FORMATTING
  const chartData = useMemo(() => {
    if (!normalized) return [];
    const start = Math.max(0, Math.floor(viewport.start));
    const end = Math.min(normalized.totalSamples, Math.ceil(viewport.end));
    const result = [];

    const activeIndices = Object.keys(visibleChannels)
      .filter(k => visibleChannels[k])
      .map(Number);

    for (let i = start; i < end; i++) {
      const exactTime = i / normalized.fs; 
      const point = { time: exactTime }; 
      activeIndices.forEach((chIdx, stackIdx) => {
        const val = normalized.channels[chIdx][i] || 0;
        point[`ch${chIdx}`] = val;
        point[`ch${chIdx}_offset`] = val + (stackIdx * offsetAmount);
      });
      result.push(point);
    }
    return result;
  }, [viewport, normalized, visibleChannels, offsetAmount]);

  // 5. HANDLERS
  const toggleChannel = (idx) => setVisibleChannels(p => ({ ...p, [idx]: !p[idx] }));
  const showAll = () => setVisibleChannels(Object.fromEntries(normalized.channels.map((_, i) => [i, true])));
  const hideAll = () => setVisibleChannels(Object.fromEntries(normalized.channels.map((_, i) => [i, false])));

  // 6. CUSTOM TOOLTIP
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: '#fff', border: '1px solid #ccc', padding: '10px', borderRadius: '4px', boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
          <p style={{ margin: '0 0 5px 0', borderBottom: '1px solid #eee' }}>
            <strong>Time:</strong> {Number(label).toFixed(3)} s
          </p>
          {payload.map((entry, i) => (
            <p key={i} style={{ color: entry.color, margin: '2px 0', fontSize: '12px' }}>
              {entry.name}: {entry.payload[entry.dataKey.replace('_offset', '')].toFixed(3)} mV
            </p>
          ))}
        </div>
      );
    }
    return null;
  }

  if (!normalized) return <div style={{ padding: '20px' }}>Loading signal data...</div>;

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#f0f2f5', fontFamily: 'sans-serif' }}>
      
      {/* HEADER */}
      <div style={{ padding: '15px', background: '#fff', borderBottom: '1px solid #ddd', display: 'flex', gap: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
        <button onClick={() => setPlayback(p => ({ ...p, playing: !p.playing }))} style={{ ...btnStyle, background: playback.playing ? '#f44336' : '#4CAF50' }}>
          {playback.playing ? 'Pause' : 'Play'}
        </button>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{fontSize: '14px'}}>Speed: {playback.speed}x</span>
          <input type="range" min="0.1" max="10" step="0.1" value={playback.speed} onChange={e => setPlayback(p => ({ ...p, speed: parseFloat(e.target.value) }))} />
        </div>

        {/* NAVIGATION BUTTONS */}
        <div style={{ display: 'flex', gap: '5px', alignItems: 'center', background: '#eee', padding: '5px', borderRadius: '6px' }}>
          <span style={{fontSize: '12px', fontWeight: 'bold', marginRight: '5px', color: '#555'}}>Navigate:</span>
          <button onClick={() => handlePan(-1)} style={navBtnStyle} title="Pan Left">⬅</button>
          <button onClick={() => handlePan(1)} style={navBtnStyle} title="Pan Right">➡</button>
          <div style={{ width: '1px', height: '20px', background: '#ccc', margin: '0 5px' }} />
          <button onClick={() => handleZoom(0.5)} style={navBtnStyle} title="Zoom In">+</button>
          <button onClick={() => handleZoom(2.0)} style={navBtnStyle} title="Zoom Out">−</button>
        </div>

        <button onClick={() => setViewport({ start: 0, end: normalized.fs * 2 })} style={{ ...btnStyle, background: '#2196F3' }}>Reset View</button>
        
        <div style={{ display: 'flex', borderRadius: '4px', overflow: 'hidden', border: '1px solid #ddd' }}>
            <button onClick={() => setDisplayMode('separate')} style={displayMode === 'separate' ? activeModeBtn : inactiveModeBtn}>Grid</button>
            <button onClick={() => setDisplayMode('overlay')} style={displayMode === 'overlay' ? activeModeBtn : inactiveModeBtn}>Stacked</button>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* SIDEBAR */}
        <div style={{ width: '260px', background: '#fff', borderRight: '1px solid #ddd', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '15px', borderBottom: '1px solid #eee' }}>
            <h4 style={{ margin: '0 0 10px 0' }}>Channels</h4>
            <div style={{ display: 'flex', gap: '5px' }}>
              <button onClick={showAll} style={smBtn}>All</button>
              <button onClick={hideAll} style={{ ...smBtn, background: '#9e9e9e' }}>None</button>
            </div>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
            {normalized.channels.map((_, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', padding: '6px', borderRadius: '4px', background: visibleChannels[i] ? '#f1f8e9' : 'transparent' }}>
                <input type="checkbox" checked={!!visibleChannels[i]} onChange={() => toggleChannel(i)} />
                <span style={{ fontSize: '12px', flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{normalized.channelNames[i]}</span>
                <input type="color" value={channelColors[i] || '#000'} onChange={e => setChannelColors(p => ({ ...p, [i]: e.target.value }))} style={{ width: '18px', height: '18px', border: 'none', cursor: 'pointer', background: 'none' }} />
              </div>
            ))}
          </div>
        </div>

        {/* CHART AREA */}
        <div style={{ flex: 1, padding: '20px', overflowY: 'auto', background: '#fafafa' }}>
          {displayMode === 'overlay' ? (
            <div style={{ height: '85%', background: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                  <XAxis 
                    dataKey="time" 
                    type="number" 
                    domain={['dataMin', 'dataMax']} 
                    tickFormatter={(val) => val.toFixed(2)}
                    label={{ value: 'Seconds (s)', position: 'bottom', offset: 0 }} 
                  />
                  <YAxis hide domain={['auto', 'auto']} />
                  <Tooltip content={<CustomTooltip />} isAnimationActive={false} />
                  {normalized.channels.map((_, i) => visibleChannels[i] && (
                    <Line 
                      key={i} 
                      type="linear" 
                      dataKey={`ch${i}_offset`} 
                      stroke={channelColors[i]} 
                      dot={false} 
                      strokeWidth={1.2} 
                      name={normalized.channelNames[i]} 
                      isAnimationActive={false} 
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', marginTop: '15px' }}>
                <label style={{ fontSize: '13px', color: '#666' }}>Overlap Spread: </label>
                <input type="range" min="0.5" max="20" step="0.5" value={offsetAmount} onChange={e => setOffsetAmount(parseFloat(e.target.value))} />
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(450px, 1fr))', gap: '20px' }}>
              {normalized.channels.map((_, i) => visibleChannels[i] && (
                <div key={i} style={{ background: '#fff', padding: '15px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                  <h5 style={{ margin: '0 0 10px 0', color: channelColors[i] }}>{normalized.channelNames[i]}</h5>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                      <CartesianGrid stroke="#f5f5f5" vertical={false} />
                      <XAxis 
                        dataKey="time" 
                        type="number" 
                        domain={['dataMin', 'dataMax']} 
                        tickFormatter={(val) => val.toFixed(2)}
                        fontSize={10}
                      />
                      <YAxis fontSize={10} domain={['auto', 'auto']} />
                      <Line type="linear" dataKey={`ch${i}`} stroke={channelColors[i]} dot={false} isAnimationActive={false} strokeWidth={1.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* FOOTER STATS */}
      <div style={{ padding: '8px 20px', background: '#2c3e50', color: '#ecf0f1', fontSize: '12px', display: 'flex', gap: '25px' }}>
        <span><strong>Sampling Rate:</strong> {normalized.fs} Hz</span>
        <span><strong>Window Size:</strong> {((viewport.end - viewport.start) / normalized.fs).toFixed(2)}s</span>
        <span><strong>Current Position:</strong> {(viewport.start / normalized.fs).toFixed(2)}s / {(normalized.totalSamples / normalized.fs).toFixed(2)}s</span>
      </div>
    </div>
  );
};

const btnStyle = { padding: '8px 16px', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '13px' };
const navBtnStyle = { padding: '5px 10px', background: 'white', border: '1px solid #ccc', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', fontSize: '14px' };
const smBtn = { flex: 1, padding: '4px', fontSize: '11px', background: '#4CAF50', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' };
const activeModeBtn = { padding: '8px 15px', background: '#2196F3', color: 'white', border: 'none', cursor: 'pointer', fontSize: '13px' };
const inactiveModeBtn = { padding: '8px 15px', background: '#eee', color: '#333', border: 'none', cursor: 'pointer', fontSize: '13px' };

export default ContinuousViewer;
