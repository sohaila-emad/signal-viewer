import React, { useState, useEffect, useRef, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ContinuousViewer = ({ data }) => {
  const [viewport, setViewport] = useState({ start: 0, end: 500 });
  const [visibleChannels, setVisibleChannels] = useState({});
  const [playback, setPlayback] = useState({ playing: false, speed: 1 });
  const [channelColors, setChannelColors] = useState({});
  const [displayMode, setDisplayMode] = useState('overlay');
  const [offsetAmount, setOffsetAmount] = useState(3);
  const animationRef = useRef();

  // 1. DATA NORMALIZATION (Restored your exact logic but optimized with Memo)
  const normalized = useMemo(() => {
    if (!data || !data.data) return null;
    const signalData = data.data;
    const fs = data.fs || 250;
    const samplesCount = signalData[0]?.length || 0;
    const channelNames = data.channel_names || signalData.map((_, i) => `Channel ${i + 1}`);
    const timeArray = Array(samplesCount).fill(0).map((_, i) => (i / fs).toFixed(3));

    return { channels: signalData, timeData: timeArray, channelNames, fs };
  }, [data]);

  // 2. INITIALIZATION (Fixed the Hide All bug with a Ref)
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
      setViewport({ start: 0, end: Math.min(500, normalized.timeData.length) });
      isInitialized.current = true;
    }
  }, [normalized]);

  // Reset if a new file is uploaded
  useEffect(() => { isInitialized.current = false; }, [data]);

  // 3. PLAYBACK
  useEffect(() => {
    if (playback.playing && normalized) {
      const animate = () => {
        setViewport(prev => {
          const nextEnd = prev.end + playback.speed;
          if (nextEnd >= normalized.timeData.length) {
            setPlayback(p => ({ ...p, playing: false }));
            return prev;
          }
          return { start: prev.start + playback.speed, end: nextEnd };
        });
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    }
    return () => cancelAnimationFrame(animationRef.current);
  }, [playback.playing, playback.speed, normalized]);

  // 4. CHART DATA FORMATTING (Restored Stacked Offset logic)
  const chartData = useMemo(() => {
    if (!normalized) return [];
    const start = Math.max(0, Math.floor(viewport.start));
    const end = Math.min(normalized.timeData.length, Math.ceil(viewport.end));
    const result = [];

    // Identify visible channels for stacking order
    const activeIndices = Object.keys(visibleChannels)
      .filter(k => visibleChannels[k])
      .map(Number)
      .sort((a, b) => a - b);

    for (let i = start; i < end; i++) {
      const point = { time: normalized.timeData[i] };
      activeIndices.forEach((chIdx, stackIdx) => {
        const val = normalized.channels[chIdx][i] || 0;
        point[`ch${chIdx}`] = val; // Raw for Separate view
        point[`ch${chIdx}_offset`] = val + (stackIdx * offsetAmount); // For Stacked view
      });
      result.push(point);
    }
    return result;
  }, [viewport, normalized, visibleChannels, offsetAmount]);

  // 5. HANDLERS
  const toggleChannel = (idx) => setVisibleChannels(p => ({ ...p, [idx]: !p[idx] }));
  const showAll = () => setVisibleChannels(Object.fromEntries(normalized.channels.map((_, i) => [i, true])));
  const hideAll = () => setVisibleChannels(Object.fromEntries(normalized.channels.map((_, i) => [i, false])));

  // 6. CUSTOM TOOLTIP (Restored your mV display)
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: '#fff', border: '1px solid #ccc', padding: '10px', borderRadius: '4px' }}>
          <p style={{ margin: 0 }}><strong>Time:</strong> {label} s</p>
          {payload.map((entry, i) => (
            <p key={i} style={{ color: entry.color, margin: '2px 0', fontSize: '12px' }}>
              {entry.name}: {entry.value.toFixed(3)} mV
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (!normalized) return <div style={{ padding: '20px' }}>Loading medical data...</div>;

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#f0f2f5' }}>
      
      {/* HEADER CONTROLS */}
      <div style={{ padding: '15px', background: '#fff', borderBottom: '1px solid #ddd', display: 'flex', gap: '15px', flexWrap: 'wrap', alignItems: 'center' }}>
        <button onClick={() => setPlayback(p => ({ ...p, playing: !p.playing }))} style={{ ...btnStyle, background: playback.playing ? '#f44336' : '#4CAF50' }}>{playback.playing ? 'Pause' : 'Play'}</button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span>Speed:</span>
          <input type="range" min="0.1" max="5" step="0.1" value={playback.speed} onChange={e => setPlayback(p => ({ ...p, speed: parseFloat(e.target.value) }))} />
          <span>{playback.speed}x</span>
        </div>
        <button onClick={() => setViewport({ start: 0, end: 500 })} style={{ ...btnStyle, background: '#2196F3' }}>Reset</button>
        <div style={{ display: 'flex', gap: '5px' }}>
            <button onClick={() => setDisplayMode('separate')} style={displayMode === 'separate' ? activeModeBtn : inactiveModeBtn}>Grid</button>
            <button onClick={() => setDisplayMode('overlay')} style={displayMode === 'overlay' ? activeModeBtn : inactiveModeBtn}>Stacked</button>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        
        {/* SCROLLABLE SIDEBAR FOR MANY CHANNELS */}
        <div style={{ width: '280px', background: '#fff', borderRight: '1px solid #ddd', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '10px', borderBottom: '1px solid #eee' }}>
            <h4 style={{ margin: '0 0 10px 0' }}>Channels ({normalized.channels.length})</h4>
            <div style={{ display: 'flex', gap: '5px' }}>
              <button onClick={showAll} style={smBtn}>Show All</button>
              <button onClick={hideAll} style={{ ...smBtn, background: '#f44336' }}>Hide All</button>
            </div>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
            {normalized.channels.map((_, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', padding: '5px', borderRadius: '4px', background: visibleChannels[i] ? '#e8f5e9' : 'transparent' }}>
                <input type="checkbox" checked={visibleChannels[i]} onChange={() => toggleChannel(i)} />
                <span style={{ fontSize: '12px', flex: 1, fontWeight: visibleChannels[i] ? 'bold' : 'normal' }}>{normalized.channelNames[i]}</span>
                <input type="color" value={channelColors[i] || '#000'} onChange={e => setChannelColors(p => ({ ...p, [i]: e.target.value }))} style={{ width: '20px', height: '20px', border: 'none', cursor: 'pointer' }} />
              </div>
            ))}
          </div>
        </div>

        {/* CHART AREA */}
        <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
          {displayMode === 'overlay' ? (
            <div style={{ height: '500px', background: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ left: 30, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" label={{ value: 'Time (s)', position: 'bottom' }} />
                  <YAxis label={{ value: 'Amplitude (mV) + Offset', angle: -90, position: 'left' }} />
                  <Tooltip content={<CustomTooltip />} />
                  {normalized.channels.map((_, i) => visibleChannels[i] && (
                    <Line key={i} type="monotone" dataKey={`ch${i}_offset`} stroke={channelColors[i]} dot={false} strokeWidth={1.5} name={normalized.channelNames[i]} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', marginTop: '10px' }}>
                <label>Vertical Spacing: </label>
                <input type="range" min="1" max="10" step="0.5" value={offsetAmount} onChange={e => setOffsetAmount(parseFloat(e.target.value))} />
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
              {normalized.channels.map((_, i) => visibleChannels[i] && (
                <div key={i} style={{ background: '#fff', padding: '15px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                  <h5 style={{ margin: '0 0 10px 0', color: channelColors[i] }}>{normalized.channelNames[i]}</h5>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData} margin={{ left: 20, bottom: 20 }}>
                      <CartesianGrid stroke="#f0f0f0" />
                      <XAxis dataKey="time" label={{ value: 'Time (s)', position: 'bottom', fontSize: 10 }} />
                      <YAxis label={{ value: 'mV', angle: -90, position: 'left', fontSize: 10 }} />
                      <Line type="monotone" dataKey={`ch${i}`} stroke={channelColors[i]} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* FOOTER */}
      <div style={{ padding: '10px 20px', background: '#34495e', color: '#fff', fontSize: '12px', display: 'flex', gap: '20px' }}>
        <div><strong>Rate:</strong> {normalized.fs} Hz</div>
        <div><strong>Window:</strong> {((viewport.end - viewport.start) / normalized.fs).toFixed(2)}s</div>
        <div><strong>Duration:</strong> {(normalized.timeData.length / normalized.fs).toFixed(2)}s</div>
      </div>
    </div>
  );
};

// Styles
const btnStyle = { padding: '8px 16px', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' };
const smBtn = { flex: 1, padding: '5px', fontSize: '11px', background: '#4CAF50', color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer' };
const activeModeBtn = { padding: '6px 12px', background: '#2196F3', color: 'white', border: 'none', borderRadius: '4px 0 0 4px', cursor: 'pointer' };
const inactiveModeBtn = { padding: '6px 12px', background: '#eee', color: '#333', border: 'none', borderRadius: '0 4px 4px 0', cursor: 'pointer' };

export default ContinuousViewer;