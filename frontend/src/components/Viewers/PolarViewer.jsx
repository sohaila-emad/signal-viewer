import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { 
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, 
  Radar, RadarChart, Tooltip, ResponsiveContainer 
} from 'recharts';

const PolarViewer = ({ data }) => {
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [mode, setMode] = useState('sliding'); 
  const [timeWindow, setTimeWindow] = useState(400); 
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playback, setPlayback] = useState(false);
  const [speed, setSpeed] = useState(5); 
  const [colorMap, setColorMap] = useState('phase');
  const [showGrid, setShowGrid] = useState(true);
  
  const animationRef = useRef();

  // 1. DATA NORMALIZATION
  const normalizedData = useCallback(() => {
    if (!data) return { channels: [], timeData: [], channelNames: [] };
    const channels = data.data || [];
    const channelNames = data.channel_names || (data.channels ? Array(data.channels).fill(0).map((_, i) => `Channel ${i+1}`) : []);
    return { channels, timeData: data.time || [], channelNames };
  }, [data]);

  const { channels, timeData, channelNames } = normalizedData();
  const selectedData = channels[selectedChannel] || [];

  // 2. MEDICAL DSP: PERIOD DETECTION (L)
  const L = useMemo(() => {
    if (selectedData.length < 50) return 100;
    const threshold = Math.max(...selectedData.slice(0, 1000)) * 0.7;
    const peaks = [];
    for (let i = 1; i < Math.min(selectedData.length, 1000); i++) {
      if (selectedData[i] > threshold && selectedData[i] > selectedData[i-1] && selectedData[i] > selectedData[i+1]) {
        peaks.push(i);
      }
    }
    if (peaks.length < 2) return 100;
    let sum = 0;
    for (let i = 1; i < peaks.length; i++) sum += (peaks[i] - peaks[i-1]);
    return sum / (peaks.length - 1);
  }, [selectedData]);

  // 3. COLOR & PHASE DEFINITIONS
  const PHASES = {
    P_WAVE: { range: [10, 70], color: '#2196F3', label: 'P-Wave (Atrial)' },
    QRS: { range: [150, 200], color: '#FF5252', label: 'QRS (Ventricular)' },
    T_WAVE: { range: [230, 290], color: '#4CAF50', label: 'T-Wave (Recovery)' },
    BASE: { color: '#90A4AE', label: 'Baseline' }
  };

  const getMedicalColor = (angle) => {
    if (angle > PHASES.QRS.range[0] && angle < PHASES.QRS.range[1]) return PHASES.QRS.color;
    if (angle > PHASES.T_WAVE.range[0] && angle < PHASES.T_WAVE.range[1]) return PHASES.T_WAVE.color;
    if (angle > PHASES.P_WAVE.range[0] && angle < PHASES.P_WAVE.range[1]) return PHASES.P_WAVE.color;
    return PHASES.BASE.color;
  };

  // 4. GENERATE POLAR POINTS
  const polarData = useMemo(() => {
    if (selectedData.length === 0) return [];
    const points = [];
    let startIdx = mode === 'sliding' ? Math.max(0, currentIndex - timeWindow) : 0;
    let endIdx = mode === 'sliding' ? currentIndex + 1 : selectedData.length;
    
    const min = Math.min(...selectedData);
    const max = Math.max(...selectedData);
    const range = max - min || 1;

    for (let i = startIdx; i < endIdx; i++) {
      const val = selectedData[i];
      const angle = ((i % L) / L) * 360; 
      const radius = 30 + ((val - min) / range) * 70;

      points.push({
        index: i,
        time: timeData[i] || i,
        angle,
        radius,
        value: val,
        x: 200 + radius * 1.4 * Math.cos((angle - 90) * Math.PI / 180),
        y: 200 + radius * 1.4 * Math.sin((angle - 90) * Math.PI / 180)
      });
    }
    return points;
  }, [selectedData, mode, currentIndex, timeWindow, L, timeData]);

  // 5. RADAR BINNED DATA (Averaged shape)
  const radarData = useMemo(() => {
    const bins = 24;
    const binData = Array(bins).fill().map((_, i) => ({ angle: i * (360/bins), values: [] }));
    polarData.forEach(p => {
      const binIdx = Math.floor(p.angle / (360/bins)) % bins;
      binData[binIdx].values.push(p.radius);
    });
    return binData.map(b => ({
      angle: b.angle,
      radius: b.values.length > 0 ? b.values.reduce((a,b)=>a+b,0)/b.values.length : 0,
      count: b.values.length
    }));
  }, [polarData]);



  // 7. ANIMATION
  useEffect(() => {
    if (playback && mode === 'sliding') {
      const animate = () => {
        setCurrentIndex(prev => (prev + speed >= selectedData.length ? 0 : prev + speed));
        animationRef.current = setTimeout(animate, 30);
      };
      animationRef.current = setTimeout(animate, 30);
    } else {
      clearTimeout(animationRef.current);
    }
    return () => clearTimeout(animationRef.current);
  }, [playback, speed, selectedData.length, mode]);

  return (
    <div style={{ padding: '20px', background: '#f0f2f5', minHeight: '100vh', fontFamily: 'Segoe UI, sans-serif' }}>
      
      {/* 1. HEADER & STABILITY GAUGE */}
      <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
        <div style={{ flex: 1, background: '#fff', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
          <h2 style={{ margin: 0, color: '#1a3353' }}>Medical Polar-Polar DSP Viewer</h2>
          <div style={{ display: 'flex', gap: '20px', marginTop: '10px' }}>
            <span><strong>Sync:</strong> {L.toFixed(1)} samples</span>
            <span><strong>Mode:</strong> {mode.toUpperCase()}</span>
            {/*<span><strong>Status:</strong> {stability > 75 ? '🟢 Stable' : '🟡 Varied'}</span>*/}
          </div>
        </div>


      </div>

      {/* 2. CONTROLS BAR */}
      <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', marginBottom: '20px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '20px' }}>
        <div>
          <label style={{ fontSize: '0.8rem', fontWeight: 'bold' }}>CHANNEL</label>
          <select value={selectedChannel} onChange={e => setSelectedChannel(Number(e.target.value))} style={{ width: '100%', padding: '8px', marginTop: '5px' }}>
            {channelNames.map((n, i) => <option key={i} value={i}>{n}</option>)}
          </select>
        </div>
        <div>
          <label style={{ fontSize: '0.8rem', fontWeight: 'bold' }}>MODE</label>
          <select value={mode} onChange={e => setMode(e.target.value)} style={{ width: '100%', padding: '8px', marginTop: '5px' }}>
            <option value="sliding">Sliding Window (Live)</option>
            <option value="cumulative">Cumulative (History)</option>
          </select>
        </div>
        {mode === 'sliding' && (
          <>
            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 'bold' }}>PERSISTENCE: {timeWindow}</label>
              <input type="range" min="0.0" max="1000" value={timeWindow} onChange={e => setTimeWindow(Number(e.target.value))} style={{ width: '100%', marginTop: '5px' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button onClick={() => setPlayback(!playback)} style={{ width: '100%', padding: '10px', background: playback ? '#f44336' : '#4CAF50', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 'bold' }}>
                {playback ? '⏸ PAUSE' : '▶ LIVE'}
              </button>
            </div>
          </>
        )}
      </div>

      {/* 3. MAIN GRAPHS (DUAL VIEW) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
        
        {/* Radar View (Averaged Shape) */}
        <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', height: '480px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
          <h4 style={{ margin: '0 0 10px 0' }}>Binned Distribution (Radar)</h4>
          <ResponsiveContainer width="100%" height="90%">
            <RadarChart outerRadius="80%" data={radarData}>
              <PolarGrid stroke="#eee" />
              <PolarAngleAxis dataKey="angle" tickFormatter={t => `${t}°`} />
              <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
              <Radar name="Signal" dataKey="radius" stroke="#8884d8" fill="#8884d8" fillOpacity={0.5} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Vector Plot (Raw Data) */}
        <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', height: '480px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
          <h4 style={{ margin: '0 0 10px 0' }}>Phasic Vector Plot (Scatter)</h4>
          <svg width="100%" height="90%" viewBox="0 0 400 400">
            {showGrid && [40, 80, 120, 150].map(r => <circle key={r} cx="200" cy="200" r={r} fill="none" stroke="#f0f0f0" />)}
            {polarData.map((p, i) => (
              <circle 
                key={i} cx={p.x} cy={p.y} r={i === polarData.length - 1 ? 6 : 2.5} 
                fill={getMedicalColor(p.angle)}
                opacity={mode === 'sliding' ? (i / polarData.length) : 0.4}
              />
            ))}
            <circle cx="200" cy="200" r="4" fill="#333" />
          </svg>
        </div>
      </div>

      {/* 4. LEGEND & DATA TABLE SECTION */}
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px' }}>
        
        {/* Legend Panel */}
        <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
          <h4 style={{ margin: '0 0 15px 0' }}>Phase Color Legend</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {Object.values(PHASES).map((phase, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ width: '18px', height: '18px', background: phase.color, borderRadius: '4px' }} />
                <span style={{ fontSize: '0.85rem' }}>{phase.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Data Table Panel */}
        <div style={{ background: '#fff', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
          <h4 style={{ margin: '0 0 15px 0' }}>Phasic Coordinate Log</h4>
          <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '2px solid #eee', color: '#888' }}>
                  <th style={{ padding: '8px' }}>INDEX</th>
                  <th style={{ padding: '8px' }}>PHASE Angle (θ)</th>
                  <th style={{ padding: '8px' }}>RADIUS (r)</th>
                  <th style={{ padding: '8px' }}>VALUE</th>
                </tr>
              </thead>
              <tbody>
                {polarData.slice(-5).reverse().map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f9f9f9' }}>
                    <td style={{ padding: '8px' }}>{p.index}</td>
                    <td style={{ padding: '8px', color: getMedicalColor(p.angle), fontWeight: 'bold' }}>{p.angle.toFixed(1)}°</td>
                    <td style={{ padding: '8px' }}>{p.radius.toFixed(2)}</td>
                    <td style={{ padding: '8px' }}>{p.value.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      {/* Time Slider */}
      {mode === 'sliding' && (
        <div style={{ marginTop: '20px', background: '#fff', padding: '15px', borderRadius: '12px' }}>
          <label style={{ fontSize: '0.8rem', fontWeight: 'bold', color: '#888' }}>TIMELINE POSITION</label>
          <input type="range" min="0" max={selectedData.length - 1} value={currentIndex} onChange={e => setCurrentIndex(Number(e.target.value))} style={{ width: '100%', marginTop: '10px' }} />
        </div>
      )}
    </div>
  );
};

export default PolarViewer;