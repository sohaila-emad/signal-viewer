import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Label } from 'recharts';

const XORViewer = ({ data }) => {
  const { channels, channelNames } = useMemo(() => {
    if (!data || !data.data) return { channels: [], channelNames: [] };
    return {
      channels: data.data,
      channelNames: data.channel_names || data.data.map((_, i) => `Channel ${i + 1}`)
    };
  }, [data]);

  const [selectedChannel, setSelectedChannel] = useState(0);
  const [chunkSize, setChunkSize] = useState(250); 
  const [offset, setOffset] = useState(125);      
  const [numChunks, setNumChunks] = useState(5);  
  const [intensity, setIntensity] = useState(1.0);

  const selectedData = channels[selectedChannel] || [];

  // Calculate safe slider ranges based on data length
  const dataLength = selectedData.length;
  const maxChunkSize = Math.max(20, Math.floor(dataLength * 0.5)); // Max 50% of data
  const maxOffset = Math.max(1, Math.floor(dataLength * 0.25)); // Max 25% of data
  const maxNumChunks = Math.max(2, Math.floor(dataLength / 50)); // Conservative estimate

  // Validate and auto-correct slider values
  const safeChunkSize = Math.min(chunkSize, maxChunkSize, dataLength);
  const safeOffset = Math.min(offset, maxOffset, safeChunkSize);
  const safeNumChunks = Math.min(numChunks, maxNumChunks);

  // Warning states
  const chunkSizeWarning = chunkSize > maxChunkSize || chunkSize > dataLength;
  const offsetWarning = offset > maxOffset || offset > safeChunkSize;
  const numChunksWarning = numChunks > maxNumChunks;

  const processedChunks = useMemo(() => {
    if (selectedData.length === 0) return [];
    
    // Calculate max possible chunks with current offset
    const maxChunks = Math.floor((selectedData.length - safeChunkSize) / safeOffset) + 1;
    const effectiveNumChunks = Math.min(safeNumChunks, Math.max(2, maxChunks));
    
    const chunks = [];
    for (let i = 0; i < effectiveNumChunks; i++) {
      const start = i * safeOffset;
      const end = start + safeChunkSize;
      if (end > selectedData.length) break;
      
      const rawSegment = selectedData.slice(start, end);
      const mean = rawSegment.reduce((a, b) => a + b, 0) / rawSegment.length;
      const zeroMeanSegment = rawSegment.map(v => v - mean);
      chunks.push({ id: i, data: zeroMeanSegment, start });
    }
    return chunks;
  }, [selectedData, safeChunkSize, safeOffset, safeNumChunks]);

  const xorResult = useMemo(() => {
    if (processedChunks.length < 2) return [];
    const N = processedChunks[0].data.length;
    const result = [];
    for (let n = 0; n < N; n++) {
      let diffSum = 0;
      const reference = processedChunks[0].data[n];
      for (let i = 1; i < processedChunks.length; i++) {
        diffSum += Math.abs(reference - processedChunks[i].data[n]);
      }
      result.push({ n, residual: (diffSum / (processedChunks.length - 1)) * intensity });
    }
    return result;
  }, [processedChunks, intensity]);

  const overlayData = useMemo(() => {
    if (processedChunks.length === 0) return [];
    return processedChunks[0].data.map((_, n) => {
      const point = { n };
      processedChunks.forEach((c, i) => { point[`chunk${i}`] = c.data[n]; });
      return point;
    });
  }, [processedChunks]);

  if (channels.length === 0) return <div style={{ padding: '20px' }}>No Data Loaded</div>;

  return (
    <div style={{ padding: '20px', backgroundColor: '#ffffff', color: '#333', fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ textAlign: 'center', color: '#444' }}>Signal In-Coherence Analysis</h2>

      {/* Control Panel */}
      <div style={{ 
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
        gap: '15px', padding: '20px', backgroundColor: '#f9f9f9', borderRadius: '10px',
        marginBottom: '30px', border: '1px solid #eee'
      }}>
        <div>
          <label style={{ fontSize: '12px', fontWeight: 'bold' }}>Source Channel</label>
          <select value={selectedChannel} onChange={e => setSelectedChannel(parseInt(e.target.value))} style={{ width: '100%', padding: '6px', borderRadius: '4px' }}>
            {channelNames.map((name, i) => <option key={i} value={i}>{name}</option>)}
          </select>
        </div>
        <div>
          <label style={{ fontSize: '12px', fontWeight: 'bold' }}>
            Chunk Count: {safeNumChunks}
            {numChunksWarning && <span style={{ color: '#d32f2f', marginLeft: '5px' }}>⚠</span>}
          </label>
          <input 
            type="range" 
            min="2" 
            max={maxNumChunks} 
            value={numChunks} 
            onChange={e => setNumChunks(parseInt(e.target.value))} 
            style={{ width: '100%', borderColor: numChunksWarning ? '#d32f2f' : '#ccc' }} 
          />
          {numChunksWarning && <div style={{ fontSize: '10px', color: '#d32f2f', marginTop: '4px' }}>Max allowed: {maxNumChunks}</div>}
        </div>
        <div>
          <label style={{ fontSize: '12px', fontWeight: 'bold' }}>
            Window (Samples): {safeChunkSize}
            {chunkSizeWarning && <span style={{ color: '#d32f2f', marginLeft: '5px' }}>⚠</span>}
          </label>
          <input 
            type="range" 
            min="20" 
            max={maxChunkSize} 
            value={chunkSize} 
            onChange={e => setChunkSize(parseInt(e.target.value))} 
            style={{ width: '100%', borderColor: chunkSizeWarning ? '#d32f2f' : '#ccc' }} 
          />
          {chunkSizeWarning && <div style={{ fontSize: '10px', color: '#d32f2f', marginTop: '4px' }}>Max for this data: {maxChunkSize}</div>}
        </div>
        <div>
          <label style={{ fontSize: '12px', fontWeight: 'bold' }}>
            Offset (Samples): {safeOffset}
            {offsetWarning && <span style={{ color: '#d32f2f', marginLeft: '5px' }}>⚠</span>}
          </label>
          <input 
            type="range" 
            min="1" 
            max={Math.max(1, maxOffset)} 
            value={offset} 
            onChange={e => setOffset(parseInt(e.target.value))} 
            style={{ width: '100%', borderColor: offsetWarning ? '#d32f2f' : '#ccc' }} 
          />
          {offsetWarning && <div style={{ fontSize: '10px', color: '#d32f2f', marginTop: '4px' }}>Max: {Math.max(1, maxOffset)}</div>}
        </div>
        <div>
          <label style={{ fontSize: '12px', fontWeight: 'bold' }}>Intensity: {intensity.toFixed(2)}</label>
          <input type="range" min="0.1" max="5" step="0.1" value={intensity} onChange={e => setIntensity(parseFloat(e.target.value))} style={{ width: '100%' }} />
        </div>
      </div>

      {/* Debug Info */}
      <div style={{ fontSize: '12px', color: '#666', marginBottom: '20px', padding: '10px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
        <strong>Data Info:</strong> {selectedData.length} samples | <strong>Active Chunks:</strong> {processedChunks.length} | <strong>Safe Ranges:</strong> Window max: {maxChunkSize}, Offset max: {maxOffset}, Chunks max: {maxNumChunks}
      </div>

      {/* Graph 1: Residual */}
      <div style={{ marginBottom: '40px' }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#d32f2f' }}>1. Residual Error Plot (XOR Logic)</h4>
        {xorResult.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', backgroundColor: '#f9f9f9', color: '#999' }}>
            Insufficient data to generate residual plot. Adjust sliders to create at least 2 valid chunks.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={xorResult} margin={{ left: 20, right: 30, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="n" hide />
              <YAxis>
                <Label value="Error Magnitude" angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#666', fontSize: '13px' }} />
              </YAxis>
              <Tooltip />
              <Line type="monotone" dataKey="residual" stroke="#000" dot={false} strokeWidth={2} name="Residual" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Graph 2: Overlay */}
      <div>
        <h4 style={{ margin: '0 0 10px 0', color: '#1976d2' }}>2. Waveform Phase Overlay (Zero-Centered)</h4>
        {overlayData.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', backgroundColor: '#f9f9f9', color: '#999' }}>
            No waveform data available. Adjust sliders to create valid chunks.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={overlayData} margin={{ left: 20, right: 30, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="n">
                <Label value="Samples (n)" offset={-10} position="insideBottom" style={{ fill: '#666' }} />
              </XAxis>
              <YAxis>
                <Label value="Amplitude (AC)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle', fill: '#666', fontSize: '13px' }} />
              </YAxis>
              <Legend verticalAlign="top" height={36}/>
              {processedChunks.map((_, i) => (
                <Line 
                  key={i} 
                  type="monotone" 
                  dataKey={`chunk${i}`} 
                  stroke={i === 0 ? "#ec6e6e" : `hsl(${i * 45}, 60%, 70%)`} 
                  dot={false} 
                  strokeWidth={i === 0 ? 3 : 2}
                  opacity={i === 0 ? 1 : 0.6}
                  name={i === 0 ? "Reference" : `Chunk ${i + 1}`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

export default XORViewer;