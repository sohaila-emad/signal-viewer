import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const XORViewer = ({ data }) => {
  // Get signal length first to set appropriate defaults
  const getSignalLength = () => {
    if (!data) return 0;
    try {
      if (data.type === 'stock' && data.data) {
        return data.data[0]?.length || 0;
      }
      if (data.data && Array.isArray(data.data)) {
        return data.data[0]?.length || 0;
      }
    } catch (err) {
      return 0;
    }
    return 0;
  };

  const signalLength = getSignalLength();
  
  // Set chunk size based on signal length (max 20% of signal or 30, whichever is smaller)
  const defaultChunkSize = Math.min(30, Math.floor(signalLength * 0.2));
  
  const [chunkSize, setChunkSize] = useState(defaultChunkSize || 20);
  const [offset, setOffset] = useState(Math.floor((defaultChunkSize || 20) / 2));
  const [selectedChannel, setSelectedChannel] = useState(0);
  const [colorMap, setColorMap] = useState('rainbow');
  const [showGrid, setShowGrid] = useState(true);
  const [intensity, setIntensity] = useState(1.0);
  const [maxChunks, setMaxChunks] = useState(10);
  const [error, setError] = useState(null);

  // Update defaults when signal length changes
  useEffect(() => {
    if (signalLength > 0) {
      const newChunkSize = Math.min(30, Math.floor(signalLength * 0.2));
      setChunkSize(newChunkSize);
      setOffset(Math.floor(newChunkSize / 2));
    }
  }, [signalLength]);

  // Normalize data
  const normalizeData = useCallback(() => {
    if (!data) return { channels: [], timeData: [], channelNames: [], isStock: false };
    
    try {
      if (data.type === 'stock' && data.date_labels) {
        return {
          channels: data.data || [],
          timeData: data.date_labels,
          channelNames: data.channel_names || [],
          isStock: true
        };
      }
      
      if (data.channel_names) {
        return {
          channels: data.data || [],
          timeData: data.time || [],
          channelNames: data.channel_names,
          isStock: false
        };
      }
      
      if (data.data && Array.isArray(data.data)) {
        return {
          channels: data.data,
          timeData: data.time || [],
          channelNames: data.channel_names || 
            (data.channels ? Array(data.channels).fill(0).map((_, i) => `Channel ${i+1}`) : []),
          isStock: false
        };
      }
    } catch (err) {
      setError('Error normalizing data: ' + err.message);
    }
    
    return { channels: [], timeData: [], channelNames: [] };
  }, [data]);

  const { channels, timeData, channelNames, isStock } = normalizeData();
  
  // Validate selected channel
  useEffect(() => {
    if (selectedChannel >= channels.length) {
      setSelectedChannel(0);
    }
  }, [channels.length, selectedChannel]);

  const selectedChannelData = channels[selectedChannel] || [];

  // Generate XOR chunks with validation
  const xorChunks = useMemo(() => {
    if (selectedChannelData.length === 0 || chunkSize <= 0) return [];
    
    // Validate chunk size
    if (chunkSize > selectedChannelData.length) {
      setError(`Chunk size (${chunkSize}) > signal length (${selectedChannelData.length})`);
      return [];
    }

    setError(null);
    const chunks = [];
    
    // Calculate chunks
    const maxPossibleChunks = Math.floor((selectedChannelData.length - chunkSize) / Math.max(1, offset)) + 1;
    const numChunks = Math.min(maxPossibleChunks, maxChunks);
    
    // If we have less than 2 chunks, show warning
    if (maxPossibleChunks < 2) {
      setError(`Need at least 2 chunks. Try smaller chunk size or offset.`);
      return [];
    }
    
    // Sample chunks evenly
    const step = maxPossibleChunks > numChunks ? Math.floor(maxPossibleChunks / numChunks) : 1;
    
    for (let i = 0; i < numChunks; i++) {
      const chunkIndex = i * step;
      const start = chunkIndex * offset;
      const end = start + chunkSize;
      
      if (end > selectedChannelData.length) continue;
      
      const chunk = selectedChannelData.slice(start, end);
      
      // Calculate min/max
      let min = Infinity, max = -Infinity;
      for (let j = 0; j < chunk.length; j++) {
        const val = chunk[j];
        if (val < min) min = val;
        if (val > max) max = val;
      }
      
      chunks.push({
        index: i,
        start,
        end,
        data: chunk,
        min,
        max
      });
    }
    
    return chunks;
  }, [selectedChannelData, chunkSize, offset, maxChunks]);

  // Apply XOR between chunks
  const xorResult = useMemo(() => {
    if (xorChunks.length < 2) return [];

    try {
      const result = [];
      const firstChunk = xorChunks[0];
      const resultLength = firstChunk.data.length;
      
      for (let j = 0; j < resultLength; j++) {
        let diffSum = 0;
        let count = 0;
        const a = firstChunk.data[j];
        
        // Compare with all other chunks
        for (let i = 1; i < xorChunks.length; i++) {
          const chunk = xorChunks[i];
          if (j < chunk.data.length) {
            const b = chunk.data[j];
            // Simple normalized difference
            const maxVal = Math.max(Math.abs(a), Math.abs(b), 1);
            const diff = Math.abs(a - b) / maxVal;
            diffSum += diff;
            count++;
          }
        }
        
        // Average difference with intensity
        const avgDiff = count > 0 ? (diffSum / count) * intensity : 0;
        result.push(avgDiff * 100); // Scale to percentage for display
      }
      
      return result;
    } catch (err) {
      setError('Error computing XOR: ' + err.message);
      return [];
    }
  }, [xorChunks, intensity]);

  // Prepare chart data
  const chartData = useMemo(() => {
    return xorResult.map((value, i) => ({
      index: i,
      value: value
    }));
  }, [xorResult]);

  // Prepare overlay data
  const chunkOverlayData = useMemo(() => {
    if (xorChunks.length === 0) return [];
    
    const maxPoints = Math.min(100, ...xorChunks.map(c => c.data.length));
    const displayChunks = xorChunks.slice(0, Math.min(8, xorChunks.length));
    
    const result = [];
    for (let i = 0; i < maxPoints; i++) {
      const point = { index: i };
      displayChunks.forEach((chunk, chunkIdx) => {
        if (i < chunk.data.length) {
          point[`chunk${chunkIdx}`] = chunk.data[i];
        }
      });
      result.push(point);
    }
    
    return result;
  }, [xorChunks]);

  // Get color
  const getColor = (index, total) => {
    const hue = (index * 360 / Math.max(1, total)) % 360;
    
    switch(colorMap) {
      case 'rainbow':
        return `hsl(${hue}, 80%, 60%)`;
      case 'heat':
        return `hsl(${30 + index * 10}, 90%, 50%)`;
      case 'cool':
        return `hsl(${180 + index * 15}, 80%, 60%)`;
      case 'monochrome':
        return `rgba(100, 100, 255, ${0.3 + (index / Math.max(1, total)) * 0.7})`;
      default:
        return `hsl(${hue}, 80%, 60%)`;
    }
  };

  if (channels.length === 0) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100%',
        color: '#666',
        fontSize: '1.2em'
      }}>
        No data available for XOR view
      </div>
    );
  }

  return (
    <div className="xor-viewer" style={{ padding: '20px' }}>
      {/* Error Display */}
      {error && (
        <div style={{ 
          backgroundColor: '#ffebee', 
          color: '#c62828',
          padding: '10px',
          borderRadius: '4px',
          marginBottom: '20px',
          border: '1px solid #ef9a9a'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Signal Info */}
      <div style={{ 
        background: '#e8f5e9', 
        padding: '10px', 
        borderRadius: '4px',
        marginBottom: '15px',
        fontSize: '0.9rem'
      }}>
        <strong>Signal Length:</strong> {selectedChannelData.length} samples | 
        <strong> Chunk Size:</strong> {chunkSize} | 
        <strong> Available Chunks:</strong> {Math.floor((selectedChannelData.length - chunkSize) / offset) + 1}
      </div>

      {/* Controls */}
      <div style={{ 
        background: '#f5f5f5', 
        padding: '15px', 
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <h3 style={{ margin: '0 0 15px 0' }}>XOR Viewer Controls</h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px' }}>
          {/* Channel Selection */}
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Channel:
            </label>
            <select 
              value={selectedChannel}
              onChange={(e) => setSelectedChannel(parseInt(e.target.value))}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            >
              {channelNames.map((name, idx) => (
                <option key={idx} value={idx}>{name || `Channel ${idx+1}`}</option>
              ))}
            </select>
          </div>

          {/* Chunk Size */}
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Chunk Size: {chunkSize}
            </label>
            <input 
              type="range" 
              min="5" 
              max={Math.min(50, Math.floor(selectedChannelData.length / 2))} 
              step="1"
              value={chunkSize}
              onChange={(e) => setChunkSize(parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
            <div style={{ fontSize: '0.8rem', color: '#666' }}>
              Max: {Math.floor(selectedChannelData.length / 2)}
            </div>
          </div>

          {/* Offset */}
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Offset: {offset}
            </label>
            <input 
              type="range" 
              min="1" 
              max={chunkSize}
              value={offset}
              onChange={(e) => setOffset(parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          {/* Max Chunks */}
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Max Chunks: {maxChunks}
            </label>
            <input 
              type="range" 
              min="2" 
              max="20" 
              step="1"
              value={maxChunks}
              onChange={(e) => setMaxChunks(parseInt(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          {/* Intensity */}
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Intensity: {intensity.toFixed(1)}
            </label>
            <input 
              type="range" 
              min="0" 
              max="2" 
              step="0.1"
              value={intensity}
              onChange={(e) => setIntensity(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          {/* Color Map */}
          <div>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Color Map:
            </label>
            <select 
              value={colorMap}
              onChange={(e) => setColorMap(e.target.value)}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            >
              <option value="rainbow">Rainbow</option>
              <option value="heat">Heat</option>
              <option value="cool">Cool</option>
              <option value="monochrome">Monochrome</option>
            </select>
          </div>

          {/* Grid Toggle */}
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <input 
                type="checkbox" 
                checked={showGrid}
                onChange={(e) => setShowGrid(e.target.checked)}
              />
              Show Grid
            </label>
          </div>
        </div>
      </div>

      {/* XOR Result Graph */}
      <div style={{ 
        background: 'white', 
        padding: '20px', 
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <h4 style={{ margin: '0 0 15px 0' }}>XOR Result (Difference %)</h4>
        {chartData.length > 0 ? (
          <LineChart
            width={800}
            height={250}
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
            <XAxis 
              dataKey="index" 
              label={{ value: 'Sample Index', position: 'bottom' }}
            />
            <YAxis 
              label={{ value: 'Difference %', angle: -90, position: 'left' }}
              domain={[0, 100]}
            />
            <Tooltip />
            <Line 
              type="monotone"
              dataKey="value"
              stroke="#ff6b6b"
              dot={false}
              strokeWidth={2}
              name="XOR Difference"
            />
          </LineChart>
        ) : (
          <div style={{ textAlign: 'center', padding: '50px', color: '#999' }}>
            {error || 'Adjust chunk size to see XOR result'}
          </div>
        )}
      </div>

      {/* Chunks Overlay */}
      <div style={{ 
        background: 'white', 
        padding: '20px', 
        borderRadius: '8px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h4 style={{ margin: '0 0 15px 0' }}>Chunks Overlay</h4>
        {chunkOverlayData.length > 0 && xorChunks.length > 0 ? (
          <LineChart
            width={800}
            height={300}
            data={chunkOverlayData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />}
            <XAxis 
              dataKey="index" 
              label={{ value: 'Sample Index', position: 'bottom' }}
            />
            <YAxis 
              label={{ value: isStock ? 'Price (USD)' : 'Amplitude', angle: -90, position: 'left' }}
            />
            <Tooltip />
            <Legend />
            {xorChunks.slice(0, 8).map((_, idx) => (
              <Line 
                key={idx}
                type="monotone"
                dataKey={`chunk${idx}`}
                stroke={getColor(idx, xorChunks.length)}
                dot={false}
                strokeWidth={1.5}
                name={`Chunk ${idx + 1}`}
                opacity={0.7}
              />
            ))}
          </LineChart>
        ) : (
          <div style={{ textAlign: 'center', padding: '50px', color: '#999' }}>
            Not enough chunks to display
          </div>
        )}
        
        {/* Stats */}
        {xorChunks.length > 0 && (
          <div style={{ 
            marginTop: '15px',
            display: 'flex',
            gap: '20px',
            padding: '10px',
            background: '#f5f5f5',
            borderRadius: '4px'
          }}>
            <div><strong>Total chunks:</strong> {xorChunks.length}</div>
            <div><strong>First chunk:</strong> samples {xorChunks[0]?.start}-{xorChunks[0]?.end}</div>
            <div><strong>Last chunk:</strong> samples {xorChunks[xorChunks.length-1]?.start}-{xorChunks[xorChunks.length-1]?.end}</div>
          </div>
        )}
        
        {/* Explanation */}
        <div style={{ 
          marginTop: '20px',
          padding: '15px',
          background: '#fff3e0',
          borderRadius: '4px',
          borderLeft: '4px solid #ff9800'
        }}>
          <h5 style={{ margin: '0 0 10px 0', color: '#e65100' }}>How XOR Viewer Works:</h5>
          <p style={{ margin: 0, fontSize: '0.9rem' }}>
            • Signal divided into <strong>{xorChunks.length}</strong> chunks of <strong>{chunkSize}</strong> samples<br />
            • Each chunk offset by <strong>{offset}</strong> samples<br />
            • Overlaid chunks compared - differences become visible<br />
            • Similar chunks partially cancel out<br />
            • Result shows patterns and anomalies
          </p>
        </div>
      </div>
    </div>
  );
};

export default XORViewer;
