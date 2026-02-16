import React, { useMemo } from 'react';

const SignalInfo = ({ data }) => {
  if (!data) return null;

  const getTypeIcon = (type) => {
    switch(type) {
      case 'medical': return '❤️';
      case 'acoustic': return '🎵';
      case 'stock': return '📈';
      default: return '📊';
    }
  };

  // Use useMemo to prevent recalculating on every render
  const stats = useMemo(() => {
    if (!data.data || !data.data[0]) return { min: 0, max: 0, mean: 0 };
    
    try {
      // Take only first 1000 samples from first channel to avoid stack overflow
      const firstChannel = data.data[0];
      const samples = firstChannel.slice(0, 1000);
      
      if (samples.length === 0) return { min: 0, max: 0, mean: 0 };
      
      const min = Math.min(...samples).toFixed(3);
      const max = Math.max(...samples).toFixed(3);
      const sum = samples.reduce((a, b) => a + b, 0);
      const mean = (sum / samples.length).toFixed(3);
      
      return { min, max, mean };
    } catch (error) {
      console.error('Error calculating stats:', error);
      return { min: 0, max: 0, mean: 0 };
    }
  }, [data]);

  return (
    <div className="signal-info-card">
      <h3 className="info-title">
        <span className="info-icon">ℹ️</span>
        Current File Info
      </h3>
      
      <div className="info-header">
        <div className="file-icon-large">{getTypeIcon(data.type)}</div>
        <div className="file-details">
          <div className="file-name">{data.filename || 'Unknown'}</div>
          <div className="file-meta">
            <span className="meta-tag">{data.type || 'unknown'}</span>
          </div>
        </div>
      </div>

      <div className="info-grid">
        <div className="info-item">
          <span className="info-label">📊 Channels</span>
          <span className="info-value">{data.channels || data.data?.length || 0}</span>
        </div>
        <div className="info-item">
          <span className="info-label">⏱️ Sampling Rate</span>
          <span className="info-value">{data.fs || 'N/A'} Hz</span>
        </div>
        <div className="info-item">
          <span className="info-label">📏 Duration</span>
          <span className="info-value">
            {data.time && data.fs ? ((data.time.length / data.fs).toFixed(1)) : 'N/A'} 
          </span>
        </div>
        <div className="info-item">
          <span className="info-label">📈 Samples</span>
          <span className="info-value">{data.time?.length || 0}</span>
        </div>
      </div>

      <div className="info-stats">
        <h4>Signal Statistics (Channel 1)</h4>
        <div className="stats-grid">
          <div className="stat-box">
            <span className="stat-label">Min</span>
            <span className="stat-number">{stats.min}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">Max</span>
            <span className="stat-number">{stats.max}</span>
          </div>
          <div className="stat-box">
            <span className="stat-label">Mean</span>
            <span className="stat-number">{stats.mean}</span>
          </div>
        </div>
      </div>

      {data.abnormality && (
        <div className="abnormality-badge">
          <span className="badge-icon">🔍</span>
          <span className="badge-text">Detected: {data.abnormality}</span>
        </div>
      )}
    </div>
  );
};

export default SignalInfo;
