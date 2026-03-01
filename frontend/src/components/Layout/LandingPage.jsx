import React from 'react';
import './LandingPage.css';

const modules = [
  {
    id: 'medical',
    label: 'ECG Analysis',
    icon: '❤️',
    color: '#ff6b6b',
    desc: 'Upload ECG/medical signals for AI-powered arrhythmia detection and multi-channel visualization.',
    tag: 'Upload required',
  },
  {
    id: 'eeg',
    label: 'EEG Analysis',
    icon: '🧠',
    color: '#9c27b0',
    desc: 'Upload EEG recordings for brain-wave classification and continuous/XOR/Polar views.',
    tag: 'Upload required',
  },
  {
    id: 'acoustic',
    label: 'Acoustic Signals',
    icon: '🎵',
    color: '#4ecdc4',
    desc: 'Doppler effect simulator, vehicle analysis, and drone detection — no file needed.',
    tag: 'Ready to use',
  },
  {
    id: 'stock',
    label: 'Stock Market',
    icon: '📈',
    color: '#45b7d1',
    desc: 'LSTM-based price forecasting for stocks, commodities, and currencies.',
    tag: 'Ready to use',
  },
  {
    id: 'microbiome',
    label: 'Microbiome',
    icon: '🦠',
    color: '#96ceb4',
    desc: 'Gut-health profiling, diversity trends, and longitudinal analysis of iHMP data.',
    tag: 'Ready to use',
  },
];

const LandingPage = ({ onModuleSelect }) => {
  return (
    <div className="landing-page">
      {/* Decorative blurred circles */}
      <div className="landing-bg-circle circle-1" />
      <div className="landing-bg-circle circle-2" />
      <div className="landing-bg-circle circle-3" />

      <div className="landing-content">
        <h1 className="landing-title">
          <span className="title-wave">📊</span>
          Signal Viewer Pro
        </h1>

        <p className="landing-subtitle">
          Explore, visualize, and analyze biomedical, acoustic, financial, and microbiome signals — powered by AI.
        </p>

        {/* ── Module Cards ── */}
        <div className="module-grid">
          {modules.map((m) => (
            <button
              key={m.id}
              className="module-card"
              style={{ '--accent': m.color }}
              onClick={() => onModuleSelect(m.id)}
            >
              <span className="module-icon">{m.icon}</span>
              <h3 className="module-label">{m.label}</h3>
              <p className="module-desc">{m.desc}</p>
              <span className="module-tag">{m.tag}</span>
            </button>
          ))}
        </div>

        {/* ── Highlights ── */}
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🔄</div>
            <h4>Multi-Channel Viewer</h4>
            <p>View multiple channels separately or overlaid</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🤖</div>
            <h4>AI Detection</h4>
            <p>Automatic abnormality detection using deep learning</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h4>Advanced Views</h4>
            <p>XOR, Polar, and Recurrence plots for pattern analysis</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h4>Real-time Controls</h4>
            <p>Play, pause, zoom, and pan through your signals</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
