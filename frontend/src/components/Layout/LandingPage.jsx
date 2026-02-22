import React, { useState, useEffect } from 'react';
import './LandingPage.css';

const LandingPage = ({ onFileSelected, onSampleClick, signalType, setSignalType }) => {
  const [dragActive, setDragActive] = useState(false);
  const [previewPoints, setPreviewPoints] = useState([]);

  // Generate animated ECG preview
  useEffect(() => {
    const generatePreview = () => {
      const points = [];
      for (let i = 0; i < 100; i++) {
        const x = i;
        // Create a realistic ECG pattern with P, Q, R, S, T waves
        let y = 50;
        if (i % 25 === 10) y = 30; // P wave
        if (i % 25 === 15) y = 20; // Q wave
        if (i % 25 === 16) y = 80; // R wave (peak)
        if (i % 25 === 17) y = 30; // S wave
        if (i % 25 === 22) y = 45; // T wave
        else y = 50 + Math.sin(i * 0.5) * 5;
        
        points.push({ x, y });
      }
      setPreviewPoints(points);
    };

    generatePreview();
    const interval = setInterval(generatePreview, 2000); // Animate every 2 seconds
    return () => clearInterval(interval);
  }, []);

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Handle drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      console.log('File dropped:', e.dataTransfer.files[0].name);
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  // Handle file input
  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);
      console.log(`Files selected: ${files.map(f => f.name).join(', ')}`);
      // Send all files at once
      for (let file of files) {
        onFileSelected(file);
      }
    }
  };

  // Signal type icons
  const signalTypes = [
    { id: 'medical', label: 'Medical ECG/EEG', icon: '❤️', color: '#ff6b6b' },
    { id: 'acoustic', label: 'Acoustic Sounds', icon: '🎵', color: '#4ecdc4' },
    { id: 'stock', label: 'Stock Market', icon: '📈', color: '#45b7d1' },
    { id: 'microbiome', label: 'Microbiome', icon: '🦠', color: '#96ceb4' }
  ];

  return (
    <div className="landing-page">
      {/* Animated Background Preview */}
      <div className="preview-background">
        <svg className="preview-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
          <path
            d={`M ${previewPoints.map(p => `${p.x},${p.y}`).join(' L ')}`}
            stroke="rgba(76, 175, 80, 0.2)"
            strokeWidth="2"
            fill="none"
          />
        </svg>
      </div>

      <div className="landing-content">
        <h1 className="landing-title">
          <span className="title-wave">📊</span> 
          Signal Viewer Pro
          <span className="title-wave">📈</span>
        </h1>
        
        <p className="landing-subtitle">
          Upload your signals and let AI analyze them in real-time
        </p>

        {/* Signal Type Selector */}
        <div className="signal-type-selector">
          {signalTypes.map(type => (
            <button
              key={type.id}
              className={`type-button ${signalType === type.id ? 'active' : ''}`}
              onClick={() => setSignalType(type.id)}
              style={{ '--type-color': type.color }}
            >
              <span className="type-icon">{type.icon}</span>
              <span className="type-label">{type.label}</span>
            </button>
          ))}
        </div>

        {/* Drag & Drop Area */}
        <div
          className={`upload-area ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-input"
            onChange={handleFileInput}
            multiple
            accept=".csv,.txt,.wav,.mp3,.edf,.hea,.dat,.xlsx,.json"
            style={{ display: 'none' }}
          />
          
          <div className="upload-content">
            <div className="upload-icon">📤</div>
            <h3>Drag & Drop your file here</h3>
            <p>or</p>
            <button 
              className="browse-button"
              onClick={() => document.getElementById('file-input').click()}
            >
              Browse Files
            </button>
            <p className="file-hint">
              Supported formats: CSV, TXT, WAV, MP3, EDF, XLSX, JSON
            </p>
          </div>
        </div>

        {/* Features Grid */}
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
            <h4>XOR & Polar Views</h4>
            <p>Advanced visualization modes for pattern analysis</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h4>Real-time Controls</h4>
            <p>Play, pause, zoom, and pan through your signals</p>
          </div>
        </div>

        {/* Sample Data Section */}
        <div className="sample-section">
          <h3>Try with sample data</h3>
          <div className="sample-buttons">
            <button 
              className="sample-button"
              onClick={() => onSampleClick('Normal ECG')}
            >
              <span>❤️</span> Normal ECG
            </button>
            <button 
              className="sample-button"
              onClick={() => onSampleClick('Arrhythmia')}
            >
              <span>⚡</span> Arrhythmia
            </button>
            <button 
              className="sample-button"
              onClick={() => onSampleClick('Car Passing')}
            >
              <span>🎵</span> Car Passing
            </button>
            <button 
              className="sample-button"
              onClick={() => onSampleClick('Stock Data')}
            >
              <span>📈</span> Stock Data
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
