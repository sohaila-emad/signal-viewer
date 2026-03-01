import React, { useState } from 'react';
import FileUploader from './components/Layout/FileUploader';
import ContinuousViewer from './components/Viewers/ContinuousViewer';
import AIPrediction from './components/Analysis/AIPrediction';
import LandingPage from './components/Layout/LandingPage';
import SignalInfo from './components/Layout/SignalInfo';
import XORViewer from './components/Viewers/XORViewer';
import PolarViewer from './components/Viewers/PolarViewer';
import RecurrenceViewer from './components/Viewers/RecurrenceViewer';
import AcousticPage from './components/Pages/AcousticPage';
import StockPage from './components/Pages/StockPage';
import MicrobiomePage from './components/Pages/MicrobiomePage';
import EEGPrediction from './components/Analysis/EEGPrediction';
import './App.css';

/* Label map used by the module-switcher dropdown */
const MODULE_LABELS = {
  medical:    '❤️  ECG',
  eeg:        '🧠  EEG',
  acoustic:   '🎵  Acoustic',
  stock:      '📈  Stock',
  microbiome: '🦠  Microbiome',
};

/* Tabs whose signal data comes from a file upload */
const UPLOAD_TABS = new Set(['medical', 'eeg']);

/* Tabs that support the advanced view buttons (XOR / Polar / Recurrence) */
const SIGNAL_VIEW_TABS = new Set(['medical', 'eeg']);


function App() {
  const [signalData, setSignalData] = useState(null);
  const [viewType, setViewType] = useState('continuous');
  const [activeTab, setActiveTab] = useState('medical');
  const [showLanding, setShowLanding] = useState(true);
  const [uploadHistory, setUploadHistory] = useState([]);

  /* ── data helpers ─────────────────────────────────────────────────────── */

  const handleDataLoaded = (data) => {
    console.log('Data loaded:', data);
    const historyEntry = {
      id: Date.now(),
      ...data,
      _loadTime: new Date().toLocaleTimeString(),
    };
    setUploadHistory((prev) => [historyEntry, ...prev].slice(0, 5));
    setSignalData(data);
    setShowLanding(false);
  };

  /* ── navigation ───────────────────────────────────────────────────────── */

  const handleModuleSelect = (moduleId) => {
    setActiveTab(moduleId);
    setViewType('continuous');
    if (UPLOAD_TABS.has(moduleId)) {
      // For upload-based modules, keep signal data if we already have some
      // of matching type; otherwise clear it so the sidebar is visible
      if (signalData?.type !== moduleId) setSignalData(null);
    }
    setShowLanding(false);
  };

  const handleBackToLanding = () => {
    setShowLanding(true);
    setSignalData(null);
  };

  return (
    <div className="app">
      {showLanding ? (
        <LandingPage onModuleSelect={handleModuleSelect} />
      ) : (
        <>
          {/* ═══════════════════  HEADER  ═══════════════════ */}
          <header className="app-header">
            <div className="header-left">
              <h1 className="logo" onClick={handleBackToLanding} style={{ cursor: 'pointer' }}>
                <span className="logo-icon">📊</span>
                Signal Viewer Pro
              </h1>

              {/* Module switcher */}
              <select
                className="module-switcher"
                value={activeTab}
                onChange={(e) => handleModuleSelect(e.target.value)}
              >
                {Object.entries(MODULE_LABELS).map(([id, label]) => (
                  <option key={id} value={id}>{label}</option>
                ))}
              </select>

              {/* Upload / Home button */}
              {UPLOAD_TABS.has(activeTab) ? (
                <button className="upload-btn" onClick={handleBackToLanding}>
                  <span className="btn-icon">📤</span>
                  Upload New File
                </button>
              ) : (
                <button className="upload-btn" onClick={handleBackToLanding}>
                  <span className="btn-icon">🏠</span>
                  Home
                </button>
              )}
            </div>

            {/* View-mode buttons — only for signal-based tabs */}
            {SIGNAL_VIEW_TABS.has(activeTab) && (
              <div className="view-selector">
                <button
                  className={`view-btn ${viewType === 'continuous' ? 'active' : ''}`}
                  onClick={() => setViewType('continuous')}
                >
                  <span className="btn-icon">📈</span> Continuous
                </button>
                <button
                  className={`view-btn ${viewType === 'xor' ? 'active' : ''}`}
                  onClick={() => setViewType('xor')}
                >
                  <span className="btn-icon">🔄</span> XOR
                </button>
                <button
                  className={`view-btn ${viewType === 'polar' ? 'active' : ''}`}
                  onClick={() => setViewType('polar')}
                >
                  <span className="btn-icon">⭕</span> Polar
                </button>
                <button
                  className={`view-btn ${viewType === 'recurrence' ? 'active' : ''}`}
                  onClick={() => setViewType('recurrence')}
                >
                  <span className="btn-icon">🔁</span> Recurrence
                </button>
              </div>
            )}
          </header>

          {/* ═══════════════════  MAIN  ═══════════════════ */}
          <main className="app-main">
            {/* Sidebar — only for upload-based tabs */}
            {UPLOAD_TABS.has(activeTab) && (
              <div className="sidebar">
                <div className="sidebar-section">
                  <h3 className="sidebar-title">
                    <span className="title-icon">📂</span>
                    Upload Signal
                  </h3>
                  <FileUploader
                    onDataLoaded={handleDataLoaded}
                    signalType={activeTab}
                  />
                </div>

                {signalData && (
                  <>
                    <SignalInfo data={signalData} />

                    {uploadHistory.length > 0 && (
                      <div className="sidebar-section history-section">
                        <h3 className="sidebar-title">
                          <span className="title-icon">⏱️</span>
                          Recent Uploads
                        </h3>
                        <div className="history-list">
                          {uploadHistory.map((entry) => (
                            <div
                              key={entry.id}
                              className={`history-item ${signalData.filename === entry.filename ? 'active' : ''}`}
                              onClick={() => setSignalData(entry)}
                            >
                              <div className="history-icon">
                                {entry.type === 'medical' ? '❤️' : entry.type === 'eeg' ? '🧠' : '📄'}
                              </div>
                              <div className="history-details">
                                <div className="history-name">{entry.filename}</div>
                                <div className="history-meta">
                                  <span>{entry.channels} ch</span>
                                  <span>·</span>
                                  <span>{entry.fs} Hz</span>
                                  <span>·</span>
                                  <span>{entry._loadTime}</span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* Viewer area */}
            <div className="viewer-area">
              {activeTab === 'acoustic'   && <AcousticPage />}
              {activeTab === 'stock'      && <StockPage />}
              {activeTab === 'microbiome' && <MicrobiomePage />}

              {activeTab === 'medical' && (
                signalData ? (
                  <div className="viewer-container">
                    <div className="stats-bar">
                      <div className="stat-item"><span className="stat-label">File</span><span className="stat-value">{signalData.filename}</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Channels</span><span className="stat-value">{signalData.channels}</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Sampling Rate</span><span className="stat-value">{signalData.fs} Hz</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Samples</span><span className="stat-value">{signalData.time?.length || 0}</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Type</span><span className="stat-value type-badge">{signalData.type || 'medical'}</span></div>
                    </div>

                    <div className="viewer-wrapper">
                      {viewType === 'continuous' && <ContinuousViewer data={signalData} />}
                      {viewType === 'xor' && <XORViewer data={signalData} />}
                      {viewType === 'polar' && <PolarViewer data={signalData} />}
                      {viewType === 'recurrence' && <RecurrenceViewer data={signalData} />}
                    </div>

                    <div className="ai-section">
                      <AIPrediction signalData={signalData} />
                    </div>
                  </div>
                ) : (
                  <div className="no-data">
                    <div className="no-data-icon">📊</div>
                    <h2>No data loaded</h2>
                    <p>Upload a file from the sidebar to start visualization</p>
                  </div>
                )
              )}

              {activeTab === 'eeg' && (
                signalData ? (
                  <div className="viewer-container">
                    <div className="stats-bar">
                      <div className="stat-item"><span className="stat-label">File</span><span className="stat-value">{signalData.filename}</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Channels</span><span className="stat-value">{signalData.channels}</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Sampling Rate</span><span className="stat-value">{signalData.fs} Hz</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Samples</span><span className="stat-value">{signalData.time?.length || 0}</span></div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item"><span className="stat-label">Type</span><span className="stat-value type-badge">EEG</span></div>
                    </div>

                    <div className="viewer-wrapper">
                      {viewType === 'continuous' && <ContinuousViewer data={signalData} />}
                      {viewType === 'xor' && <XORViewer data={signalData} />}
                      {viewType === 'polar' && <PolarViewer data={signalData} />}
                      {viewType === 'recurrence' && <RecurrenceViewer data={signalData} />}
                    </div>

                    <div className="ai-section">
                      <EEGPrediction signalData={signalData} />
                    </div>
                  </div>
                ) : (
                  <div className="no-data">
                    <div className="no-data-icon">🧠</div>
                    <h2>No EEG data loaded</h2>
                    <p>Upload an EEG file from the sidebar to start visualization</p>
                  </div>
                )
              )}
            </div>
          </main>
        </>
      )}
    </div>
  );
}

export default App;


