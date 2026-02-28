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


function App() {
  const [signalData, setSignalData] = useState(null);
  const [viewType, setViewType] = useState('continuous');
  const [activeTab, setActiveTab] = useState('medical');
  const [showLanding, setShowLanding] = useState(true);
  const [uploadHistory, setUploadHistory] = useState([]);

  const handleDataLoaded = (data) => {
    console.log('✅ Data loaded successfully:', data);
    
    // Add to upload history - store FULL data object
    const historyEntry = {
      id: Date.now(),
      ...data,  // Store entire response from backend
      _loadTime: new Date().toLocaleTimeString()  // Add load timestamp
    };
    
    setUploadHistory(prev => [historyEntry, ...prev].slice(0, 5)); // Keep last 5
    setSignalData(data);
    setShowLanding(false);
  };

  const handleFileSelected = async (file) => {
    // Accept either a single File or an array of Files
    const files = Array.isArray(file) ? file : [file];
    console.log('📁 File(s) selected from landing:', files.map(f => f.name).join(', '));

    const formData = new FormData();
    files.forEach(f => formData.append('file', f));
    formData.append('type', activeTab);

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      console.log('✅ Upload response:', data);
      
      if (data.error) {
        alert('Error: ' + data.error);
      } else {
        handleDataLoaded(data);
      }
    } catch (error) {
      console.error('❌ Upload error:', error);
      alert('Upload failed: ' + error.message);
    }
  };

  const handleSampleClick = (sampleType) => {
    console.log('Sample clicked:', sampleType);
    if (sampleType === 'Normal ECG' || sampleType === 'Arrhythmia') {
      setActiveTab('medical');
    } else if (sampleType === 'Car Passing') {
      setActiveTab('acoustic');
    } else if (sampleType === 'Stock Data') {
      setActiveTab('stock');
    }
    setShowLanding(false);
  };

  const handleBackToUpload = () => {
    setShowLanding(true);
    setSignalData(null);
  };

  // Microbiome has bundled data — no upload required; navigate directly.
  const handleSignalTypeChange = (type) => {
    setActiveTab(type);
    if (type === 'microbiome') {
      setShowLanding(false);
    }
  };

  return (
    <div className="app">
      {showLanding ? (
        <LandingPage
          onFileSelected={handleFileSelected}
          onSampleClick={handleSampleClick}
          signalType={activeTab}
          setSignalType={handleSignalTypeChange}
        />
      ) : (
        <>
          <header className="app-header">
            <div className="header-left">
              <h1 className="logo">
                <span className="logo-icon">📊</span>
                Signal Viewer Pro
              </h1>
              <button 
                className="upload-btn"
                onClick={handleBackToUpload}
              >
                <span className="btn-icon">📤</span>
                Upload New File
              </button>
            </div>
            
            <div className="view-selector">
              <button 
                className={`view-btn ${viewType === 'continuous' ? 'active' : ''}`}
                onClick={() => setViewType('continuous')}
              >
                <span className="btn-icon">📈</span>
                Continuous
              </button>
              <button 
                className={`view-btn ${viewType === 'xor' ? 'active' : ''}`}
                onClick={() => setViewType('xor')}
              >
                <span className="btn-icon">🔄</span>
                XOR
              </button>
              <button 
                className={`view-btn ${viewType === 'polar' ? 'active' : ''}`}
                onClick={() => setViewType('polar')}
              >
                <span className="btn-icon">⭕</span>
                Polar
              </button>
              <button 
                className={`view-btn ${viewType === 'recurrence' ? 'active' : ''}`}
                onClick={() => setViewType('recurrence')}
              >
                <span className="btn-icon">🔁</span>
                Recurrence
              </button>
            </div>
          </header>
          
          <main className="app-main">
            {/* Left Sidebar - hidden for microbiome (uses bundled data, no upload needed) */}
            {activeTab !== 'microbiome' && (
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
                  {/* Current File Info */}
                  <SignalInfo data={signalData} />

                  {/* Upload History */}
                  {uploadHistory.length > 0 && (
                    <div className="sidebar-section history-section">
                      <h3 className="sidebar-title">
                        <span className="title-icon">⏱️</span>
                        Recent Uploads
                      </h3>
                      <div className="history-list">
                        {uploadHistory.map(entry => (
                          <div
                            key={entry.id}
                            className={`history-item ${signalData.filename === entry.filename ? 'active' : ''}`}
                            onClick={() => setSignalData(entry)}
                          >
                            <div className="history-icon">
                              {entry.type === 'medical' ? '❤️' : entry.type === 'acoustic' ? '🎵' : '📈'}
                            </div>
                            <div className="history-details">
                              <div className="history-name">{entry.filename}</div>
                              <div className="history-meta">
                                <span>{entry.channels} ch</span>
                                <span>•</span>
                                <span>{entry.fs} Hz</span>
                                <span>•</span>
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
            
            {/* Main Viewer Area - Show different pages based on activeTab */}
            <div className="viewer-area">
              {/* Show AcousticPage when activeTab is acoustic */}
              {activeTab === 'acoustic' && <AcousticPage />}
              
              {/* Show StockPage when activeTab is stock */}
              {activeTab === 'stock' && <StockPage />}
              
              {/* Show MicrobiomePage when activeTab is microbiome */}
              {activeTab === 'microbiome' && <MicrobiomePage />}
              
              {/* Show standard medical viewers when activeTab is medical */}
              {activeTab === 'medical' && (
                signalData ? (
                  <div className="viewer-container">
                    {/* Quick Stats Bar */}
                    <div className="stats-bar">
                      <div className="stat-item">
                        <span className="stat-label">File</span>
                        <span className="stat-value">{signalData.filename}</span>
                      </div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item">
                        <span className="stat-label">Channels</span>
                        <span className="stat-value">{signalData.channels}</span>
                      </div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item">
                        <span className="stat-label">Sampling Rate</span>
                        <span className="stat-value">{signalData.fs} Hz</span>
                      </div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item">
                        <span className="stat-label">Samples</span>
                        <span className="stat-value">{signalData.time?.length || 0}</span>
                      </div>
                      <div className="stat-divider">|</div>
                      <div className="stat-item">
                        <span className="stat-label">Type</span>
                        <span className="stat-value type-badge">{signalData.type || 'medical'}</span>
                      </div>
                    </div>

                    {/* Viewer */}
                    <div className="viewer-wrapper">
                      {viewType === 'continuous' && <ContinuousViewer data={signalData} />}
                      {viewType === 'xor' && <XORViewer data={signalData} />}
                      {viewType === 'polar' && <PolarViewer data={signalData} />}
                      {viewType === 'recurrence' && <RecurrenceViewer data={signalData} />}
                    </div>

                    {/* AI Prediction (for medical signals) */}
                    {signalData && (
                      <div className="ai-section">
                        <AIPrediction signalData={signalData} />
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="no-data">
                    <div className="no-data-icon">📊</div>
                    <h2>No data loaded</h2>
                    <p>Upload a file from the sidebar to start visualization</p>
                    <button 
                      className="upload-prompt-btn"
                      onClick={handleBackToUpload}
                    >
                      Go to Upload Page
                    </button>
                  </div>
                )
              )}
              {/*  added eeg part here */}
              {activeTab === 'eeg' && (
  signalData ? (
    <div className="viewer-container">
      {/* Quick Stats Bar - same as medical */}
      <div className="stats-bar">
        <div className="stat-item">
          <span className="stat-label">File</span>
          <span className="stat-value">{signalData.filename}</span>
        </div>
        <div className="stat-divider">|</div>
        <div className="stat-item">
          <span className="stat-label">Channels</span>
          <span className="stat-value">{signalData.channels}</span>
        </div>
        <div className="stat-divider">|</div>
        <div className="stat-item">
          <span className="stat-label">Sampling Rate</span>
          <span className="stat-value">{signalData.fs} Hz</span>
        </div>
        <div className="stat-divider">|</div>
        <div className="stat-item">
          <span className="stat-label">Samples</span>
          <span className="stat-value">{signalData.time?.length || 0}</span>
        </div>
        <div className="stat-divider">|</div>
        <div className="stat-item">
          <span className="stat-label">Type</span>
          <span className="stat-value type-badge">EEG</span>
        </div>
      </div>

      {/* Viewer */}
      <div className="viewer-wrapper">
        {viewType === 'continuous' && <ContinuousViewer data={signalData} />}
        {viewType === 'xor' && <XORViewer data={signalData} />}
        {viewType === 'polar' && <PolarViewer data={signalData} />}
        {viewType === 'recurrence' && <RecurrenceViewer data={signalData} />}
      </div>

      {/* EEG Prediction */}
      <div className="ai-section">
        <EEGPrediction signalData={signalData} />
      </div>
    </div>
  ) : (
    <div className="no-data">
      <div className="no-data-icon">🧠</div>
      <h2>No EEG data loaded</h2>
      <p>Upload an EEG file from the sidebar to start visualization</p>
      <button className="upload-prompt-btn" onClick={handleBackToUpload}>
        Go to Upload Page
      </button>
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


