import React, { useState, useEffect } from 'react';
import { acousticAPI } from '../../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const AcousticPage = () => {
  const [activeMode, setActiveMode] = useState('doppler'); // 'doppler', 'vehicle', 'drone'

  // Doppler state
  const [velocity, setVelocity] = useState(30);
  const [frequency, setFrequency] = useState(440);
  const [dopplerParams, setDopplerParams] = useState(null);
  const [generatedAudio, setGeneratedAudio] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  // Vehicle analysis state
  const [vehicleResult, setVehicleResult] = useState(null);
  const [analyzingVehicle, setAnalyzingVehicle] = useState(false);

  // Drone detection state
  const [detectionResult, setDetectionResult] = useState(null);
  const [detecting, setDetecting] = useState(false);

  // Spectrogram state
  const [spectrogramData, setSpectrogramData] = useState(null);

  useEffect(() => {
    // Get Doppler parameters when velocity or frequency changes
    const fetchDopplerParams = async () => {
      try {
        const response = await acousticAPI.getDopplerParameters(velocity, frequency);
        setDopplerParams(response.data);
      } catch (error) {
        console.error('Error fetching Doppler parameters:', error);
      }
    };

    if (activeMode === 'doppler') {
      fetchDopplerParams();
    }
  }, [velocity, frequency, activeMode]);

  const handleGenerateDoppler = async () => {
    try {
      setGeneratedAudio(null);
      const response = await acousticAPI.generateDoppler({
        velocity,
        frequency,
        duration: 5.0,
        sample_rate: 44100
      });
      setGeneratedAudio(response.data);
      console.log(response.data);

    } catch (error) {
      console.error('Error generating Doppler:', error);
      alert('Error generating Doppler effect: ' + error.message);
    }
  };

  const handleAnalyzeVehicle = async () => {
    setAnalyzingVehicle(true);
    try {

      const arrayBuffer = await audioFile.arrayBuffer();
      const audioContext = new AudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      const audioData = Array.from(audioBuffer.getChannelData(0));
      const analysis = await acousticAPI.analyzeVehicle(audioData, 44100);
      setVehicleResult(analysis.data);


    } catch (error) {
      console.error('Error analyzing vehicle:', error);
    } finally {
      setAnalyzingVehicle(false);
    }
  };

  const handleDetectDrone = async () => {
    setDetecting(true);
    try {
      const arrayBuffer = await audioFile.arrayBuffer();
      const audioData = Array.from(new Float32Array(arrayBuffer));
      const detection = await acousticAPI.detectVehicle(
        audioData,
        44100,
        'auto'
      );
      setDetectionResult(detection.data);

      // Also compute spectrogram
      const spectrogram = await acousticAPI.computeSpectrogram(
        audioData,
        44100,
        256
      );
      setSpectrogramData(spectrogram.data);

    } catch (error) {
      console.error('Error detecting drone:', error);
    } finally {
      setDetecting(false);
    }
  };

  return (
    <div className="acoustic-page" style={{ padding: '20px', height: '100%', overflow: 'auto' }}>
      <h1 style={{ marginBottom: '20px' }}>🎵 Acoustic Signal Processing</h1>

      {/* Mode Selector */}
      <div style={{
        display: 'flex',
        gap: '10px',
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px'
      }}>
        <button
          onClick={() => setActiveMode('doppler')}
          style={{
            padding: '10px 20px',
            backgroundColor: activeMode === 'doppler' ? '#4ecdc4' : '#ddd',
            color: activeMode === 'doppler' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          🚗 Doppler Effect
        </button>
        <button
          onClick={() => setActiveMode('vehicle')}
          style={{
            padding: '10px 20px',
            backgroundColor: activeMode === 'vehicle' ? '#4ecdc4' : '#ddd',
            color: activeMode === 'vehicle' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          📊 Vehicle Analysis
        </button>
        <button
          onClick={() => setActiveMode('drone')}
          style={{
            padding: '10px 20px',
            backgroundColor: activeMode === 'drone' ? '#4ecdc4' : '#ddd',
            color: activeMode === 'drone' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          🚁 Drone Detection
        </button>
      </div>

      {/* Doppler Effect Section */}
      {activeMode === 'doppler' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Controls */}
          <div style={{
            padding: '20px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>🚗 Doppler Effect Generator</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Generate synthetic vehicle passing sound with Doppler effect
            </p>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Vehicle Velocity (v): {velocity} m/s
              </label>
              <input
                type="range"
                min="1"
                max="100"
                value={velocity}
                onChange={(e) => setVelocity(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: '0.8rem', color: '#666' }}>
                Typical values: 10-50 m/s (36-180 km/h)
              </div>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Horn Frequency (f): {frequency} Hz
              </label>
              <input
                type="range"
                min="100"
                max="2000"
                value={frequency}
                onChange={(e) => setFrequency(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: '0.8rem', color: '#666' }}>
                Typical values: 200-1000 Hz
              </div>
            </div>

            <button
              onClick={handleGenerateDoppler}
              style={{
                padding: '12px 24px',
                backgroundColor: '#4ecdc4',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '1rem'
              }}
            >
              🔊 Generate Sound
            </button>
            {generatedAudio && generatedAudio.audio_base64 && (
              <div>
                <h3>Generated Doppler Sound</h3>
                <audio controls>
                  <source
                    src={`data:audio/wav;base64,${generatedAudio.audio_base64}`}
                    type="audio/wav"
                  />
                  Your browser does not support the audio element.
                </audio>
              </div>
            )}

          </div>

          {/* Results */}
          <div style={{
            padding: '20px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>


            {dopplerParams ? (
              <div>

                {generatedAudio && (
                  <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#e8f5e9', borderRadius: '4px' }}>
                    <h4>✅ Audio Generated!</h4>
                    <p>Duration: {generatedAudio.duration?.toFixed(2)} seconds</p>
                    <p>Sample Rate: {generatedAudio.sample_rate} Hz</p>
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: '#666' }}>Adjust parameters to see Doppler effect</p>
            )}

            {/* Formula explanation */}

          </div>
        </div>
      )}

      {/* Vehicle Analysis Section */}
      {activeMode === 'vehicle' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{
            padding: '20px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>📊 Vehicle Sound Analysis</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Analyze vehicle passing sound to estimate velocity and frequency
            </p>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Upload Audio File
              </label>
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setAudioFile(e.target.files[0])}
              />
              {!audioFile && (
                <p style={{ color: '#e74c3c', fontSize: '0.85rem', marginTop: '5px' }}>
                  No file selected. Please upload an audio file to analyze.
                </p>
              )}
            </div>

            <button
              onClick={handleAnalyzeVehicle}
              disabled={analyzingVehicle || !audioFile}
              style={{
                padding: '12px 24px',
                backgroundColor: analyzingVehicle ? '#ccc' : '#45b7d1',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: analyzingVehicle ? 'not-allowed' : 'pointer',
                fontWeight: 'bold',
                fontSize: '1rem'
              }}
            >
              {analyzingVehicle ? '⏳ Analyzing...' : '🔍 Analyze Vehicle Sound'}
            </button>
          </div>

          <div style={{
            padding: '20px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>📈 Analysis Results</h3>

            {vehicleResult ? (
              <div>
                <div style={{ marginBottom: '15px' }}>
                  <strong>Estimated Velocity:</strong> {vehicleResult.estimated_velocity?.toFixed(2)} m/s
                </div>
                <div style={{ marginBottom: '15px' }}>
                  <strong>Estimated Frequency:</strong> {vehicleResult.estimated_horn_frequency?.toFixed(2)} Hz
                </div>
                <div style={{ marginBottom: '15px' }}>
                  <strong>Confidence:</strong> {((vehicleResult.confidence || 0) * 100).toFixed(1)}%
                </div>
        
              </div>
            ) : (
              <p style={{ color: '#666' }}>
                Click "Analyze Vehicle Sound" to estimate velocity and frequency from the generated audio
              </p>
            )}
          </div>
        </div>
      )}

      {/* Drone Detection Section */}
      {activeMode === 'drone' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{
            padding: '20px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>🚁 Drone/Submarine Detection</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Detect drone or submarine sounds in audio using spectral analysis
            </p>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Upload Audio File
              </label>
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setAudioFile(e.target.files[0])}
              />
              {!audioFile && (
                <p style={{ color: '#e74c3c', fontSize: '0.85rem', marginTop: '5px' }}>
                  No file selected. Please upload an audio file to analyze.
                </p>
              )}
            </div>

            <button
              onClick={handleDetectDrone}
              disabled={detecting || !audioFile}
              style={{
                padding: '12px 24px',
                backgroundColor: detecting ? '#ccc' : '#96ceb4',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: detecting ? 'not-allowed' : 'pointer',
                fontWeight: 'bold',
                fontSize: '1rem'
              }}
            >
              {detecting ? '⏳ Detecting...' : '🔍 Detect Vehicle'}
            </button>
          </div>

          <div style={{
            padding: '20px',
            backgroundColor: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>📡 Detection Results</h3>

            {detectionResult ? (
              <div>
                <div style={{
                  padding: '15px',
                  backgroundColor: detectionResult.detection === 'drone' ? '#e8f5e9' : '#fff3e0',
                  borderRadius: '4px',
                  marginBottom: '15px'
                }}>
                  <strong>Detection:</strong> {detectionResult.detection?.toUpperCase() || 'Unknown'}
                </div>

                {detectionResult.drone_score !== undefined && (
                  <div style={{ marginBottom: '10px' }}>
                    <strong>Drone Score:</strong> {(detectionResult.drone_score * 100).toFixed(1)}%
                    <div style={{
                      width: '100%',
                      height: '20px',
                      backgroundColor: '#eee',
                      borderRadius: '10px',
                      overflow: 'hidden',
                      marginTop: '5px'
                    }}>
                      <div style={{
                        width: `${detectionResult.drone_score * 100}%`,
                        height: '100%',
                        backgroundColor: '#4ecdc4',
                        transition: 'width 0.3s'
                      }} />
                    </div>
                  </div>
                )}

                {detectionResult.submarine_score !== undefined && (
                  <div style={{ marginBottom: '10px' }}>
                    <strong>Submarine Score:</strong> {(detectionResult.submarine_score * 100).toFixed(1)}%
                    <div style={{
                      width: '100%',
                      height: '20px',
                      backgroundColor: '#eee',
                      borderRadius: '10px',
                      overflow: 'hidden',
                      marginTop: '5px'
                    }}>
                      <div style={{
                        width: `${detectionResult.submarine_score * 100}%`,
                        height: '100%',
                        backgroundColor: '#45b7d1',
                        transition: 'width 0.3s'
                      }} />
                    </div>
                  </div>
                )}

                {detectionResult.confidence && (
                  <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
                    <strong>Confidence:</strong> {(detectionResult.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: '#666' }}>
                Click "Detect Vehicle" to analyze audio for drone/submarine signatures
              </p>
            )}
          </div>
        </div>
      )}

      {/* Spectrogram Display */}
      {spectrogramData && activeMode === 'drone' && (
        <div style={{
          marginTop: '20px',
          padding: '20px',
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h3>📊 Spectrogram</h3>
          <div style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={spectrogramData.frequencies?.map((f, i) => ({ frequency: f, value: spectrogramData.spectrogram?.[0]?.[i] || 0 }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="frequency" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#4ecdc4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

export default AcousticPage;
