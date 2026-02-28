import React, { useState, useEffect } from 'react';
import { eegAPI } from '../../services/api';

const EEGPrediction = ({ signalData }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (signalData) {
      fetchPrediction();
    }
  }, [signalData]);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await eegAPI.predict(signalData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.message || 'Failed to get EEG prediction');
      console.error('EEG prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!signalData) return null;

  if (loading) {
    return (
      <div className="ai-section loading">
        <h3>🧠 EEG Analysis</h3>
        <p style={{ color: '#666' }}>Loading predictions...</p>
        <div style={{ fontSize: '24px' }}>⏳</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ai-section error">
        <h3>🧠 EEG Analysis</h3>
        <div style={{ color: '#d32f2f', background: '#ffebee', padding: '10px', borderRadius: '4px' }}>
          ⚠️ Error: {error}
        </div>
      </div>
    );
  }

  if (!prediction) return null;

  const biot = prediction.biot;
  const rf = prediction.random_forest;
  const comp = prediction.comparison;
  const predictionWarnings = prediction.warnings || [];

  const getColorForPrediction = (pred) => {
    if (pred === 'normal') return '#4CAF50';
    return '#ff9800'; // warning for any abnormality
  };

  const formatClassName = (name) => {
    // return name.replace(/_/g, ' ');
  if (!name) return 'unknown';
  return name.replace(/_/g, ' ');
  };

  return (
    <div className="ai-section">
      <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span>🧠</span> EEG Analysis (Ensemble)
      </h3>

      {predictionWarnings.length > 0 && (
        <div style={{
          background: '#fff8e1', border: '1px solid #ffc107', borderRadius: '4px',
          padding: '8px 12px', marginBottom: '10px', fontSize: '0.85em', color: '#5d4037'
        }}>
          {predictionWarnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
        </div>
      )}

      <div className="prediction-grid" style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '15px',
        marginBottom: '20px'
      }}>
        {/* Ensemble Prediction Card */}
        <div style={{
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '2px solid ' + getColorForPrediction(comp?.ensemble_prediction)
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
            🎯 Ensemble Prediction
          </div>
          <div style={{
            fontSize: '24px',
            fontWeight: 'bold',
            color: getColorForPrediction(comp?.ensemble_prediction),
            textTransform: 'capitalize'
          }}>
            {formatClassName(comp?.ensemble_prediction || 'unknown')}
          </div>
          <div style={{ marginTop: '8px', fontSize: '14px', color: '#2196F3' }}>
            Confidence: {(comp?.ensemble_confidence * 100).toFixed(1)}%
          </div>
        </div>

        {/* BIOT Card */}
        {biot && (
          <div style={{
            padding: '15px',
            backgroundColor: 'white',
            borderRadius: '8px',
            border: '1px solid #ddd'
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
              🧠 BIOT (Deep Learning)
            </div>
            <div style={{ fontSize: '18px', fontWeight: 'bold', textTransform: 'capitalize', marginBottom: '5px' }}>
              {formatClassName(biot.prediction)}
            </div>
            <div style={{ fontSize: '14px', color: '#2196F3' }}>
              {(biot.confidence * 100).toFixed(1)}% confidence
            </div>
            {biot.n_windows && (
              <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                {biot.n_windows} windows
              </div>
            )}
          </div>
        )}

        {/* Random Forest Card */}
        {rf && (
          <div style={{
            padding: '15px',
            backgroundColor: 'white',
            borderRadius: '8px',
            border: '1px solid #ddd'
          }}>
            <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
              🌳 Random Forest (Classical ML)
            </div>
            <div style={{ fontSize: '18px', fontWeight: 'bold', textTransform: 'capitalize', marginBottom: '5px' }}>
              {formatClassName(rf.prediction)}
            </div>
            <div style={{ fontSize: '14px', color: '#2196F3' }}>
              {(rf.confidence * 100).toFixed(1)}% confidence
            </div>
            {rf.n_windows && (
              <div style={{ fontSize: '12px', color: '#999', marginTop: '8px' }}>
                {rf.n_windows} windows
              </div>
            )}
          </div>
        )}
      </div>

      {/* Model Agreement */}
      {comp && (
        <div style={{
          padding: '15px',
          backgroundColor: comp.agreement ? '#e8f5e9' : '#fff3e0',
          borderRadius: '8px',
          border: '1px solid ' + (comp.agreement ? '#4CAF50' : '#ff9800'),
          marginBottom: '15px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '20px' }}>{comp.agreement ? '✓' : '⚠'}</span>
            <div>
              <div style={{ fontWeight: 'bold', color: comp.agreement ? '#2e7d32' : '#e65100' }}>
                {comp.agreement ?   'Models Agree' : 'Models Disagree'}
              </div>
              <div style={{ fontSize: '12px', color: '#555' }}>
                {comp.confidence_source}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Detailed probabilities (optional) */}
      {biot?.all_probabilities && (
        <details style={{ marginTop: '10px' }}>
          <summary style={{ cursor: 'pointer', color: '#666' }}>Show detailed probabilities</summary>
          <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            <div>
              <h4 style={{ margin: '0 0 5px 0', fontSize: '14px' }}>BIOT</h4>
              {Object.entries(biot.all_probabilities).map(([cls, prob]) => (
                <div key={cls} style={{ fontSize: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{formatClassName(cls)}</span>
                  <span>{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            <div>
              <h4 style={{ margin: '0 0 5px 0', fontSize: '14px' }}>Random Forest</h4>
              {rf?.all_probabilities && Object.entries(rf.all_probabilities).map(([cls, prob]) => (
                <div key={cls} style={{ fontSize: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{formatClassName(cls)}</span>
                  <span>{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </details>
      )}

      {/* Model info */}
      <div style={{
        marginTop: '15px',
        padding: '10px',
        backgroundColor: '#e3f2fd',
        borderRadius: '4px',
        fontSize: '12px',
        color: '#1565c0'
      }}>
        ✓ Using real trained EEG models (BIOT + Random Forest ensemble)
      </div>
    </div>
  );
};

export default EEGPrediction;
