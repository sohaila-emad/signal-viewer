import React, { useState, useEffect } from 'react';
import api from '../../services/api';

const AIPrediction = ({ signalData }) => {
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
      const response = await api.post('/medical/predict', signalData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.message || 'Failed to get prediction');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!signalData) {
    return null;
  }

  if (loading) {
    return (
      <div style={{
        padding: '20px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px',
        marginTop: '20px',
        textAlign: 'center'
      }}>
        <h3>🤖 AI Analysis</h3>
        <p style={{ color: '#666' }}>Loading predictions...</p>
        <div style={{ fontSize: '24px' }}>⏳</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '20px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px',
        marginTop: '20px'
      }}>
        <h3 style={{ margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span>🤖</span> AI Analysis
        </h3>
        <div style={{
          padding: '15px',
          backgroundColor: '#ffebee',
          borderRadius: '8px',
          border: '1px solid #ef5350',
          color: '#d32f2f'
        }}>
          ⚠️ Error: {error}
        </div>
      </div>
    );
  }

  // Check if using real models or synthetic
  const usingRealModels = prediction?.using_real_models;
  const isEnsemble = prediction?.using_real_models;

  let mainPrediction, mainConfidence, modelSource;

  if (usingRealModels) {
    // Use ensemble prediction
    const comparison = prediction?.comparison;
    mainPrediction = comparison?.ensemble_prediction || 'Unknown';
    mainConfidence = comparison?.ensemble_confidence || 0;
    modelSource = 'Real Models (Ensemble)';
  } else {
    // Use synthetic prediction
    mainPrediction = prediction?.synthetic_ai?.prediction || 'Unknown';
    mainConfidence = prediction?.synthetic_ai?.confidence || 0;
    modelSource = 'Synthetic (Demo)';
  }

  const getColorForPrediction = (pred) => {
    if (pred === 'normal') return '#4CAF50';
    return '#ff9800'; // warning color for any abnormality
  };

  const getRecommendations = (pred) => {
    const recs = {
      'normal': ['Continue regular monitoring', 'No immediate action required'],
      'bradycardia': ['Monitor heart rate carefully', 'Consider consulting with a cardiologist'],
      'atrial_fibrillation': ['Seek medical attention', 'ECG confirmation recommended'],
      'ventricular_tachycardia': ['🚨 URGENT: Seek immediate medical attention', 'Emergency evaluation needed'],
      'premature_ventricular_contraction': ['Consult with a cardiologist', 'Further evaluation may be needed']
    };
    return recs[pred] || ['Consult a healthcare professional'];
  };

  return (
    <div style={{
      padding: '20px',
      backgroundColor: '#f5f5f5',
      borderRadius: '8px',
      marginTop: '20px'
    }}>
      <h3 style={{ margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span>🤖</span> AI Analysis ({modelSource})
      </h3>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '15px'
      }}>
        {/* Ensemble/Main Prediction Card */}
        <div style={{
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '2px solid ' + getColorForPrediction(mainPrediction)
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
            {isEnsemble ? '🎯 Ensemble Prediction' : 'Prediction'}
          </div>
          <div style={{
            fontSize: '24px',
            fontWeight: 'bold',
            color: getColorForPrediction(mainPrediction)
          }}>
            {mainPrediction?.replace(/_/g, ' ').toUpperCase()}
          </div>
        </div>

        {/* Confidence Card */}
        <div style={{
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '1px solid #ddd'
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
            Confidence
          </div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2196F3' }}>
            {(mainConfidence * 100).toFixed(1)}%
          </div>
        </div>

        {/* Real Models Comparison (if available) */}
        {usingRealModels && prediction?.comparison && (
          <>
            <div style={{
              padding: '15px',
              backgroundColor: 'white',
              borderRadius: '8px',
              border: '1px solid #ddd'
            }}>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                🧠 ECGNet (Deep Learning)
              </div>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>
                {prediction?.ecgnet?.prediction?.replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: '12px', color: '#2196F3' }}>
                {(prediction?.ecgnet?.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>

            <div style={{
              padding: '15px',
              backgroundColor: 'white',
              borderRadius: '8px',
              border: '1px solid #ddd'
            }}>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                🌳 Classical ML (Random Forest)
              </div>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '5px' }}>
                {prediction?.classical_ml?.prediction?.replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: '12px', color: '#2196F3' }}>
                {(prediction?.classical_ml?.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>

            {/* Agreement Status */}
            <div style={{
              padding: '15px',
              backgroundColor: prediction?.comparison?.agreement ? '#e8f5e9' : '#fff3e0',
              borderRadius: '8px',
              border: '1px solid ' + (prediction?.comparison?.agreement ? '#4CAF50' : '#ff9800'),
              gridColumn: '1 / -1'
            }}>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
                📊 Model Agreement
              </div>
              <div style={{
                fontSize: '14px',
                fontWeight: 'bold',
                color: prediction?.comparison?.agreement ? '#2e7d32' : '#e65100'
              }}>
                {prediction?.comparison?.agreement ? '✓ Models Agree' : '⚠ Models Disagree'}
              </div>
              <div style={{ fontSize: '12px', marginTop: '5px', color: '#555' }}>
                {prediction?.comparison?.confidence_source}
              </div>
            </div>
          </>
        )}

        {/* Recommendations */}
        <div style={{
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '1px solid #ddd',
          gridColumn: '1 / -1'
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
            💡 Recommendations
          </div>
          <ul style={{ margin: 0, paddingLeft: '20px' }}>
            {getRecommendations(mainPrediction).map((rec, index) => (
              <li key={index} style={{ marginBottom: '5px', color: '#333' }}>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Model Info */}
      <div style={{
        marginTop: '15px',
        padding: '10px',
        backgroundColor: usingRealModels ? '#e3f2fd' : '#fff3cd',
        borderRadius: '4px',
        fontSize: '12px',
        color: usingRealModels ? '#1565c0' : '#856404'
      }}>
        {usingRealModels ? (
          '✓ Using real trained models (ECGNet + Classical ML ensemble)'
        ) : (
          '⚠️ Using synthetic predictions (real models not available)'
        )}
        <br />
        {usingRealModels ? '🔬 For validation or research purposes' : '👨‍⚕️ This is a demo prediction. For actual medical decisions, consult a healthcare professional.'}
      </div>
    </div>
  );
};

export default AIPrediction;
