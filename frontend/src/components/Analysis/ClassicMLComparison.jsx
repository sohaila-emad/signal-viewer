import React, { useState, useEffect } from 'react';
import api from '../../services/api';

const ClassicMLComparison = ({ signalData }) => {
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (signalData) {
      fetchComparison();
    }
  }, [signalData]);

  const fetchComparison = async () => {
    setLoading(true);
    try {
      const response = await api.post('/medical/predict', signalData);
      setComparison(response.data);
    } catch (err) {
      console.error('Comparison error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!signalData || loading) {
    return null;
  }

  if (!comparison?.using_real_models) {
    return null;
  }

  const comp = comparison.comparison;
  if (!comp) return null;

  const ModelCard = ({ title, icon, pred, conf, probs }) => (
    <div style={{
      padding: '20px',
      backgroundColor: 'white',
      borderRadius: '8px',
      border: '2px solid #2196F3',
      flex: 1,
      minWidth: '250px'
    }}>
      <h4 style={{ margin: '0 0 15px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ fontSize: '20px' }}>{icon}</span>
        {title}
      </h4>

      <div style={{ marginBottom: '15px' }}>
        <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
          Prediction
        </div>
        <div style={{
          fontSize: '18px',
          fontWeight: 'bold',
          color: '#1976d2',
          textTransform: 'capitalize'
        }}>
          {pred?.replace(/_/g, ' ')}
        </div>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
          Confidence
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            fontSize: '16px',
            fontWeight: 'bold',
            color: '#4CAF50'
          }}>
            {(conf * 100).toFixed(1)}%
          </div>
          <div style={{
            width: '100px',
            height: '8px',
            backgroundColor: '#e0e0e0',
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              width: `${conf * 100}%`,
              backgroundColor: '#4CAF50',
              transition: 'width 0.3s'
            }} />
          </div>
        </div>
      </div>

      {probs && Object.keys(probs).length > 0 && (
        <div>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
            Class Probabilities
          </div>
          <div style={{ fontSize: '12px' }}>
            {Object.entries(probs).map(([cls, prob]) => (
              <div key={cls} style={{
                marginBottom: '4px',
                display: 'flex',
                justifyContent: 'space-between'
              }}>
                <span style={{ textTransform: 'capitalize' }}>
                  {cls.replace(/_/g, ' ')}
                </span>
                <span style={{ fontWeight: 'bold', color: '#2196F3' }}>
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div style={{
      padding: '20px',
      backgroundColor: '#f5f5f5',
      borderRadius: '8px',
      marginTop: '20px'
    }}>
      <h3 style={{ margin: '0 0 20px 0' }}>
        📊 Model Comparison
      </h3>

      {/* Side-by-side comparison */}
      <div style={{
        display: 'flex',
        gap: '20px',
        marginBottom: '20px',
        flexWrap: 'wrap'
      }}>
        <ModelCard
          title="ECGNet (Deep Learning)"
          icon="🧠"
          pred={comp.ecgnet_prediction}
          conf={comp.ecgnet_confidence}
          probs={comparison.ecgnet?.all_probabilities}
        />

        <ModelCard
          title="Classical ML (Random Forest)"
          icon="🌳"
          pred={comp.classical_prediction}
          conf={comp.classical_confidence}
          probs={comparison.classical_ml?.all_probabilities}
        />
      </div>

      {/* Agreement Status */}
      <div style={{
        padding: '15px',
        backgroundColor: comp.agreement ? '#e8f5e9' : '#fff3e0',
        borderRadius: '8px',
        border: '2px solid ' + (comp.agreement ? '#4CAF50' : '#ff9800')
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span style={{ fontSize: '24px' }}>
            {comp.agreement ? '✓' : '⚠'}
          </span>
          <div>
            <div style={{
              fontSize: '14px',
              fontWeight: 'bold',
              color: comp.agreement ? '#2e7d32' : '#e65100'
            }}>
              {comp.agreement ? 'Models Agree' : 'Models Disagree'}
            </div>
            <div style={{
              fontSize: '12px',
              color: comp.agreement ? '#558b2f' : '#bf360c',
              marginTop: '4px'
            }}>
              {comp.confidence_source}
            </div>
          </div>
        </div>
      </div>

      {/* Ensemble Recommendation */}
      <div style={{
        marginTop: '15px',
        padding: '15px',
        backgroundColor: 'white',
        borderRadius: '8px',
        border: '2px solid #673AB7'
      }}>
        <div style={{
          fontSize: '12px',
          color: '#666',
          marginBottom: '5px'
        }}>
          🎯 Ensemble Prediction (Recommended)
        </div>
        <div style={{
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#673AB7',
          textTransform: 'capitalize',
          marginBottom: '8px'
        }}>
          {comp.ensemble_prediction?.replace(/_/g, ' ')}
        </div>
        <div style={{
          fontSize: '14px',
          color: '#555'
        }}>
          Confidence: <strong>{(comp.ensemble_confidence * 100).toFixed(1)}%</strong>
        </div>
      </div>

      {/* Information */}
      <div style={{
        marginTop: '15px',
        padding: '10px',
        backgroundColor: '#e3f2fd',
        borderRadius: '4px',
        fontSize: '12px',
        color: '#1565c0'
      }}>
        💡 The ensemble combines both models for more robust predictions. When models agree (✓), confidence is higher.
      </div>
    </div>
  );
};

export default ClassicMLComparison;
