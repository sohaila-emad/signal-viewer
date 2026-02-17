import React from 'react';

const AIPrediction = ({ signalData }) => {
  if (!signalData) {
    return null;
  }

  // Mock AI prediction - in a real app, this would call the backend API
  const mockPrediction = {
    prediction: 'Normal',
    confidence: 0.85,
    analysis: 'Signal appears normal with typical waveform patterns',
    recommendations: [
      'Continue regular monitoring',
      'No immediate action required'
    ]
  };

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
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '15px'
      }}>
        {/* Prediction Card */}
        <div style={{
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '1px solid #ddd'
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
            Prediction
          </div>
          <div style={{ 
            fontSize: '24px', 
            fontWeight: 'bold',
            color: mockPrediction.prediction === 'Normal' ? '#4CAF50' : '#f44336'
          }}>
            {mockPrediction.prediction}
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
            {(mockPrediction.confidence * 100).toFixed(1)}%
          </div>
        </div>

        {/* Analysis Card */}
        <div style={{
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '1px solid #ddd',
          gridColumn: '1 / -1'
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '5px' }}>
            Analysis
          </div>
          <div style={{ fontSize: '14px', color: '#333' }}>
            {mockPrediction.analysis}
          </div>
        </div>

        {/* Recommendations */}
        <div style={{
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '1px solid #ddd',
          gridColumn: '1 / -1'
        }}>
          <div style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
            Recommendations
          </div>
          <ul style={{ margin: 0, paddingLeft: '20px' }}>
            {mockPrediction.recommendations.map((rec, index) => (
              <li key={index} style={{ marginBottom: '5px' }}>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Disclaimer */}
      <div style={{ 
        marginTop: '15px', 
        padding: '10px', 
        backgroundColor: '#fff3cd', 
        borderRadius: '4px',
        fontSize: '12px',
        color: '#856404'
      }}>
        ⚠️ This is a demo AI prediction. For actual medical decisions, please consult a healthcare professional.
      </div>
    </div>
  );
};

export default AIPrediction;
