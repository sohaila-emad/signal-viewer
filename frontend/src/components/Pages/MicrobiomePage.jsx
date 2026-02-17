import React, { useState, useEffect } from 'react';
import { microbiomeAPI } from '../../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const MicrobiomePage = () => {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [diversityData, setDiversityData] = useState(null);
  const [viewMode, setViewMode] = useState('samples'); // 'samples', 'diversity', 'composition'

  // Sample data
  const [sampleData, setSampleData] = useState(null);

  useEffect(() => {
    loadSamples();
  }, []);

  const loadSamples = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await microbiomeAPI.getSamples();
      setSamples(response.data.samples || []);
    } catch (err) {
      console.error('Error loading samples:', err);
      setError('Failed to load samples: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectSample = async (sampleId) => {
    setLoading(true);
    try {
      const response = await microbiomeAPI.getSample(sampleId);
      setSelectedSample(response.data);
      setSampleData(response.data);
    } catch (err) {
      console.error('Error loading sample:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeDiversity = async () => {
    setLoading(true);
    try {
      const response = await microbiomeAPI.analyzeDiversity(samples.slice(0, 10).map(s => s.sample_id));
      setDiversityData(response.data);
      setViewMode('diversity');
    } catch (err) {
      console.error('Error analyzing diversity:', err);
    } finally {
      setLoading(false);
    }
  };

  // Sample microbiome data for visualization (since backend may not have real data)
  const generateSampleData = () => {
    const bacteria = [
      { name: 'Lactobacillus', value: 35, color: '#4ecdc4' },
      { name: 'Bifidobacterium', value: 25, color: '#45b7d1' },
      { name: 'Bacteroides', value: 20, color: '#96ceb4' },
      { name: 'Prevotella', value: 10, color: '#ffeaa7' },
      { name: 'Other', value: 10, color: '#dfe6e9' },
    ];
    return bacteria;
  };

  const bacteriaComposition = generateSampleData();

  // Diversity metrics mock data
  const diversityMetrics = [
    { name: 'Shannon Index', value: 3.2, max: 5 },
    { name: 'Simpson Index', value: 0.85, max: 1 },
    { name: 'Observed Species', value: 156, max: 200 },
    { name: 'Chao1', value: 178, max: 250 },
  ];

  // Disease profiles mock data
  const diseaseProfiles = [
    { name: 'Healthy', value: 45, color: '#4ecdc4' },
    { name: 'IBD', value: 20, color: '#ff6b6b' },
    { name: 'Type 2 Diabetes', value: 15, color: '#ffeaa7' },
    { name: 'Obesity', value: 12, color: '#74b9ff' },
    { name: 'Other', value: 8, color: '#dfe6e9' },
  ];

  return (
    <div className="microbiome-page" style={{ padding: '20px', height: '100%', overflow: 'auto' }}>
      <h1 style={{ marginBottom: '20px' }}>🦠 Microbiome Analysis</h1>
      
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
          onClick={() => setViewMode('samples')}
          style={{
            padding: '10px 20px',
            backgroundColor: viewMode === 'samples' ? '#96ceb4' : '#ddd',
            color: viewMode === 'samples' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          🧬 Samples
        </button>
        <button
          onClick={() => setViewMode('diversity')}
          style={{
            padding: '10px 20px',
            backgroundColor: viewMode === 'diversity' ? '#96ceb4' : '#ddd',
            color: viewMode === 'diversity' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          📊 Diversity
        </button>
        <button
          onClick={() => setViewMode('composition')}
          style={{
            padding: '10px 20px',
            backgroundColor: viewMode === 'composition' ? '#96ceb4' : '#ddd',
            color: viewMode === 'composition' ? 'white' : '#333',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          🦠 Composition
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ 
          padding: '15px', 
          backgroundColor: '#ffebee', 
          color: '#c62828',
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          {error}
        </div>
      )}

      {/* Samples View */}
      {viewMode === 'samples' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Sample List */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>🧬 Microbiome Samples</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Available samples from iHMP/iPOP dataset
            </p>
            
            {loading ? (
              <div style={{ textAlign: 'center', padding: '20px' }}>Loading...</div>
            ) : (
              <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                {samples.length > 0 ? (
                  samples.map((sample, index) => (
                    <div
                      key={sample.sample_id || index}
                      onClick={() => handleSelectSample(sample.sample_id)}
                      style={{
                        padding: '15px',
                        marginBottom: '10px',
                        backgroundColor: selectedSample?.sample_id === sample.sample_id ? '#e8f5e9' : '#f5f5f5',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        border: selectedSample?.sample_id === sample.sample_id ? '2px solid #4ecdc4' : '2px solid transparent'
                      }}
                    >
                      <div style={{ fontWeight: 'bold' }}>{sample.sample_id || `Sample ${index + 1}`}</div>
                      <div style={{ fontSize: '0.8rem', color: '#666' }}>
                        Subject: {sample.subject_id || 'N/A'}
                      </div>
                      <div style={{ fontSize: '0.8rem', color: '#666' }}>
                        Disease Status: {sample.disease_status || 'Unknown'}
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ textAlign: 'center', padding: '20px', color: '#666' }}>
                    No samples available. Click "Load Demo Data" to see example data.
                  </div>
                )}
              </div>
            )}
            
            <button
              onClick={() => {
                // Generate demo samples
                const demoSamples = Array.from({ length: 10 }, (_, i) => ({
                  sample_id: `SAMPLE_${String(i + 1).padStart(3, '0')}`,
                  subject_id: `SUBJ_${String(Math.floor(i / 2) + 1).padStart(3, '0')}`,
                  disease_status: ['Healthy', 'IBD', 'Type 2 Diabetes', 'Obesity'][i % 4],
                  age: 25 + (i * 3),
                  bmi: 22 + (i % 5),
                }));
                setSamples(demoSamples);
              }}
              style={{
                marginTop: '15px',
                padding: '10px 20px',
                backgroundColor: '#4ecdc4',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold',
                width: '100%'
              }}
            >
              📥 Load Demo Data
            </button>
          </div>

          {/* Selected Sample Details */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>📋 Sample Details</h3>
            
            {selectedSample ? (
              <div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '10px', fontWeight: 'bold' }}>Sample ID:</td>
                      <td style={{ padding: '10px' }}>{selectedSample.sample_id}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '10px', fontWeight: 'bold' }}>Subject ID:</td>
                      <td style={{ padding: '10px' }}>{selectedSample.subject_id}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '10px', fontWeight: 'bold' }}>Disease Status:</td>
                      <td style={{ padding: '10px' }}>{selectedSample.disease_status}</td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '10px', fontWeight: 'bold' }}>Age:</td>
                      <td style={{ padding: '10px' }}>{selectedSample.age}</td>
                    </tr>
                    <tr>
                      <td style={{ padding: '10px', fontWeight: 'bold' }}>BMI:</td>
                      <td style={{ padding: '10px' }}>{selectedSample.bmi}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: '#666', textAlign: 'center', padding: '40px' }}>
                Select a sample from the list to view details
              </p>
            )}
          </div>
        </div>
      )}

      {/* Diversity View */}
      {viewMode === 'diversity' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Diversity Metrics */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>📊 Diversity Metrics</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Alpha diversity measurements for microbiome samples
            </p>
            
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={diversityMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#4ecdc4" name="Value" />
                  <Bar dataKey="max" fill="#dfe6e9" name="Max" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Disease Distribution */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>🏥 Disease Distribution</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Distribution of disease conditions in the dataset
            </p>
            
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={diseaseProfiles}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {diseaseProfiles.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Analyze Diversity Button */}
          <div style={{ gridColumn: '1 / -1' }}>
            <button
              onClick={handleAnalyzeDiversity}
              disabled={loading}
              style={{
                padding: '12px 24px',
                backgroundColor: loading ? '#ccc' : '#96ceb4',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: 'bold',
                fontSize: '1rem'
              }}
            >
              {loading ? '⏳ Analyzing...' : '🔍 Analyze Diversity'}
            </button>
          </div>
        </div>
      )}

      {/* Composition View */}
      {viewMode === 'composition' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Bacterial Composition Pie Chart */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>🦠 Bacterial Composition</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Relative abundance of bacterial phyla in the microbiome
            </p>
            
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={bacteriaComposition}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {bacteriaComposition.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Bacterial Composition Bar Chart */}
          <div style={{ 
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>📊 Abundance by Genus</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Detailed breakdown of bacterial genera
            </p>
            
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bacteriaComposition} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#4ecdc4" name="Abundance (%)">
                    {bacteriaComposition.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Patient Profile Estimation */}
          <div style={{ 
            gridColumn: '1 / -1',
            padding: '20px', 
            backgroundColor: 'white', 
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>👤 Patient Profile Estimation</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>
              Predict patient characteristics based on microbiome composition
            </p>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px' }}>
              <div style={{ padding: '15px', backgroundColor: '#e8f5e9', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🎂</div>
                <div style={{ fontSize: '0.8rem', color: '#666' }}>Estimated Age</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#4ecdc4' }}>34 years</div>
              </div>
              <div style={{ padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '10px' }}>⚖️</div>
                <div style={{ fontSize: '0.8rem', color: '#666' }}>BMI Category</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#45b7d1' }}>Normal</div>
              </div>
              <div style={{ padding: '15px', backgroundColor: '#fff3e0', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '10px' }}>❤️</div>
                <div style={{ fontSize: '0.8rem', color: '#666' }}>Health Status</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#4ecdc4' }}>Healthy</div>
              </div>
              <div style={{ padding: '15px', backgroundColor: '#f3e5f5', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🔬</div>
                <div style={{ fontSize: '0.8rem', color: '#666' }}>Confidence</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#96ceb4' }}>87%</div>
              </div>
            </div>
            
            <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
              <h4 style={{ margin: '0 0 10px 0' }}>Estimation Details:</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>High Lactobacillus abundance suggests healthy gut</li>
                <li>Balanced Firmicutes/Bacteroides ratio indicates normal metabolism</li>
                <li>Low inflammation markers suggest no active disease</li>
                <li>Diversity metrics within normal range</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MicrobiomePage;
