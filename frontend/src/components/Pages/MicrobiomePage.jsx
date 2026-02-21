import React, { useState, useEffect } from 'react';
import { microbiomeAPI } from '../../services/api';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

const MicrobiomePage = () => {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('dashboard');
  
  // Data states
  const [summary, setSummary] = useState(null);
  const [composition, setComposition] = useState([]);
  const [diversity, setDiversity] = useState([]);
  const [diseases, setDiseases] = useState({});
  const [taxa, setTaxa] = useState({ phyla: [], genera: [] });
  const [patientProfile, setPatientProfile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Form state for custom sample
  const [customSample, setCustomSample] = useState({
    Bacteroides: 0.15,
    Prevotella: 0.10,
    Faecalibacterium: 0.12,
    Bifidobacterium: 0.08,
    Lactobacillus: 0.10,
    Escherichia: 0.05,
    Streptococcus: 0.05,
    Clostridium: 0.08,
    Ruminococcus: 0.07,
    Akkermansia: 0.05,
    Blautia: 0.08,
    Roseburia: 0.07
  });

  // Load initial data
  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Load all required data
      const [summaryRes, compositionRes, diversityRes, diseasesRes, taxaRes, samplesRes] = await Promise.all([
        microbiomeAPI.getSummary(),
        microbiomeAPI.getComposition(),
        microbiomeAPI.getDiversity(),
        microbiomeAPI.getDiseases(),
        microbiomeAPI.getTaxa(),
        microbiomeAPI.getSamples()
      ]);

      setSummary(summaryRes.data);
      setComposition(compositionRes.data.composition || []);
      setDiversity(diversityRes.data.samples || []);
      setDiseases(diseasesRes.data);
      setTaxa(taxaRes.data);
      setSamples(samplesRes.data.samples || []);
    } catch (err) {
      console.error('Error loading dashboard:', err);
      setError('Failed to load data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    try {
      const response = await microbiomeAPI.estimatePatient(sample);
      setPatientProfile(response.data);
    } catch (err) {
      console.error('Error analyzing sample:', err);
    }
  };

  const handleAnalyzeCustom = async () => {
    setLoading(true);
    try {
      const response = await microbiomeAPI.analyze(customSample);
      setAnalysisResult(response.data);
    } catch (err) {
      console.error('Error analyzing:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEstimatePatient = async () => {
    setLoading(true);
    try {
      const sampleData = {
        ...customSample,
        age: 35,
        bmi: 24
      };
      const response = await microbiomeAPI.estimatePatient(sampleData);
      setPatientProfile(response.data);
      setViewMode('patient');
    } catch (err) {
      console.error('Error estimating patient:', err);
    } finally {
      setLoading(false);
    }
  };

  // Color palette
  const colors = ['#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#ff6b6b', '#74b9ff', '#a29bfe', '#fd79a8', '#fdcb6e', '#00b894', '#e17055', '#0984e3'];

  // Prepare disease distribution data for pie chart
  const diseaseDistribution = summary?.disease_distribution ? 
    Object.entries(summary.disease_distribution).map(([name, value], index) => ({
      name,
      value,
      color: colors[index % colors.length]
    })) : [];

  // Prepare composition data for charts
  const topComposition = composition.slice(0, 8).map((item, index) => ({
    ...item,
    color: colors[index % colors.length]
  }));

  return (
    <div className="microbiome-page" style={{ padding: '20px', height: '100%', overflow: 'auto', backgroundColor: '#f5f7fa' }}>
      <h1 style={{ marginBottom: '20px', color: '#2d3436' }}>🦠 Microbiome Signals</h1>
      
      {/* Mode Selector */}
      <div style={{ 
        display: 'flex', 
        gap: '10px', 
        marginBottom: '20px',
        flexWrap: 'wrap'
      }}>
        {['dashboard', 'samples', 'composition', 'diversity', 'patient', 'analyze'].map(mode => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            style={{
              padding: '10px 20px',
              backgroundColor: viewMode === mode ? '#4ecdc4' : '#fff',
              color: viewMode === mode ? 'white' : '#333',
              border: '1px solid #ddd',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              boxShadow: viewMode === mode ? '0 2px 4px rgba(0,0,0,0.1)' : 'none'
            }}
          >
            {mode === 'dashboard' && '📊 Dashboard'}
            {mode === 'samples' && '🧬 Samples'}
            {mode === 'composition' && '🦠 Composition'}
            {mode === 'diversity' && '📈 Diversity'}
            {mode === 'patient' && '👤 Patient Profile'}
            {mode === 'analyze' && '🔬 Analyze'}
          </button>
        ))}
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ padding: '15px', backgroundColor: '#ffebee', color: '#c62828', borderRadius: '4px', marginBottom: '20px' }}>
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div style={{ fontSize: '24px' }}>⏳ Loading microbiome data...</div>
        </div>
      )}

      {/* DASHBOARD VIEW */}
      {viewMode === 'dashboard' && !loading && summary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '20px' }}>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <div style={{ fontSize: '2rem', color: '#4ecdc4' }}>🧬</div>
            <div style={{ fontSize: '0.9rem', color: '#666' }}>Total Samples</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{summary.total_samples}</div>
          </div>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <div style={{ fontSize: '2rem', color: '#45b7d1' }}>👥</div>
            <div style={{ fontSize: '0.9rem', color: '#666' }}>Subjects</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{summary.total_subjects}</div>
          </div>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <div style={{ fontSize: '2rem', color: '#96ceb4' }}>🦠</div>
            <div style={{ fontSize: '0.9rem', color: '#666' }}>Bacterial Taxa</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{summary.bacterial_taxa}</div>
          </div>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <div style={{ fontSize: '2rem', color: '#ff6b6b' }}>🏥</div>
            <div style={{ fontSize: '0.9rem', color: '#666' }}>Diseases</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>{Object.keys(diseases).length}</div>
          </div>
        </div>
      )}

      {viewMode === 'dashboard' && !loading && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Disease Distribution */}
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🏥 Disease Distribution</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={diseaseDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {diseaseDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Top Bacteria */}
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🦠 Top Bacterial Genera</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topComposition} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 0.2]} />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#4ecdc4" name="Abundance">
                    {topComposition.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Diversity Metrics */}
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>📊 Diversity Metrics</h3>
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={diversity.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sample_id" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="shannon_index" fill="#4ecdc4" name="Shannon Index" />
                  <Bar dataKey="simpson_index" fill="#45b7d1" name="Simpson Index" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Disease Profiles */}
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🔬 Disease Profiles</h3>
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {Object.entries(diseases).map(([disease, profile], index) => (
                <div key={disease} style={{ padding: '10px', marginBottom: '10px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold', color: colors[index % colors.length] }}>{disease}</div>
                  <div style={{ fontSize: '0.8rem', color: '#666' }}>{profile.description}</div>
                  <div style={{ fontSize: '0.75rem', marginTop: '5px' }}>
                    <span style={{ color: '#27ae60' }}>↓ {profile.decreased?.join(', ')}</span>
                  </div>
                  <div style={{ fontSize: '0.75rem' }}>
                    <span style={{ color: '#e74c3c' }}>↑ {profile.increased?.join(', ')}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* SAMPLES VIEW */}
      {viewMode === 'samples' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🧬 Sample List</h3>
            <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
              {samples.slice(0, 50).map((sample, index) => (
                <div
                  key={sample.sample_id || index}
                  onClick={() => handleSelectSample(sample)}
                  style={{
                    padding: '12px',
                    marginBottom: '8px',
                    backgroundColor: selectedSample?.sample_id === sample.sample_id ? '#e8f5e9' : '#f5f5f5',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    border: selectedSample?.sample_id === sample.sample_id ? '2px solid #4ecdc4' : '2px solid transparent'
                  }}
                >
                  <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>{sample.sample_id}</div>
                  <div style={{ fontSize: '0.75rem', color: '#666' }}>
                    {sample.disease_status} | Age: {sample.age} | BMI: {sample.bmi?.toFixed(1)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>📋 Sample Details & Analysis</h3>
            {selectedSample ? (
              <div>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '20px' }}>
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
                    <tr style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '10px', fontWeight: 'bold' }}>BMI:</td>
                      <td style={{ padding: '10px' }}>{selectedSample.bmi?.toFixed(1)}</td>
                    </tr>
                  </tbody>
                </table>

                {patientProfile && (
                  <div>
                    <h4 style={{ color: '#4ecdc4' }}>Patient Profile Estimation</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', marginBottom: '15px' }}>
                      <div style={{ padding: '10px', backgroundColor: '#e8f5e9', borderRadius: '4px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#666' }}>Age Group</div>
                        <div style={{ fontWeight: 'bold' }}>{patientProfile.patient_profile?.age_group}</div>
                      </div>
                      <div style={{ padding: '10px', backgroundColor: '#e3f2fd', borderRadius: '4px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#666' }}>BMI Category</div>
                        <div style={{ fontWeight: 'bold' }}>{patientProfile.patient_profile?.bmi_category}</div>
                      </div>
                      <div style={{ padding: '10px', backgroundColor: '#fff3e0', borderRadius: '4px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#666' }}>Gut Health Score</div>
                        <div style={{ fontWeight: 'bold' }}>{patientProfile.patient_profile?.gut_health_score}%</div>
                      </div>
                      <div style={{ padding: '10px', backgroundColor: '#f3e5f5', borderRadius: '4px' }}>
                        <div style={{ fontSize: '0.8rem', color: '#666' }}>Microbial Diversity</div>
                        <div style={{ fontWeight: 'bold' }}>{patientProfile.patient_profile?.microbial_diversity}</div>
                      </div>
                    </div>

                    <h4 style={{ color: '#ff6b6b' }}>Disease Risk Assessment</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                      {Object.entries(patientProfile.disease_risks?.diseases || {}).map(([disease, risk]) => (
                        <div key={disease} style={{ 
                          padding: '10px', 
                          backgroundColor: risk.risk_level === 'High' ? '#ffebee' : risk.risk_level === 'Moderate' ? '#fff3e0' : '#e8f5e9',
                          borderRadius: '4px',
                          borderLeft: `3px solid ${risk.risk_level === 'High' ? '#e74c3c' : risk.risk_level === 'Moderate' ? '#f39c12' : '#27ae60'}`
                        }}>
                          <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>{disease}</div>
                          <div style={{ fontSize: '0.8rem', color: risk.risk_level === 'High' ? '#e74c3c' : risk.risk_level === 'Moderate' ? '#f39c12' : '#27ae60' }}>
                            {risk.risk_level} Risk
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: '#666', textAlign: 'center', padding: '40px' }}>
                Select a sample from the list to view details
              </p>
            )}
          </div>
        </div>
      )}

      {/* COMPOSITION VIEW */}
      {viewMode === 'composition' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🦠 Bacterial Composition (Pie)</h3>
            <div style={{ height: '400px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={topComposition}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                    outerRadius={150}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {topComposition.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>📊 Bacterial Composition (Bar)</h3>
            <div style={{ height: '400px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topComposition}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#4ecdc4" name="Relative Abundance">
                    {topComposition.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ gridColumn: '1 / -1', padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🔬 Phylogenetic Taxonomy</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <h4 style={{ color: '#4ecdc4' }}>Bacterial Phyla</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                  {taxa.phyla.map((phylum, index) => (
                    <span key={phylum} style={{ padding: '5px 10px', backgroundColor: colors[index % colors.length] + '20', borderRadius: '4px', fontSize: '0.85rem' }}>
                      {phylum}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <h4 style={{ color: '#45b7d1' }}>Bacterial Genera</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                  {taxa.genera.map((genus, index) => (
                    <span key={genus} style={{ padding: '5px 10px', backgroundColor: colors[index % colors.length] + '20', borderRadius: '4px', fontSize: '0.85rem' }}>
                      {genus}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DIVERSITY VIEW */}
      {viewMode === 'diversity' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>📈 Shannon Diversity Index</h3>
            <div style={{ height: '400px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={diversity.slice(0, 30)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sample_id" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="shannon_index" stroke="#4ecdc4" strokeWidth={2} name="Shannon Index" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>📊 Simpson Diversity Index</h3>
            <div style={{ height: '400px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={diversity.slice(0, 30)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sample_id" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="simpson_index" stroke="#45b7d1" strokeWidth={2} name="Simpson Index" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ gridColumn: '1 / -1', padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🔢 Diversity Metrics Summary</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ backgroundColor: '#f5f5f5' }}>
                  <th style={{ padding: '10px', textAlign: 'left' }}>Metric</th>
                  <th style={{ padding: '10px', textAlign: 'left' }}>Description</th>
                  <th style={{ padding: '10px', textAlign: 'right' }}>Value</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '10px' }}>Shannon Index</td>
                  <td style={{ padding: '10px', color: '#666' }}>Measures species diversity (higher = more diverse)</td>
                  <td style={{ padding: '10px', textAlign: 'right', fontWeight: 'bold' }}>
                    {(diversity.reduce((sum, d) => sum + (d.shannon_index || 0), 0) / (diversity.length || 1)).toFixed(3)}
                  </td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '10px' }}>Simpson Index</td>
                  <td style={{ padding: '10px', color: '#666' }}>Measures dominance (higher = more diverse)</td>
                  <td style={{ padding: '10px', textAlign: 'right', fontWeight: 'bold' }}>
                    {(diversity.reduce((sum, d) => sum + (d.simpson_index || 0), 0) / (diversity.length || 1)).toFixed(3)}
                  </td>
                </tr>
                <tr style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '10px' }}>Chao1</td>
                  <td style={{ padding: '10px', color: '#666' }}>Estimated species richness</td>
                  <td style={{ padding: '10px', textAlign: 'right', fontWeight: 'bold' }}>
                    {(diversity.reduce((sum, d) => sum + (d.chao1 || 0), 0) / (diversity.length || 1)).toFixed(1)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* PATIENT PROFILE VIEW */}
      {viewMode === 'patient' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>👤 Patient Profile</h3>
            {patientProfile ? (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '20px' }}>
                  <div style={{ padding: '15px', backgroundColor: '#e8f5e9', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem' }}>🎂</div>
                    <div style={{ fontSize: '0.8rem', color: '#666' }}>Age Group</div>
                    <div style={{ fontWeight: 'bold' }}>{patientProfile.patient_profile?.age_group}</div>
                  </div>
                  <div style={{ padding: '15px', backgroundColor: '#e3f2fd', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem' }}>⚖️</div>
                    <div style={{ fontSize: '0.8rem', color: '#666' }}>BMI Category</div>
                    <div style={{ fontWeight: 'bold' }}>{patientProfile.patient_profile?.bmi_category}</div>
                  </div>
                  <div style={{ padding: '15px', backgroundColor: '#fff3e0', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem' }}>❤️</div>
                    <div style={{ fontSize: '0.8rem', color: '#666' }}>Gut Health</div>
                    <div style={{ fontWeight: 'bold', color: '#4ecdc4' }}>{patientProfile.patient_profile?.gut_health_score}%</div>
                  </div>
                  <div style={{ padding: '15px', backgroundColor: '#f3e5f5', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '2rem' }}>🔬</div>
                    <div style={{ fontSize: '0.8rem', color: '#666' }}>Diversity</div>
                    <div style={{ fontWeight: 'bold' }}>{patientProfile.patient_profile?.microbial_diversity}</div>
                  </div>
                </div>

                <h4 style={{ color: '#ff6b6b' }}>Disease Risk Radar</h4>
                <div style={{ height: '300px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={Object.entries(patientProfile.disease_risks?.diseases || {}).map(([name, data]) => ({
                      disease: name,
                      score: data.score * 25,
                      fullMark: 100
                    }))}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="disease" />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} />
                      <Radar name="Risk Score" dataKey="score" stroke="#4ecdc4" fill="#4ecdc4" fillOpacity={0.3} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <p style={{ color: '#666', textAlign: 'center', padding: '20px' }}>
                Go to Analyze tab to create a patient profile
              </p>
            )}
          </div>

          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>💊 Recommendations</h3>
            {patientProfile?.recommendations ? (
              <ul style={{ paddingLeft: '20px' }}>
                {patientProfile.recommendations.map((rec, index) => (
                  <li key={index} style={{ marginBottom: '10px', padding: '10px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
                    {rec}
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ color: '#666', textAlign: 'center', padding: '20px' }}>
                No recommendations available
              </p>
            )}
          </div>
        </div>
      )}

      {/* ANALYZE VIEW */}
      {viewMode === 'analyze' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>🔬 Enter Microbiome Profile</h3>
            <p style={{ color: '#666', marginBottom: '20px' }}>Adjust bacterial relative abundances (must sum to ~1)</p>
            
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              {Object.entries(customSample).map(([bacteria, value]) => (
                <div key={bacteria} style={{ marginBottom: '15px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                    <label style={{ fontWeight: 'bold' }}>{bacteria}</label>
                    <span>{value.toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="0.5"
                    step="0.001"
                    value={value}
                    onChange={(e) => setCustomSample({...customSample, [bacteria]: parseFloat(e.target.value)})}
                    style={{ width: '100%' }}
                  />
                </div>
              ))}
            </div>

            <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
              <button
                onClick={handleAnalyzeCustom}
                disabled={loading}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: loading ? '#ccc' : '#4ecdc4',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold'
                }}
              >
                {loading ? 'Analyzing...' : '🔍 Analyze'}
              </button>
              <button
                onClick={handleEstimatePatient}
                disabled={loading}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: loading ? '#ccc' : '#45b7d1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold'
                }}
              >
                {loading ? 'Processing...' : '👤 Get Patient Profile'}
              </button>
            </div>
          </div>

          <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h3>📊 Analysis Results</h3>
            {analysisResult ? (
              <div>
                <h4 style={{ color: '#4ecdc4' }}>Bacterial Profile</h4>
                <div style={{ marginBottom: '20px' }}>
                  {analysisResult.bacterial_profile?.top_genera?.slice(0, 5).map((item, index) => (
                    <div key={index} style={{ padding: '8px', marginBottom: '5px', backgroundColor: '#f5f5f5', borderRadius: '4px', display: 'flex', justifyContent: 'space-between' }}>
                      <span>{item.genus}</span>
                      <span style={{ fontWeight: 'bold' }}>{(item.abundance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>

                <h4 style={{ color: '#ff6b6b' }}>Disease Risks</h4>
                {Object.entries(analysisResult.disease_risks?.diseases || {}).map(([disease, risk]) => (
                  <div key={disease} style={{ 
                    padding: '10px', 
                    marginBottom: '10px', 
                    backgroundColor: risk.risk_level === 'High' ? '#ffebee' : risk.risk_level === 'Moderate' ? '#fff3e0' : '#e8f5e9',
                    borderRadius: '4px',
                    borderLeft: `4px solid ${risk.risk_level === 'High' ? '#e74c3c' : risk.risk_level === 'Moderate' ? '#f39c12' : '#27ae60'}`
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 'bold' }}>
                      <span>{disease}</span>
                      <span style={{ color: risk.risk_level === 'High' ? '#e74c3c' : risk.risk_level === 'Moderate' ? '#f39c12' : '#27ae60' }}>
                        {risk.risk_level}
                      </span>
                    </div>
                    {risk.details?.length > 0 && (
                      <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '5px' }}>
                        {risk.details.join(', ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#666', textAlign: 'center', padding: '40px' }}>
                Enter microbiome profile and click Analyze to see results
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MicrobiomePage;
