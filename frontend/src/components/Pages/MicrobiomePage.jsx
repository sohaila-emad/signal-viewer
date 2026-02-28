import React, { useState, useEffect, useCallback } from 'react';
import { microbiomeAPI } from '../../services/api';
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts';

const COLORS = [
  '#4ecdc4','#ff6b6b','#45b7d1','#ffeaa7',
  '#96ceb4','#fd79a8','#a29bfe','#fdcb6e',
  '#55efc4','#e17055','#74b9ff','#fab1a0',
];

const CARD = {
  padding: '20px',
  backgroundColor: 'white',
  borderRadius: '8px',
  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
};

const DX_COLOR = { CD: '#e17055', UC: '#fdcb6e', nonIBD: '#55efc4', IBD: '#e17055', Obesity: '#fdcb6e', Healthy: '#55efc4', CVD: '#a29bfe', Type2Diabetes: '#ff6b6b' };

const fmt = (v) => {
  if (v === null || v === undefined) return 'N/A';
  if (typeof v !== 'number') return String(v);
  if (Math.abs(v) < 0.0001) return v.toExponential(3);
  return v.toFixed(4);
};

const MetricRow = ({ label, value, unit = '', info = '' }) => (
  <div style={{
    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
    padding: '8px 12px', marginBottom: '6px',
    backgroundColor: '#f7f9fc', borderRadius: '6px', fontSize: '0.88rem',
  }}>
    <div>
      <span style={{ fontWeight: 'bold' }}>{label}</span>
      {info && <div style={{ color: '#636e72', fontSize: '0.78rem', marginTop: '2px' }}>{info}</div>}
    </div>
    <span style={{ fontWeight: 'bold', color: '#2d3436', whiteSpace: 'nowrap', marginLeft: '12px' }}>
      {fmt(value)}{unit}
    </span>
  </div>
);

// ── Main component ────────────────────────────────────────────────────────────
const MicrobiomePage = () => {
  const [summary, setSummary]     = useState(null);
  const [composition, setComp]    = useState([]);
  const [error, setError]         = useState(null);
  const [loading, setLoading]     = useState(true);
  const [tab, setTab]             = useState('overview');

  // Timeline
  const [participant, setParticipant] = useState('');
  const [participants, setParticipants] = useState([]);
  const [selectedGenera, setSelectedGenera] = useState([]);
  const [tlData, setTlData]       = useState(null);
  const [tlLoading, setTlLoading] = useState(false);
  const [pickerExpanded, setPickerExpanded] = useState(false);

  // Profile
  const [profParticipant, setProfParticipant] = useState('');
  const [profile, setProfile]     = useState(null);
  const [profLoading, setProfLoading] = useState(false);

  // ── Initial load ─────────────────────────────────────────────────────────────
  const fetchSummary = useCallback(async () => {
    try {
      const res = await microbiomeAPI.getSummary();
      setSummary(res.data);
      setError(null);

      const [compRes, partRes] = await Promise.all([
        microbiomeAPI.getComposition(),
        microbiomeAPI.getParticipants(),
      ]);
      setComp(compRes.data || []);
      const ids = partRes.data.participants || [];
      setParticipants(ids);
      if (ids.length && !participant) {
        setParticipant(ids[0]);
        setProfParticipant(ids[0]);
      }
      if (!selectedGenera.length && res.data.genera?.length) {
        setSelectedGenera(res.data.genera.slice(0, 5));
      }
    } catch (err) {
      setError('Backend error: ' + err.message);
    } finally {
      setLoading(false);
    }
  }, [participant, selectedGenera.length]);

  useEffect(() => { fetchSummary(); }, []);

  // ── Timeline ─────────────────────────────────────────────────────────────────
  const fetchTimeline = useCallback(async (pid, genera) => {
    if (!pid || !genera.length) return;
    setTlLoading(true);
    setError(null);
    try {
      const res = await microbiomeAPI.getTimeline(pid, genera);
      setTlData(res.data);
    } catch (err) {
      setError('Timeline error: ' + (err.response?.data?.error || err.message));
    } finally {
      setTlLoading(false);
    }
  }, []);

  const handleParticipantChange = (pid) => {
    setParticipant(pid);
    setTlData(null);
    fetchTimeline(pid, selectedGenera);
  };

  const toggleGenus = (genus) => {
    const next = selectedGenera.includes(genus)
      ? selectedGenera.filter(g => g !== genus)
      : [...selectedGenera, genus];
    setSelectedGenera(next);
    if (next.length && participant) fetchTimeline(participant, next);
  };

  const handleTabChange = (t) => {
    setTab(t);
    if (t === 'timeline' && participant && !tlData) {
      fetchTimeline(participant, selectedGenera);
    }
  };

  // ── Profile ───────────────────────────────────────────────────────────────────
  const fetchProfile = useCallback(async (pid) => {
    if (!pid) return;
    setProfLoading(true);
    setError(null);
    try {
      const res = await microbiomeAPI.getParticipantProfile(pid);
      setProfile(res.data);
    } catch (err) {
      setError('Profile error: ' + (err.response?.data?.error || err.message));
    } finally {
      setProfLoading(false);
    }
  }, []);

  const handleProfParticipantChange = (pid) => {
    setProfParticipant(pid);
    setProfile(null);
    fetchProfile(pid);
  };

  // ── Derived ───────────────────────────────────────────────────────────────────
  const compBarData = composition.slice(0, 15).map(c => ({
    genus: c.genus,
    abundance: parseFloat((c.mean_abundance * 100).toFixed(3)),
  }));

  const timelineChart = (tlData?.timeline || []).map(pt => {
    const row = { label: pt.label, shannon: pt.shannon };
    selectedGenera.forEach(g => { row[g] = pt[g] ?? 0; });
    return row;
  });

  const sortedGenera = composition.length
    ? composition.map(c => c.genus).filter(g => (summary?.genera || []).includes(g))
    : (summary?.genera || []);

  const COLLAPSED_COUNT = 5;
  const visibleGenera = pickerExpanded ? sortedGenera : sortedGenera.slice(0, COLLAPSED_COUNT);

  const selectTop5 = () => { const n = sortedGenera.slice(0, 5); setSelectedGenera(n); if (participant) fetchTimeline(participant, n); };
  const selectAll  = () => { const n = [...sortedGenera];         setSelectedGenera(n); if (participant) fetchTimeline(participant, n); };
  const selectNone = () => { setSelectedGenera([]); setTlData(null); };

  // ── Loading ───────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{ padding: '20px', height: '100%', overflow: 'auto', backgroundColor: '#f5f7fa' }}>
        <h1 style={{ marginBottom: '6px', color: '#2d3436' }}>Microbiome Analysis</h1>
        {error
          ? <div style={{ padding: '10px 14px', backgroundColor: '#ffebee', color: '#c62828', borderRadius: '4px', fontSize: '0.88rem' }}>{error}</div>
          : <div style={{ ...CARD, textAlign: 'center', padding: '60px 40px' }}><div style={{ color: '#636e72' }}>Loading dataset…</div></div>
        }
      </div>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div style={{ padding: '20px', height: '100%', overflow: 'auto', backgroundColor: '#f5f7fa' }}>

      {/* Header */}
      <div style={{ marginBottom: '16px' }}>
        <h1 style={{ margin: 0, color: '#2d3436' }}>Microbiome Analysis</h1>
        <p style={{ color: '#636e72', margin: '4px 0 0', fontSize: '0.85rem' }}>
          iHMP longitudinal dataset — {summary.n_participants} subjects · {summary.n_samples} visits · {summary.n_genera} genera
        </p>
      </div>

      {error && (
        <div style={{ padding: '10px 14px', backgroundColor: '#fff3e0', color: '#bf360c',
          borderRadius: '4px', marginBottom: '16px', fontSize: '0.88rem' }}>
          {error}
        </div>
      )}

      {/* Stat chips */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
        {[
          { label: 'Subjects',  value: summary.n_participants },
          { label: 'Visits',    value: summary.n_samples },
          { label: 'Genera',    value: summary.n_genera },
        ].map(({ label, value }) => (
          <div key={label} style={{
            ...CARD, display: 'flex', flexDirection: 'column', alignItems: 'center',
            padding: '12px 24px', minWidth: '90px',
          }}>
            <span style={{ fontSize: '1.6rem', fontWeight: 'bold', color: '#4ecdc4' }}>{value ?? '—'}</span>
            <span style={{ fontSize: '0.75rem', color: '#636e72' }}>{label}</span>
          </div>
        ))}
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' }}>
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'timeline', label: 'Timeline' },
          { id: 'profile',  label: 'Patient Profile' },
        ].map(t => (
          <button key={t.id} onClick={() => handleTabChange(t.id)} style={{
            padding: '9px 20px',
            backgroundColor: tab === t.id ? '#4ecdc4' : '#fff',
            color: tab === t.id ? 'white' : '#333',
            border: '1px solid #ddd', borderRadius: '4px',
            cursor: 'pointer', fontWeight: 'bold', fontSize: '0.9rem',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW ──────────────────────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div style={CARD}>
          <h3 style={{ marginBottom: '4px' }}>Mean Taxonomic Composition</h3>
          <p style={{ fontSize: '0.8rem', color: '#636e72', marginBottom: '14px' }}>
            Average relative abundance per genus across all visits
          </p>
          {compBarData.length > 0 ? (
            <div style={{ height: `${Math.max(300, compBarData.length * 26)}px` }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={compBarData} layout="vertical" margin={{ left: 10, right: 34 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" unit="%" />
                  <YAxis dataKey="genus" type="category" width={125} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={v => [`${v}%`, 'Mean abundance']} />
                  <Bar dataKey="abundance">
                    {compBarData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div style={{ textAlign: 'center', color: '#b2bec3', padding: '30px' }}>No composition data.</div>
          )}
        </div>
      )}

      {/* ── TIMELINE ──────────────────────────────────────────────────────────── */}
      {tab === 'timeline' && (
        <div style={{ display: 'grid', gap: '16px' }}>
          <div style={CARD}>
            {/* Participant selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', marginBottom: '14px' }}>
              <label style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>Participant:</label>
              <select
                value={participant}
                onChange={e => handleParticipantChange(e.target.value)}
                style={{ padding: '7px 12px', borderRadius: '4px', border: '1px solid #ddd', fontSize: '0.9rem' }}
              >
                {participants.map(id => <option key={id} value={id}>{id}</option>)}
              </select>
              {tlData && (
                <span style={{ color: '#636e72', fontSize: '0.82rem' }}>
                  {tlData.n_visits} visit{tlData.n_visits !== 1 ? 's' : ''}
                </span>
              )}
            </div>

            {/* Genus picker */}
            <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px', flexWrap: 'wrap' }}>
                <span style={{ fontWeight: 'bold', fontSize: '0.85rem', color: '#2d3436' }}>
                  Genera ({selectedGenera.length} / {sortedGenera.length} selected)
                </span>
                <div style={{ display: 'flex', gap: '6px', marginLeft: 'auto' }}>
                  {[{ label: 'Top 5', action: selectTop5 }, { label: 'All', action: selectAll }, { label: 'None', action: selectNone }].map(({ label, action }) => (
                    <button key={label} onClick={action} style={{
                      padding: '3px 10px', fontSize: '0.78rem',
                      border: '1px solid #b2bec3', borderRadius: '4px',
                      backgroundColor: 'white', cursor: 'pointer', color: '#636e72',
                    }}>{label}</button>
                  ))}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '6px' }}>
                {visibleGenera.map((g) => {
                  const colorIdx = sortedGenera.indexOf(g);
                  const active   = selectedGenera.includes(g);
                  const abund    = composition.find(c => c.genus === g);
                  const pct      = abund ? (abund.mean_abundance * 100).toFixed(2) : null;
                  return (
                    <button key={g} onClick={() => toggleGenus(g)} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '5px 10px', borderRadius: '6px',
                      border: `2px solid ${COLORS[colorIdx % COLORS.length]}`,
                      backgroundColor: active ? COLORS[colorIdx % COLORS.length] : 'white',
                      color: active ? 'white' : '#2d3436',
                      fontSize: '0.80rem', cursor: 'pointer',
                      fontWeight: active ? 'bold' : 'normal',
                      transition: 'background-color 0.12s', textAlign: 'left',
                    }}>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{g}</span>
                      {pct !== null && (
                        <span style={{ marginLeft: '6px', fontSize: '0.70rem', opacity: active ? 0.85 : 0.55, flexShrink: 0 }}>
                          {pct}%
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>

              {sortedGenera.length > COLLAPSED_COUNT && (
                <button onClick={() => setPickerExpanded(x => !x)} style={{
                  marginTop: '8px', padding: '4px 12px', fontSize: '0.78rem',
                  border: '1px solid #dfe6e9', borderRadius: '4px',
                  backgroundColor: '#f5f7fa', cursor: 'pointer', color: '#636e72',
                }}>
                  {pickerExpanded ? 'Show fewer' : `Show ${sortedGenera.length - COLLAPSED_COUNT} more…`}
                </button>
              )}
            </div>
          </div>

          {tlLoading && <div style={{ textAlign: 'center', color: '#636e72', padding: '30px' }}>Loading…</div>}

          {!tlLoading && timelineChart.length > 0 && (
            <>
              <div style={CARD}>
                <h3 style={{ marginBottom: '4px' }}>Relative Abundance per Visit</h3>
                <p style={{ fontSize: '0.8rem', color: '#636e72', marginBottom: '14px' }}>
                  Abundance (%) of selected genera across visits
                </p>
                {tlData?.trends?.shannon_delta != null && (
                  <div style={{ display: 'flex', gap: '20px', marginBottom: '10px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.82rem', color: '#636e72' }}>
                      Shannon Δ: <strong style={{ color: tlData.trends.shannon_delta >= 0 ? '#27ae60' : '#e74c3c' }}>
                        {tlData.trends.shannon_delta > 0 ? '+' : ''}{tlData.trends.shannon_delta}
                      </strong>
                    </span>
                    {tlData.trends.fb_delta != null && (
                      <span style={{ fontSize: '0.82rem', color: '#636e72' }}>
                        F/B Δ: <strong style={{ color: tlData.trends.fb_delta <= 0 ? '#27ae60' : '#e74c3c' }}>
                          {tlData.trends.fb_delta > 0 ? '+' : ''}{tlData.trends.fb_delta}
                        </strong>
                      </span>
                    )}
                  </div>
                )}
                <div style={{ height: '320px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timelineChart} margin={{ right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                      <YAxis unit="%" domain={[0, 'auto']} />
                      <Tooltip formatter={(v, name) => [`${v}%`, name]} />
                      <Legend />
                      {selectedGenera.map((g, i) => (
                        <Line key={g} type="monotone" dataKey={g}
                          stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={{ r: 3 }} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div style={CARD}>
                <h3 style={{ marginBottom: '4px' }}>Alpha Diversity — Shannon H'</h3>
                <p style={{ fontSize: '0.8rem', color: '#636e72', marginBottom: '14px' }}>
                  Higher H' = more even distribution across genera
                </p>
                <div style={{ height: '220px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timelineChart} margin={{ right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                      <YAxis domain={['auto', 'auto']} />
                      <Tooltip />
                      <Line type="monotone" dataKey="shannon" stroke="#a29bfe"
                        strokeWidth={2} dot={{ r: 3 }} name="Shannon H'" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}

          {!tlLoading && !timelineChart.length && participant && (
            <div style={{ ...CARD, textAlign: 'center', color: '#b2bec3', padding: '40px' }}>
              No visits found for {participant}. Select genera above.
            </div>
          )}
        </div>
      )}

      {/* ── PROFILE ───────────────────────────────────────────────────────────── */}
      {tab === 'profile' && (
        <div>
          <div style={{ ...CARD, marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
            <label style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>Participant:</label>
            <select
              value={profParticipant}
              onChange={e => handleProfParticipantChange(e.target.value)}
              style={{ padding: '7px 12px', borderRadius: '4px', border: '1px solid #ddd', fontSize: '0.9rem' }}
            >
              <option value="">— select —</option>
              {participants.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
            {profLoading && <span style={{ color: '#636e72', fontSize: '0.85rem' }}>Loading…</span>}
            {!profLoading && profile && (
              <span style={{
                padding: '4px 12px', borderRadius: '12px', fontSize: '0.82rem', fontWeight: 'bold',
                backgroundColor: DX_COLOR[profile.diagnosis] || '#dfe6e9', color: 'white',
              }}>
                {profile.diagnosis || 'Unknown'}
              </span>
            )}
          </div>

          {!profLoading && profile && (
            <div style={CARD}>
              <h3 style={{ marginBottom: '4px' }}>Gut Health Indicators</h3>
              <p style={{ fontSize: '0.78rem', color: '#636e72', marginBottom: '16px' }}>
                Formula-based estimates from latest visit — {profile.participant_id}
              </p>

              {/* Gut health score bar */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem', marginBottom: '4px' }}>
                  <span style={{ fontWeight: 'bold' }}>Gut Health Score</span>
                  <span style={{ fontWeight: 'bold', color:
                    profile.gut_health_label === 'Good' ? '#27ae60'
                    : profile.gut_health_label === 'Moderate' ? '#f39c12' : '#e74c3c' }}>
                    {profile.gut_health_score} / 100 — {profile.gut_health_label}
                  </span>
                </div>
                <div style={{ background: '#eee', borderRadius: '4px', height: '8px' }}>
                  <div style={{
                    width: `${profile.gut_health_score}%`,
                    background: profile.gut_health_label === 'Good' ? '#27ae60'
                      : profile.gut_health_label === 'Moderate' ? '#f39c12' : '#e74c3c',
                    height: '8px', borderRadius: '4px', transition: 'width 0.4s',
                  }} />
                </div>
              </div>

              {/* Top genera */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '0.8rem', color: '#636e72', marginBottom: '8px', fontWeight: 'bold' }}>
                  Top genera (latest visit)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '4px' }}>
                  {(profile.top_genera || []).map(({ genus, abundance_pct }, i) => (
                    <div key={genus} style={{
                      display: 'flex', justifyContent: 'space-between',
                      padding: '5px 10px', backgroundColor: '#f7f9fc', borderRadius: '4px', fontSize: '0.83rem',
                    }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ width: '10px', height: '10px', borderRadius: '50%',
                          backgroundColor: COLORS[i % COLORS.length], display: 'inline-block' }} />
                        {genus}
                      </span>
                      <span style={{ fontWeight: 'bold' }}>{abundance_pct}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <MetricRow label="Shannon Diversity (H')" value={profile.shannon_diversity}
                info="Higher = greater species evenness" />
              <MetricRow label="Dominant Genus"         value={profile.dominant_genus} />
              <MetricRow label="Beneficial Bacteria"    value={profile.beneficial_pct} unit="%"
                info="Faecalibacterium + Akkermansia + Bifidobacterium + Roseburia" />
              <MetricRow label="F/B Ratio"              value={profile.fb_ratio}
                info="Firmicutes / Bacteroidetes proxy (Ley et al. 2006)" />
              <MetricRow label="B/P Ratio"              value={profile.bp_ratio}
                info="Bacteroides / Prevotella — dietary pattern (Wu et al. 2011)" />
              <MetricRow label="Inflammation Index"     value={profile.inflammation_index}
                info="Pro / anti-inflammatory genera ratio" />
              <MetricRow label="Dysbiosis Index"        value={profile.dysbiosis_index}
                info="Escherichia / (Faecalibacterium + Akkermansia)" />

              <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div style={{ padding: '10px 12px', backgroundColor: '#e8f8f7', borderRadius: '6px' }}>
                  <div style={{ fontSize: '0.75rem', color: '#636e72' }}>Enterotype</div>
                  <div style={{ fontWeight: 'bold', fontSize: '0.85rem', marginTop: '3px' }}>{profile.enterotype}</div>
                </div>
                <div style={{ padding: '10px 12px', backgroundColor: '#f0fafa', borderRadius: '6px' }}>
                  <div style={{ fontSize: '0.75rem', color: '#636e72' }}>Age Pattern</div>
                  <div style={{ fontWeight: 'bold', fontSize: '0.85rem', marginTop: '3px' }}>{profile.age_pattern}</div>
                </div>
              </div>
            </div>
          )}

          {!profLoading && !profile && (
            <div style={{ ...CARD, textAlign: 'center', color: '#b2bec3', padding: '40px' }}>
              Select a participant above to view their profile.
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MicrobiomePage;
