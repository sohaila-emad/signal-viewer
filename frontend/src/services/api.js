import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Add request interceptor for logging
api.interceptors.request.use(request => {
  console.log('Starting Request:', request.method, request.url);
  return request;
});

// Add response interceptor for logging
api.interceptors.response.use(
  response => {
    console.log('Response:', response.status, response.data);
    return response;
  },
  error => {
    console.error('API Error:', error.message);
    if (error.response) {
      console.error('Status:', error.response.status);
      console.error('Data:', error.response.data);
    }
    return Promise.reject(error);
  }
);

export const medicalAPI = {
  getSignals: () => api.get('/medical/signals'),
  getSignal: (id) => api.get(`/medical/signal/${id}`),
  predict: (id) => api.get(`/medical/predict/${id}`),
};

export const acousticAPI = {
  test: () => api.get('/acoustic/test'),
  generateDoppler: (data) => api.post('/acoustic/doppler/generate', data),
  getDopplerParameters: (velocity, frequency) => 
    api.get('/acoustic/doppler/parameters', { params: { velocity, frequency } }),
  analyzeVehicle: (audioData, sampleRate) => 
    api.post('/acoustic/vehicle/analyze', { audio_data: audioData, sample_rate: sampleRate }),
  detectVehicle: (audioData, sampleRate, vehicleType) => 
    api.post('/acoustic/vehicle/detect', { 
      audio_data: audioData, 
      sample_rate: sampleRate,
      vehicle_type: vehicleType 
    }),
  computeSpectrogram: (audioData, sampleRate, nperseg) => 
    api.post('/acoustic/spectrogram', { 
      audio_data: audioData, 
      sample_rate: sampleRate,
      nperseg: nperseg || 256 
    }),
};

export const stockAPI = {
  // Stock data
  getStockData: (symbol, period = '1y', interval = '1d') => 
    api.get(`/stock/data/${symbol}`, { params: { period, interval } }),
  getStockInfo: (symbol) => api.get(`/stock/info/${symbol}`),
  getStockList: (category = 'tech') => api.get(`/stock/list/${category}`),
  
  // Currency data
  getCurrencyList: (category = 'major') => api.get(`/stock/currency/list/${category}`),
  getCurrencyData: (symbol, period = '1y', interval = '1d') => 
    api.get(`/stock/currency/data/${symbol}`, { params: { period, interval } }),
  
  // Mineral/commodity data
  getMineralList: (category = 'precious') => api.get(`/stock/mineral/list/${category}`),
  getMineralData: (symbol, period = '1y', interval = '1d') => 
    api.get(`/stock/mineral/data/${symbol}`, { params: { period, interval } }),
  
  // Prediction
  predictStock: (symbol, method = 'sma', nDays = 7) => 
    api.get(`/stock/predict/${symbol}`, { params: { method, n_days: nDays } }),
  
  // Technical analysis
  getTechnicalAnalysis: (symbol) => api.get(`/stock/analysis/${symbol}`),
  
  // Comparison
  compareStocks: (symbols, period = '1y') => 
    api.post('/stock/compare', { symbols, period }),
  
  // Market summary
  getMarketSummary: () => api.get('/stock/market/summary'),
};

export const microbiomeAPI = {
  // Sample management
  getSamples: () => api.get('/microbiome/'),
  getSample: (sampleId) => api.get(`/microbiome/${sampleId}`),
  createSample: (data) => api.post('/microbiome/', data),
  deleteSample: (sampleId) => api.delete(`/microbiome/${sampleId}`),
  
  // Analysis
  analyzeDiversity: (samples) => 
    api.post('/microbiome/analyze-diversity', { samples }),
    
  // NEW ENDPOINTS
  getSummary: () => api.get('/microbiome/summary'),
  getDiseases: () => api.get('/microbiome/diseases'),
  getTaxa: () => api.get('/microbiome/taxa'),
  analyze: (data) => api.post('/microbiome/analyze', data),
  estimatePatient: (data) => api.post('/microbiome/estimate-patient', data),
  getStatistics: () => api.get('/microbiome/statistics'),
  getDiversity: () => api.get('/microbiome/diversity'),
  compareSamples: (sample1Id, sample2Id) => 
    api.post('/microbiome/compare', { sample1_id: sample1Id, sample2_id: sample2Id }),
  getComposition: () => api.get('/microbiome/composition'),
  getDiseaseSamples: (diseaseName) => api.get(`/microbiome/disease/${diseaseName}`),
  loadData: (nSamples) => api.post('/microbiome/load-data', { n_samples: nSamples }),
};

export default api;
