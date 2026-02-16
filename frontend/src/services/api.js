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
  generateDoppler: (data) => api.post('/acoustic/generate', data),
};

export default api;
