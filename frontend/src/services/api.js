import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 min timeout for model inference
});

export const healthCheck = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const predictImage = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await api.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      if (onProgress) {
        const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(pct);
      }
    },
  });
  return res.data;
};

export const getHistory = async () => {
  const res = await api.get('/api/history');
  return res.data;
};

export const clearHistory = async () => {
  const res = await api.delete('/api/history');
  return res.data;
};

export default api;
