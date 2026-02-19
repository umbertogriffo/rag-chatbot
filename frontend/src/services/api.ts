import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

// Auth
export const getToken = async (username: string) => {
  const { data } = await apiClient.post('/auth/token', { username });
  return data;
};

// Health
export const getHealth = async () => {
  const { data } = await apiClient.get('/health');
  return data;
};

// Models
export const getModels = async () => {
  const { data } = await apiClient.get('/models/');
  return data;
};

export const getStrategies = async () => {
  const { data } = await apiClient.get('/models/strategies');
  return data;
};

// Chat Sessions
export const getSessions = async () => {
  const { data } = await apiClient.get('/chat/sessions');
  return data;
};

export const createSession = async (modelName?: string) => {
  const { data } = await apiClient.post('/chat/sessions', null, {
    params: modelName ? { model_name: modelName } : {},
  });
  return data;
};

export const deleteSession = async (sessionId: string) => {
  const { data } = await apiClient.delete(`/chat/sessions/${sessionId}`);
  return data;
};

export const getSessionMessages = async (sessionId: string) => {
  const { data } = await apiClient.get(`/chat/sessions/${sessionId}/messages`);
  return data;
};

// Chat
export const sendMessage = async (params: {
  message: string;
  session_id?: string;
  model_name?: string;
  max_new_tokens?: number;
  k?: number;
  use_rag?: boolean;
  synthesis_strategy?: string;
}) => {
  const { data } = await apiClient.post('/chat/', params);
  return data;
};

// Documents
export const getDocuments = async () => {
  const { data } = await apiClient.get('/documents/');
  return data;
};

export const uploadDocument = async (file: File, chunkSize?: number, chunkOverlap?: number) => {
  const formData = new FormData();
  formData.append('file', file);
  const params = new URLSearchParams();
  if (chunkSize) params.set('chunk_size', String(chunkSize));
  if (chunkOverlap) params.set('chunk_overlap', String(chunkOverlap));

  const { data } = await apiClient.post(`/documents/upload?${params.toString()}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

// WebSocket
export const createChatWebSocket = (sessionId: string): WebSocket => {
  const wsBase = API_BASE_URL.replace(/^http/, 'ws').replace(/\/api$/, '');
  return new WebSocket(`${wsBase}/api/chat/ws/${sessionId}`);
};
