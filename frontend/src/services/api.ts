import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export interface DocumentInfo {
  document_id: string;
  filename: string;
  size: number;
  content_type: string;
}

interface DocumentUploadResponse {
  document_id: string;
  filename: string;
}

interface DocumentListResponse {
  documents: DocumentInfo[];
}

export async function uploadDocument(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post<DocumentUploadResponse>(
    `${API_BASE}/documents`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (event) => {
        if (onProgress && event.total) {
          onProgress(Math.round((event.loaded * 100) / event.total));
        }
      },
    },
  );
  return response.data;
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const response = await axios.get<DocumentListResponse>(`${API_BASE}/documents`);
  return response.data;
}

export async function deleteDocument(documentId: string): Promise<void> {
  await axios.delete(`${API_BASE}/documents/${documentId}`);
}

export async function resetChatHistory(): Promise<void> {
  await axios.delete(`${API_BASE}/chat/history`);
}
