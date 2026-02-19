export interface ChatSession {
  id: string;
  title: string;
  model_name: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: DocumentSource[] | null;
  created_at: string;
}

export interface DocumentSource {
  source: string;
  page_content?: string;
  score?: number;
}

export interface ModelInfo {
  name: string;
  available: boolean;
}

export interface DocumentInfo {
  source: string;
  chunk_count?: number;
}

export interface ChatSettings {
  model_name: string;
  max_new_tokens: number;
  k: number;
  use_rag: boolean;
  synthesis_strategy: string;
}
