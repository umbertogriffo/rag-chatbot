import { useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useChatStore } from '../store/chatStore';
import * as api from '../services/api';
import type { ChatMessage } from '../types';

export function useSessions() {
  const { setSessions } = useChatStore();
  return useQuery({
    queryKey: ['sessions'],
    queryFn: async () => {
      const data = await api.getSessions();
      setSessions(data);
      return data;
    },
  });
}

export function useSessionMessages(sessionId: string | null) {
  const { setMessages } = useChatStore();
  return useQuery({
    queryKey: ['messages', sessionId],
    queryFn: async () => {
      if (!sessionId) return [];
      const data = await api.getSessionMessages(sessionId);
      setMessages(data);
      return data;
    },
    enabled: !!sessionId,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: api.getModels,
  });
}

export function useStrategies() {
  return useQuery({
    queryKey: ['strategies'],
    queryFn: api.getStrategies,
  });
}

export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: api.getDocuments,
  });
}

export function useSendMessage() {
  const queryClient = useQueryClient();
  const {
    currentSessionId,
    settings,
    addMessage,
    setIsStreaming,
    setStreamingContent,
    appendStreamingContent,
    setCurrentSessionId,
  } = useChatStore();
  const wsRef = useRef<WebSocket | null>(null);

  const sendViaHttp = useMutation({
    mutationFn: async (message: string) => {
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: message,
        created_at: new Date().toISOString(),
      };
      addMessage(userMsg);
      setIsStreaming(true);
      setStreamingContent('');

      const response = await api.sendMessage({
        message,
        session_id: currentSessionId || undefined,
        ...settings,
      });

      if (!currentSessionId) {
        setCurrentSessionId(response.session_id);
      }

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.message,
        sources: response.sources,
        created_at: new Date().toISOString(),
      };
      addMessage(assistantMsg);
      setIsStreaming(false);
      setStreamingContent('');

      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      return response;
    },
  });

  const sendViaWebSocket = useCallback(
    (message: string, sessionId: string) => {
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: message,
        created_at: new Date().toISOString(),
      };
      addMessage(userMsg);
      setIsStreaming(true);
      setStreamingContent('');

      if (wsRef.current) {
        wsRef.current.close();
      }

      const ws = api.createChatWebSocket(sessionId);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            message,
            ...settings,
          })
        );
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          appendStreamingContent(data.content);
        } else if (data.type === 'done') {
          const assistantMsg: ChatMessage = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: data.content,
            created_at: new Date().toISOString(),
          };
          addMessage(assistantMsg);
          setIsStreaming(false);
          setStreamingContent('');
          queryClient.invalidateQueries({ queryKey: ['sessions'] });
        }
      };

      ws.onerror = () => {
        setIsStreaming(false);
        setStreamingContent('');
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    },
    [settings, addMessage, setIsStreaming, setStreamingContent, appendStreamingContent, queryClient]
  );

  return { sendViaHttp, sendViaWebSocket };
}

export function useCreateSession() {
  const queryClient = useQueryClient();
  const { setCurrentSessionId, setMessages, settings } = useChatStore();

  return useMutation({
    mutationFn: async () => {
      const session = await api.createSession(settings.model_name);
      setCurrentSessionId(session.id);
      setMessages([]);
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
      return session;
    },
  });
}

export function useDeleteSession() {
  const queryClient = useQueryClient();
  const { currentSessionId, setCurrentSessionId, setMessages } = useChatStore();

  return useMutation({
    mutationFn: async (sessionId: string) => {
      await api.deleteSession(sessionId);
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
        setMessages([]);
      }
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (file: File) => {
      const result = await api.uploadDocument(file);
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      return result;
    },
  });
}
