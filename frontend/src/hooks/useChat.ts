import { useCallback, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { ChatWebSocket } from '../services/websocket';

export interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isStreaming?: boolean;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<ChatWebSocket | null>(null);
  const idRef = useRef(0);
  const streamingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const ws = new ChatWebSocket(
      (token) => {
        flushSync(() => {
          // Clear any existing timeout
          if (streamingTimeoutRef.current) {
            clearTimeout(streamingTimeoutRef.current);
          }

          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.sender !== 'bot') return prev;
            const updated = { ...last, text: last.text + token, isStreaming: true };
            return [...prev.slice(0, -1), updated];
          });

          // Set timeout to mark streaming as done after no tokens for 500ms
          streamingTimeoutRef.current = setTimeout(() => {
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last?.sender === 'bot' && last.isStreaming) {
                return [...prev.slice(0, -1), { ...last, isStreaming: false }];
              }
              return prev;
            });
            setIsStreaming(false);
          }, 500);
        });
      },
      (error) => {
        if (streamingTimeoutRef.current) {
          clearTimeout(streamingTimeoutRef.current);
        }
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.sender === 'bot') {
            const updated = { ...last, text: `Error: ${error}`, isStreaming: false };
            return [...prev.slice(0, -1), updated];
          }
          return [
            ...prev,
            {
              id: ++idRef.current,
              text: `Error: ${error}`,
              sender: 'bot',
              timestamp: new Date(),
            },
          ];
        });
        setIsStreaming(false);
      },
    );
    wsRef.current = ws;
    return () => {
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
      }
      ws.disconnect();
    };
  }, []);

  const sendMessage = useCallback((text: string, rag: boolean) => {
    if (!text.trim() || isStreaming) return;

    const userMsg: Message = {
      id: ++idRef.current,
      text,
      sender: 'user',
      timestamp: new Date(),
    };
    const botPlaceholder: Message = {
      id: ++idRef.current,
      text: '',
      sender: 'bot',
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, botPlaceholder]);
    setIsStreaming(true);
    wsRef.current?.sendMessage(text, rag);
  }, [isStreaming]);

  return { messages, isStreaming, sendMessage };
}
