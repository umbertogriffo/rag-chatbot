import { useCallback, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { ChatWebSocket } from '../services/websocket';

export interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<ChatWebSocket | null>(null);
  const idRef = useRef(0);

  useEffect(() => {
    const ws = new ChatWebSocket(
      (token) => {
        flushSync(() => {
          setIsStreaming(false);
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.sender !== 'bot') return prev;
            const updated = { ...last, text: last.text + token };
            return [...prev.slice(0, -1), updated];
          });
        });
      },
      (error) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.sender === 'bot') {
            const updated = { ...last, text: `Error: ${error}` };
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
    return () => ws.disconnect();
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
    };

    setMessages((prev) => [...prev, userMsg, botPlaceholder]);
    setIsStreaming(true);
    wsRef.current?.sendMessage(text, rag);
  }, [isStreaming]);

  return { messages, isStreaming, sendMessage };
}
