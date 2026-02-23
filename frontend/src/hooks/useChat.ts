import { useCallback, useEffect, useRef, useState } from 'react';
import { ChatWebSocket } from '../services/websocket';

export interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  streaming?: boolean;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<ChatWebSocket | null>(null);
  const idRef = useRef(0);

  useEffect(() => {
    const ws = new ChatWebSocket(
      (token, done) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.sender !== 'bot' || !last.streaming) return prev;
          const updated = { ...last, text: last.text + token, streaming: !done };
          return [...prev.slice(0, -1), updated];
        });
        if (done) setIsStreaming(false);
      },
      (error) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.sender === 'bot' && last.streaming) {
            const updated = { ...last, text: `Error: ${error}`, streaming: false };
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
    ws.connect();
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
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, botPlaceholder]);
    setIsStreaming(true);
    wsRef.current?.sendMessage(text, rag);
  }, [isStreaming]);

  return { messages, isStreaming, sendMessage };
}
