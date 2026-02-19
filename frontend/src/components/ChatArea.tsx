import { useEffect, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';
import { useSendMessage } from '../hooks/useChat';
import { Bot } from 'lucide-react';

export default function ChatArea() {
  const { messages, isStreaming, streamingContent } = useChatStore();
  const { sendViaHttp } = useSendMessage();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  const handleSend = (message: string) => {
    sendViaHttp.mutate(message);
  };

  return (
    <div className="flex flex-1 flex-col bg-zinc-900">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && !isStreaming ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-zinc-800">
                <Bot size={32} className="text-zinc-500" />
              </div>
              <h2 className="text-xl font-semibold text-zinc-300">RAG Chatbot</h2>
              <p className="mt-2 text-sm text-zinc-500">
                Ask a question about your documents or start a conversation.
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isStreaming && streamingContent && (
              <MessageBubble
                message={{
                  id: 'streaming',
                  role: 'assistant',
                  content: streamingContent + '▊',
                  created_at: new Date().toISOString(),
                }}
              />
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </div>
  );
}
