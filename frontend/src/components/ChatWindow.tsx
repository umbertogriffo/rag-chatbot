import React from 'react';
import Markdown from 'react-markdown';

export interface Message {
  id?: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  streaming?: boolean;
}

interface ChatWindowProps {
  messages: Message[];
}

const MessageBubble: React.FC<Message> = ({ text, sender, timestamp, streaming }) => (
  <div className="flex items-start gap-3 mb-4">
    <div
      className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center text-sm font-bold ${
        sender === 'user' ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-200'
      }`}
    >
      {sender === 'user' ? 'U' : '🤖'}
    </div>
    <div className="flex-1 min-w-0">
      <div
        className={`rounded-2xl px-4 py-2 text-sm leading-relaxed ${
          sender === 'user'
            ? 'bg-green-700 text-white'
            : 'bg-gray-800 text-gray-100'
        }`}
      >
        <div className="prose prose-invert prose-sm max-w-none">
          <Markdown>{text}</Markdown>
        </div>
        {streaming && (
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-green-400 animate-pulse align-middle" />
        )}
      </div>
      <span className="block mt-1 text-xs text-gray-500">
        {timestamp.toLocaleTimeString()}
      </span>
    </div>
  </div>
);

const ChatWindow: React.FC<ChatWindowProps> = ({ messages }) => {
  const endRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) return null;

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-1 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
      {messages.map((msg, idx) => (
        <MessageBubble key={msg.id ?? idx} {...msg} />
      ))}
      <div ref={endRef} />
    </div>
  );
};

export default ChatWindow;

