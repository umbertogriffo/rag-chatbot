import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot } from 'lucide-react';
import type { ChatMessage } from '../types';

interface MessageBubbleProps {
  message: ChatMessage;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-4 px-4 py-6 ${isUser ? 'bg-transparent' : 'bg-zinc-800/50'}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? 'bg-blue-600' : 'bg-emerald-600'
        }`}
      >
        {isUser ? <User size={16} className="text-white" /> : <Bot size={16} className="text-white" />}
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 text-sm font-medium text-zinc-400">{isUser ? 'You' : 'Assistant'}</div>
        <div className="prose prose-invert max-w-none text-zinc-200">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 border-t border-zinc-700 pt-3">
            <div className="text-xs font-medium text-zinc-500">Sources:</div>
            {message.sources.map((source, i) => (
              <div key={i} className="mt-1 text-xs text-zinc-400">
                📄 {source.source}
                {source.score !== undefined && (
                  <span className="ml-2 text-zinc-500">({(source.score * 100).toFixed(1)}%)</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
