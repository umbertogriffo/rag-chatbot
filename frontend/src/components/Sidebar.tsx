import { useRef } from 'react';
import { Plus, Trash2, Upload, FileText, X, MessageSquare } from 'lucide-react';
import { useChatStore } from '../store/chatStore';
import {
  useSessions,
  useSessionMessages,
  useCreateSession,
  useDeleteSession,
  useDocuments,
  useUploadDocument,
} from '../hooks/useChat';
import ModelSelector from './ModelSelector';
import SettingsPanel from './SettingsPanel';

export default function Sidebar() {
  const { currentSessionId, setCurrentSessionId, setMessages, sidebarOpen, setSidebarOpen } = useChatStore();
  const { data: sessions } = useSessions();
  const { refetch: refetchMessages } = useSessionMessages(currentSessionId);
  const createSession = useCreateSession();
  const deleteSession = useDeleteSession();
  const { data: docsData } = useDocuments();
  const uploadDoc = useUploadDocument();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSessionClick = (sessionId: string) => {
    setCurrentSessionId(sessionId);
    refetchMessages();
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    setMessages([]);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      uploadDoc.mutate(file);
    }
  };

  if (!sidebarOpen) return null;

  return (
    <div className="flex h-full w-72 flex-col border-r border-zinc-700 bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-700 p-4">
        <h1 className="text-sm font-semibold text-zinc-200">RAG Chatbot</h1>
        <button onClick={() => setSidebarOpen(false)} className="text-zinc-400 hover:text-zinc-200">
          <X size={18} />
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={handleNewChat}
          className="flex w-full items-center gap-2 rounded-lg border border-zinc-700 px-3 py-2 text-sm text-zinc-300 transition-colors hover:bg-zinc-800"
        >
          <Plus size={16} />
          New Chat
        </button>
      </div>

      {/* Chat Sessions */}
      <div className="flex-1 overflow-y-auto px-3">
        <div className="mb-2 text-xs font-medium text-zinc-500">Recent Chats</div>
        {sessions?.map((session: { id: string; title: string }) => (
          <div
            key={session.id}
            className={`group mb-1 flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${
              currentSessionId === session.id
                ? 'bg-zinc-800 text-zinc-200'
                : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300'
            }`}
            onClick={() => handleSessionClick(session.id)}
          >
            <MessageSquare size={14} className="shrink-0" />
            <span className="flex-1 truncate">{session.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteSession.mutate(session.id);
              }}
              className="hidden text-zinc-500 hover:text-red-400 group-hover:block"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>

      {/* Documents Section */}
      <div className="border-t border-zinc-700 p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-500">Documents</span>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="text-zinc-400 hover:text-zinc-200"
          >
            <Upload size={14} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".md,.txt,.pdf"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
        <div className="max-h-24 overflow-y-auto">
          {docsData?.documents?.map((doc: { source: string }, i: number) => (
            <div key={i} className="flex items-center gap-2 py-1 text-xs text-zinc-400">
              <FileText size={12} />
              <span className="truncate">{doc.source}</span>
            </div>
          ))}
          {(!docsData?.documents || docsData.documents.length === 0) && (
            <p className="text-xs text-zinc-600">No documents indexed</p>
          )}
        </div>
      </div>

      {/* Model Selector */}
      <ModelSelector />

      {/* Settings */}
      <SettingsPanel />
    </div>
  );
}
