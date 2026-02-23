import { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import DocumentUpload from './components/DocumentUpload';
import SearchBar from './components/SearchBar';
import { useChat } from './hooks/useChat';
import { useDocuments } from './hooks/useDocuments';
import robotLogo from './assets/robot-logo.png';

function App() {
  const { messages, isStreaming, sendMessage } = useChat();
  const { documents, uploading, error: docError, upload, remove } = useDocuments();
  const [showDocs, setShowDocs] = useState(false);

  const handleSearch = (query: string, options: { rag: boolean; reasoning: boolean; webSearch: boolean }) => {
    sendMessage(query, options.rag);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-screen w-screen bg-gray-950 text-gray-100 overflow-hidden">
      {/* Main column */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <img src={robotLogo} alt="RAG Chatbot" className="h-8 w-8 object-contain" />
            <span className="font-semibold text-lg">RAG Chatbot</span>
          </div>
          <button
            onClick={() => setShowDocs((v) => !v)}
            className="flex items-center gap-1 rounded-lg px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 transition-colors"
          >
            📚 {showDocs ? 'Hide' : 'Documents'}
            {documents.length > 0 && (
              <span className="ml-1 rounded-full bg-green-600 px-1.5 py-0.5 text-xs font-bold">
                {documents.length}
              </span>
            )}
          </button>
        </header>

        {/* Chat area */}
        <div className="flex flex-1 flex-col min-h-0">
          {isEmpty ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 px-4">
              <img src={robotLogo} alt="Robot" className="h-16 w-16 opacity-30" />
              <h1 className="text-2xl font-semibold text-gray-400">What can I help with?</h1>
              <p className="text-sm text-gray-600">Upload documents, then ask questions.</p>
            </div>
          ) : (
            <ChatWindow messages={messages} />
          )}

          {/* Input bar */}
          <div className="px-4 pb-4 pt-2 border-t border-gray-800">
            <div className="mx-auto max-w-3xl">
              <SearchBar onSearch={handleSearch} disabled={isStreaming} />
            </div>
          </div>
        </div>
      </div>

      {/* Documents panel */}
      {showDocs && (
        <aside className="w-72 border-l border-gray-800 flex flex-col bg-gray-900">
          <div className="px-4 py-3 border-b border-gray-800 font-semibold text-sm">
            Knowledge Base
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <DocumentUpload
              documents={documents}
              uploading={uploading}
              error={docError}
              onUpload={upload}
              onDelete={remove}
            />
          </div>
        </aside>
      )}
    </div>
  );
}

export default App;

