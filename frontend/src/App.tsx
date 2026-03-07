import { useState, useCallback, useMemo } from 'react';
import { ChatHeader, ChatViewport, ChatInput, type Message, type ChatModes } from '@/components/chat';
import { useChat } from '@/hooks/useChat';
import { useDocuments } from '@/hooks/useDocuments';

function App() {
  const { messages: rawMessages, isStreaming, sendMessage } = useChat();
  const { documents, uploading, setDocuments, setUploading, setError } = useDocuments();

  const [modes, setModes] = useState<ChatModes>({
    rag: false,
    reasoning: false,
    webSearch: false,
  });

  // Map hook's Message shape to template's Message shape
  const messages: Message[] = useMemo(
    () =>
      rawMessages.map((msg) => ({
        id: String(msg.id),
        role: msg.sender === 'user' ? ('user' as const) : ('assistant' as const),
        content: msg.text,
        isStreaming: msg.isStreaming,
      })),
    [rawMessages],
  );

  const handleSend = useCallback(
    (content: string) => {
      sendMessage(content, modes.rag);
    },
    [sendMessage, modes.rag],
  );

  const handleNewChat = useCallback(() => {
    window.location.reload();
  }, []);

  const handleUploadStart = useCallback(
    (filename: string) => {
      setError(null);
      setUploading({ filename, progress: 0 });
    },
    [setError, setUploading],
  );

  const handleUploadProgress = useCallback(
    (filename: string, progress: number) => {
      setUploading({ filename, progress });
    },
    [setUploading],
  );

  const handleUploadEnd = useCallback(() => {
    setUploading(null);
  }, [setUploading]);

  const handleError = useCallback(
    (message: string) => {
      setError(message);
    },
    [setError],
  );

  return (
    <div className="flex flex-col h-screen bg-background">
      <ChatHeader onNewChat={handleNewChat} />

      <main className="flex-1 flex flex-col min-h-0">
        <ChatViewport messages={messages} />

        <div className="shrink-0 border-t border-border/30 bg-background/50 backdrop-blur-sm">
          <ChatInput
            onSend={handleSend}
            isLoading={isStreaming}
            modes={modes}
            onModesChange={setModes}
            documents={documents}
            uploading={uploading}
            onDocumentsChange={setDocuments}
            onUploadStart={handleUploadStart}
            onUploadProgress={handleUploadProgress}
            onUploadEnd={handleUploadEnd}
            onError={handleError}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
