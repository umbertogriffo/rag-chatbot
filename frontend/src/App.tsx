import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { useChatStore } from './store/chatStore';
import { Menu } from 'lucide-react';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppLayout() {
  const { sidebarOpen, setSidebarOpen } = useChatStore();

  return (
    <div className="flex h-screen bg-zinc-900 text-zinc-200">
      <Sidebar />
      <div className="flex flex-1 flex-col">
        {!sidebarOpen && (
          <div className="border-b border-zinc-700 p-2">
            <button
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Menu size={20} />
            </button>
          </div>
        )}
        <ChatArea />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppLayout />
      <Toaster position="bottom-right" />
    </QueryClientProvider>
  );
}
