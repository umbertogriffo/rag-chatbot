import { describe, it, expect, beforeEach } from 'vitest';
import { useChatStore } from '../chatStore';

describe('chatStore', () => {
  beforeEach(() => {
    useChatStore.setState({
      sessions: [],
      currentSessionId: null,
      messages: [],
      isStreaming: false,
      streamingContent: '',
      sidebarOpen: true,
      settings: {
        model_name: 'llama-3.1',
        max_new_tokens: 512,
        k: 2,
        use_rag: true,
        synthesis_strategy: 'async-tree-summarization',
      },
    });
  });

  it('starts with default state', () => {
    const state = useChatStore.getState();
    expect(state.sessions).toEqual([]);
    expect(state.currentSessionId).toBeNull();
    expect(state.messages).toEqual([]);
    expect(state.isStreaming).toBe(false);
    expect(state.sidebarOpen).toBe(true);
  });

  it('adds a message', () => {
    const msg = {
      id: '1',
      role: 'user' as const,
      content: 'Hello',
      created_at: new Date().toISOString(),
    };
    useChatStore.getState().addMessage(msg);
    expect(useChatStore.getState().messages).toHaveLength(1);
    expect(useChatStore.getState().messages[0].content).toBe('Hello');
  });

  it('toggles sidebar', () => {
    expect(useChatStore.getState().sidebarOpen).toBe(true);
    useChatStore.getState().toggleSidebar();
    expect(useChatStore.getState().sidebarOpen).toBe(false);
    useChatStore.getState().toggleSidebar();
    expect(useChatStore.getState().sidebarOpen).toBe(true);
  });

  it('updates settings', () => {
    useChatStore.getState().updateSettings({ model_name: 'phi-3.5' });
    expect(useChatStore.getState().settings.model_name).toBe('phi-3.5');
    expect(useChatStore.getState().settings.k).toBe(2); // unchanged
  });

  it('appends streaming content', () => {
    useChatStore.getState().appendStreamingContent('Hello ');
    useChatStore.getState().appendStreamingContent('world');
    expect(useChatStore.getState().streamingContent).toBe('Hello world');
  });
});
