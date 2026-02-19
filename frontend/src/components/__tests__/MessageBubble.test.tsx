import { describe, it, expect } from 'vitest';
import { render, screen } from '../../test/test-utils';
import MessageBubble from '../MessageBubble';

describe('MessageBubble', () => {
  it('renders user message', () => {
    render(
      <MessageBubble
        message={{
          id: '1',
          role: 'user',
          content: 'Hello',
          created_at: new Date().toISOString(),
        }}
      />
    );
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('renders assistant message', () => {
    render(
      <MessageBubble
        message={{
          id: '2',
          role: 'assistant',
          content: 'Hi there!',
          created_at: new Date().toISOString(),
        }}
      />
    );
    expect(screen.getByText('Assistant')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
  });

  it('renders sources when provided', () => {
    render(
      <MessageBubble
        message={{
          id: '3',
          role: 'assistant',
          content: 'Answer with source',
          sources: [{ source: 'test.md', score: 0.95 }],
          created_at: new Date().toISOString(),
        }}
      />
    );
    expect(screen.getByText('Sources:')).toBeInTheDocument();
    expect(screen.getByText(/test\.md/)).toBeInTheDocument();
  });

  it('renders markdown content', () => {
    render(
      <MessageBubble
        message={{
          id: '4',
          role: 'assistant',
          content: '**bold text**',
          created_at: new Date().toISOString(),
        }}
      />
    );
    expect(screen.getByText('bold text')).toBeInTheDocument();
  });
});
