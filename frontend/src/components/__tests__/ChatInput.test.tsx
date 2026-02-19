import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../test/test-utils';
import userEvent from '@testing-library/user-event';
import ChatInput from '../ChatInput';

describe('ChatInput', () => {
  it('renders the input textarea', () => {
    render(<ChatInput onSend={() => {}} />);
    expect(screen.getByPlaceholderText('Send a message...')).toBeInTheDocument();
  });

  it('calls onSend when clicking send button', async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<ChatInput onSend={onSend} />);

    const input = screen.getByPlaceholderText('Send a message...');
    await user.type(input, 'Hello world');

    const button = screen.getByRole('button');
    await user.click(button);

    expect(onSend).toHaveBeenCalledWith('Hello world');
  });

  it('clears input after sending', async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<ChatInput onSend={onSend} />);

    const input = screen.getByPlaceholderText('Send a message...');
    await user.type(input, 'Hello world');

    const button = screen.getByRole('button');
    await user.click(button);

    expect(input).toHaveValue('');
  });

  it('does not send empty messages', async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<ChatInput onSend={onSend} />);

    const button = screen.getByRole('button');
    await user.click(button);

    expect(onSend).not.toHaveBeenCalled();
  });

  it('disables input when disabled prop is true', () => {
    render(<ChatInput onSend={() => {}} disabled />);
    expect(screen.getByPlaceholderText('Send a message...')).toBeDisabled();
  });
});
