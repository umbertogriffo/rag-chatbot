export type TokenHandler = (token: string, done: boolean) => void;
export type ErrorHandler = (error: string) => void;

const WS_BASE = (() => {
  const apiUrl = import.meta.env.VITE_API_URL ?? '';
  if (apiUrl) {
    return apiUrl.replace(/^http/, 'ws');
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}`;
})();

const WS_URL = `${WS_BASE}/api/chat/stream`;

export class ChatWebSocket {
  private ws: WebSocket | null = null;
  private onToken: TokenHandler;
  private onError: ErrorHandler;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 16000;
  private shouldReconnect = false;

  constructor(onToken: TokenHandler, onError: ErrorHandler) {
    this.onToken = onToken;
    this.onError = onError;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
      return;
    }
    this.shouldReconnect = true;
    this.ws = new WebSocket(WS_URL);

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string) as {
          token?: string;
          done?: boolean;
          error?: string;
        };
        if (data.error) {
          this.onError(data.error);
        } else {
          this.onToken(data.token ?? '', data.done ?? false);
        }
      } catch {
        this.onError('Failed to parse server message');
      }
    };

    this.ws.onerror = () => {
      this.onError('WebSocket connection error');
    };

    this.ws.onclose = () => {
      this.scheduleReconnect();
    };
  }

  sendMessage(text: string, rag: boolean): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ text, rag }));
    } else {
      this.onError('WebSocket is not connected');
    }
  }

  disconnect(): void {
    this.shouldReconnect = false;
    this.ws?.close();
    this.ws = null;
  }

  private scheduleReconnect(): void {
    if (!this.shouldReconnect) return;
    setTimeout(() => this.connect(), this.reconnectDelay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
  }
}
