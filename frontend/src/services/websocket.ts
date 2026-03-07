export type TokenHandler = (token: string) => void;
export type ErrorHandler = (error: string) => void;

const WS_BASE = (() => {
  const apiUrl = import.meta.env.VITE_API_URL ?? '';
  if (apiUrl) {
    return apiUrl.replace(/^http/, 'ws');
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}`;
})();

const WS_URL = `${WS_BASE}/chat/stream`;

export class ChatWebSocket {
  private ws: WebSocket | null = null;
  private readonly onToken: TokenHandler;
  private readonly onError: ErrorHandler;

  constructor(onToken: TokenHandler, onError: ErrorHandler) {
    this.onToken = onToken;
    this.onError = onError;
  }

  private connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      if (this.ws?.readyState === WebSocket.CONNECTING) {
        // Wait for existing connection attempt
        const checkConnection = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            clearInterval(checkConnection);
            resolve();
          } else if (this.ws?.readyState === WebSocket.CLOSED) {
            clearInterval(checkConnection);
            reject(new Error('Connection failed'));
          }
        }, 50);
        return;
      }

      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        resolve();
      };

      this.ws.onmessage = (event) => {
        this.onToken(event.data as string);
      };

      this.ws.onerror = () => {
        this.onError('WebSocket connection error');
        reject(new Error('WebSocket connection error'));
      };

      this.ws.onclose = () => {
        this.ws = null;
        // No auto-reconnect
      };
    });
  }

  async sendMessage(text: string, rag: boolean): Promise<void> {
    try {
      await this.connect();
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ text, rag }));
      } else {
        this.onError('WebSocket is not connected');
      }
    } catch (error) {
      this.onError(error instanceof Error ? error.message : 'Failed to connect');
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }
}
