type EventName = "telemetry" | "connection" | "mode" | "log";
type Listener = (data: unknown) => void;

const RECONNECT_DELAY_MS = 2000;

class GcsWebSocket {
  private socket: WebSocket | null = null;
  private listeners = new Map<EventName, Set<Listener>>();
  private shouldReconnect = true;

  connect(): void {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${protocol}://${window.location.host}/ws`);

    this.socket.onmessage = (event) => {
      const parsed = JSON.parse(event.data) as { event: EventName; data: unknown };
      const handlers = this.listeners.get(parsed.event);
      handlers?.forEach((handler) => handler(parsed.data));
    };

    this.socket.onclose = () => {
      if (this.shouldReconnect) {
        setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
      }
    };
  }

  disconnect(): void {
    this.shouldReconnect = false;
    this.socket?.close();
  }

  on(event: EventName, handler: Listener): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }
}

export const gcsWebSocket = new GcsWebSocket();
