import { useEffect, useState } from "react";
import { StatusBar } from "../components/StatusBar";
import { api } from "../services/api";
import { useGcs } from "../state/GcsContext";
import type { SerialPort } from "../types";

const MODES = ["MANUAL", "AUTO", "HOLD", "RTL"];

export function Vehicle() {
  const { telemetry, connectionState } = useGcs();
  const [ports, setPorts] = useState<SerialPort[]>([]);
  const [selectedPort, setSelectedPort] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getPorts()
      .then((list) => {
        setPorts(list);
        if (list.length > 0) setSelectedPort(list[0].device);
      })
      .catch(() => undefined);
  }, []);

  const runAction = async (action: () => Promise<unknown>) => {
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="page">
      <h1>Vehicle Control</h1>
      <StatusBar state={connectionState} />

      <div className="control-row">
        <select value={selectedPort} onChange={(e) => setSelectedPort(e.target.value)}>
          {ports.length === 0 && <option value="">No ports found</option>}
          {ports.map((p) => (
            <option key={p.device} value={p.device}>
              {p.device} — {p.description}
            </option>
          ))}
        </select>
        <button disabled={!selectedPort} onClick={() => runAction(() => api.connect(selectedPort))}>
          Connect
        </button>
        <button onClick={() => runAction(() => api.disconnect())}>Disconnect</button>
      </div>

      <div className="control-row">
        <button onClick={() => runAction(() => api.arm())}>Arm</button>
        <button onClick={() => runAction(() => api.disarm())}>Disarm</button>
      </div>

      <div className="control-row">
        {MODES.map((mode) => (
          <button
            key={mode}
            className={telemetry.mode === mode ? "active" : ""}
            onClick={() => runAction(() => api.setMode(mode))}
          >
            {mode}
          </button>
        ))}
      </div>

      {error && <p className="error">{error}</p>}
    </div>
  );
}
