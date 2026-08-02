import { useState } from "react";
import { api } from "../services/api";

// Placeholders — no backend support yet (mission workflows are out of
// scope for Phase 1). Wire these up when mission upload/start/pause/
// resume/RTL-as-mission-step lands.
function startMission(): void {}
function pauseMission(): void {}
function resumeMission(): void {}
function emergencyStop(): void {}

export function ControlPanel() {
  const [error, setError] = useState<string | null>(null);

  const runAction = async (action: () => Promise<unknown>) => {
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <section className="gcs-panel control-panel">
      <h2 className="gcs-panel-title">Vehicle Control</h2>

      <button className="gcs-btn gcs-btn-success" onClick={() => runAction(() => api.arm())}>
        ARM
      </button>
      <button className="gcs-btn gcs-btn-danger" onClick={() => runAction(() => api.disarm())}>
        DISARM
      </button>
      <button className="gcs-btn" onClick={startMission}>
        START MISSION
      </button>
      <button className="gcs-btn" onClick={pauseMission}>
        PAUSE
      </button>
      <button className="gcs-btn" onClick={resumeMission}>
        RESUME
      </button>
      <button className="gcs-btn gcs-btn-accent" onClick={() => runAction(() => api.setMode("RTL"))}>
        RTL
      </button>
      <button className="gcs-btn gcs-btn-danger gcs-btn-emergency" onClick={emergencyStop}>
        EMERGENCY STOP
      </button>

      {error && <p className="error">{error}</p>}
    </section>
  );
}
