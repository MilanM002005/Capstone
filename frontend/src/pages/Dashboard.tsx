import { BottomStatusBar } from "../components/BottomStatusBar";
import { ControlPanel } from "../components/ControlPanel";
import { GcsHeader } from "../components/GcsHeader";
import { MissionMap } from "../components/MissionMap";
import { MissionPlanner } from "../components/MissionPlanner";
import { useGcs } from "../state/GcsContext";

function missionStatusLabel(armed: boolean, connected: boolean): string {
  if (!connected) return "Standby";
  return armed ? "Armed" : "Idle";
}

export function Dashboard() {
  const { telemetry, connectionState } = useGcs();

  return (
    <div className="gcs-shell">
      <GcsHeader
        missionStatus={missionStatusLabel(telemetry.armed, telemetry.connected)}
        connectionState={connectionState}
      />
      <main className="gcs-main">
        <ControlPanel />
        <MissionMap />
        <MissionPlanner />
      </main>
      <BottomStatusBar telemetry={telemetry} connectionState={connectionState} />
    </div>
  );
}
