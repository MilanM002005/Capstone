import { useGcs } from "../state/GcsContext";

export function Logs() {
  const { logs } = useGcs();

  return (
    <div className="page">
      <h1>Logs</h1>
      <div className="log-list">
        {logs.length === 0 && <p>No log entries yet.</p>}
        {logs
          .slice()
          .reverse()
          .map((entry, i) => (
            <div className={`log-entry log-${entry.level}`} key={i}>
              <span className="log-time">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
              <span>{entry.message}</span>
            </div>
          ))}
      </div>
    </div>
  );
}
