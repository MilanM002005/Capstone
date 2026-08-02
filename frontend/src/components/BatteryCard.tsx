export function BatteryCard({
  voltage,
  remaining,
}: {
  voltage: number | null;
  remaining: number | null;
}) {
  return (
    <div className="card">
      <h3>Battery</h3>
      <p>{voltage !== null ? `${voltage.toFixed(2)} V` : "—"}</p>
      <p>{remaining !== null ? `${remaining}%` : "—"}</p>
    </div>
  );
}
