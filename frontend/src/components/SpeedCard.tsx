export function SpeedCard({ groundSpeed }: { groundSpeed: number | null }) {
  return (
    <div className="card">
      <h3>Ground Speed</h3>
      <p>{groundSpeed !== null ? `${groundSpeed.toFixed(1)} m/s` : "—"}</p>
    </div>
  );
}
