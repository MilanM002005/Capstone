export function GPSCard({
  latitude,
  longitude,
  fix,
  satellites,
}: {
  latitude: number | null;
  longitude: number | null;
  fix: string | null;
  satellites: number | null;
}) {
  return (
    <div className="card">
      <h3>GPS</h3>
      <p>Lat: {latitude !== null ? latitude.toFixed(6) : "—"}</p>
      <p>Lon: {longitude !== null ? longitude.toFixed(6) : "—"}</p>
      <p>Fix: {fix ?? "—"}</p>
      <p>Satellites: {satellites ?? "—"}</p>
    </div>
  );
}
