export function HeadingCard({ heading }: { heading: number | null }) {
  return (
    <div className="card">
      <h3>Heading</h3>
      <p>{heading !== null ? `${heading.toFixed(0)}°` : "—"}</p>
    </div>
  );
}
