const MITS_MAP_EMBED_SRC =
  "https://www.google.com/maps?q=Muthoot+Institute+of+Technology+%26+Science,+Varikoli,+Puthencruz,+Kerala&t=k&z=18&output=embed";

export function MissionMap() {
  return (
    <section className="gcs-panel mission-map-panel">
      <h2 className="gcs-panel-title">Mission Map</h2>
      <div className="mission-map-frame">
        <iframe
          title="Mission Map"
          src={MITS_MAP_EMBED_SRC}
          loading="lazy"
          referrerPolicy="no-referrer-when-downgrade"
        />
      </div>
    </section>
  );
}
