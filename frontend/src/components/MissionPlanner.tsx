import { useState } from "react";

interface CoordinateFields {
  latitude: string;
  longitude: string;
}

const EMPTY_COORDS: CoordinateFields = { latitude: "", longitude: "" };

// Placeholder — mission upload isn't implemented in Phase 1. This just
// holds form state locally until the mission-planning backend exists.
function addMission(_start: CoordinateFields, _destination: CoordinateFields): void {}

export function MissionPlanner() {
  const [start, setStart] = useState<CoordinateFields>(EMPTY_COORDS);
  const [destination, setDestination] = useState<CoordinateFields>(EMPTY_COORDS);

  const handleClear = () => {
    setStart(EMPTY_COORDS);
    setDestination(EMPTY_COORDS);
  };

  return (
    <section className="gcs-panel">
      <h2 className="gcs-panel-title">Mission Planner</h2>

      <div className="mission-planner-section">
        <h4>Starting Coordinates</h4>
        <div className="coord-fields">
          <div className="coord-field">
            <label htmlFor="start-lat">Latitude</label>
            <input
              id="start-lat"
              type="text"
              value={start.latitude}
              onChange={(e) => setStart({ ...start, latitude: e.target.value })}
            />
          </div>
          <div className="coord-field">
            <label htmlFor="start-lon">Longitude</label>
            <input
              id="start-lon"
              type="text"
              value={start.longitude}
              onChange={(e) => setStart({ ...start, longitude: e.target.value })}
            />
          </div>
        </div>
      </div>

      <div className="mission-planner-section">
        <h4>Destination Coordinates</h4>
        <div className="coord-fields">
          <div className="coord-field">
            <label htmlFor="dest-lat">Latitude</label>
            <input
              id="dest-lat"
              type="text"
              value={destination.latitude}
              onChange={(e) => setDestination({ ...destination, latitude: e.target.value })}
            />
          </div>
          <div className="coord-field">
            <label htmlFor="dest-lon">Longitude</label>
            <input
              id="dest-lon"
              type="text"
              value={destination.longitude}
              onChange={(e) => setDestination({ ...destination, longitude: e.target.value })}
            />
          </div>
        </div>
      </div>

      <div className="mission-planner-actions">
        <button className="gcs-btn gcs-btn-accent" onClick={() => addMission(start, destination)}>
          Add Mission
        </button>
        <button className="gcs-btn" onClick={handleClear}>
          Clear
        </button>
      </div>
    </section>
  );
}
