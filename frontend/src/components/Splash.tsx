import { useEffect, useState } from "react";

interface SplashProps {
  exiting: boolean;
  onDismiss: () => void;
}

export function Splash({ exiting, onDismiss }: SplashProps) {
  const [entered, setEntered] = useState(false);

  useEffect(() => {
    const frame = requestAnimationFrame(() => setEntered(true));
    return () => cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Enter" || event.key === " ") onDismiss();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onDismiss]);

  const phaseClass = exiting ? "splash-out" : entered ? "splash-in" : "";

  return (
    <div
      className={`splash-overlay ${phaseClass}`}
      role="button"
      tabIndex={0}
      aria-label="Dismiss splash screen"
      onClick={onDismiss}
      onTouchEnd={onDismiss}
    >
      <div className="splash-band-white">
        <img className="splash-band-logo" src="/mits_logo.png" alt="Muthoot Institute of Technology & Science" />
      </div>
      <div className="splash-band-red">
        <img className="splash-band-logo" src="/dept_logo.png" alt="Department of Artificial Intelligence and Data Science" />
        <p className="splash-caption">Ground Control Station</p>
        <p className="splash-hint">Tap anywhere to continue</p>
      </div>
    </div>
  );
}
