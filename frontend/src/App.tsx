import { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { Splash } from "./components/Splash";
import { Dashboard } from "./pages/Dashboard";
import { Logs } from "./pages/Logs";
import { Telemetry } from "./pages/Telemetry";
import { Vehicle } from "./pages/Vehicle";
import { GcsProvider } from "./state/GcsContext";
import { ThemeProvider } from "./state/ThemeContext";

type SplashPhase = "visible" | "exiting" | "hidden";

const SPLASH_FADE_MS = 500;
const SPLASH_MIN_HOLD_MS = 600;

export default function App() {
  const [splashPhase, setSplashPhase] = useState<SplashPhase>("visible");
  const [canDismiss, setCanDismiss] = useState(false);

  useEffect(() => {
    const holdTimer = setTimeout(() => setCanDismiss(true), SPLASH_MIN_HOLD_MS);
    return () => clearTimeout(holdTimer);
  }, []);

  useEffect(() => {
    if (splashPhase !== "exiting") return;
    const fadeTimer = setTimeout(() => setSplashPhase("hidden"), SPLASH_FADE_MS);
    return () => clearTimeout(fadeTimer);
  }, [splashPhase]);

  const dismissSplash = () => {
    if (!canDismiss || splashPhase !== "visible") return;
    setSplashPhase("exiting");
  };

  return (
    <ThemeProvider>
      <GcsProvider>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/telemetry" element={<Telemetry />} />
          <Route path="/vehicle" element={<Vehicle />} />
          <Route path="/logs" element={<Logs />} />
        </Routes>
        {splashPhase !== "hidden" && (
          <Splash exiting={splashPhase === "exiting"} onDismiss={dismissSplash} />
        )}
      </GcsProvider>
    </ThemeProvider>
  );
}
