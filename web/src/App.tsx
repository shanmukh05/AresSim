/**
 * Root AresSim shell: chrome, drawers, and keyboard movement.
 * Authoritative state lives in `useAresStore`; this file only composes the layout.
 */

import { useEffect, useState } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import { Activity, AlertTriangle, PanelRightClose, Satellite, Target } from "lucide-react";
import { TopStatusRibbon } from "./components/TopStatusRibbon";
import { MissionRail, AlertsSection } from "./components/MissionRail";
import { GameViewport } from "./components/GameViewport";
import { InspectorPanel } from "./components/InspectorPanel";
import { ActionBar } from "./components/ActionBar";
import { AgentHistoryPanel } from "./components/AgentHistoryPanel";
import { GameInfoModal } from "./components/GameInfoModal";
import { AnalyticsModal } from "./components/analytics/AnalyticsModal";
import { AlertHud, MissionHud, SelectionHud } from "./components/WorldHud";
import { useAresStore } from "./state/useAresStore";
import { useSimulationAudio } from "./audio/useSimulationAudio";
import { movementDeltaForArrow } from "./lib/cameraControls";

type Drawer = "mission" | "alerts" | "inspector" | "history" | null;

function App() {
  useSimulationAudio();
  const [infoOpen, setInfoOpen] = useState(false);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const start = useAresStore((state) => state.start);
  const snapshot = useAresStore((state) => state.snapshot);
  const backendError = useAresStore((state) => state.backendError);
  const backendBusy = useAresStore((state) => state.backendBusy);
  const retryBackend = useAresStore((state) => state.retryBackend);
  const moveRover = useAresStore((state) => state.moveRover);
  const cameraView = useAresStore((state) => state.cameraView);
  const roverCameraYaw = useAresStore((state) => state.roverCameraYaw);
  const actionWarning = useAresStore((state) => state.actionWarning);
  const clearActionWarning = useAresStore((state) => state.clearActionWarning);
  const analyticsOpen = useAresStore((state) => state.analyticsOpen);
  const setAnalyticsOpen = useAresStore((state) => state.setAnalyticsOpen);

  useEffect(() => { void start(1447); }, [start]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (infoOpen || drawer) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
      const delta = movementDeltaForArrow(event.key, cameraView, roverCameraYaw);
      if (!delta) return;
      event.preventDefault();
      moveRover(delta[0], delta[1]);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [cameraView, drawer, infoOpen, moveRover, roverCameraYaw]);

  useEffect(() => {
    if (!actionWarning || actionWarning.kind === "terminal") return;
    const timer = window.setTimeout(clearActionWarning, 4200);
    return () => window.clearTimeout(timer);
  }, [actionWarning, clearActionWarning]);

  if (!snapshot) {
    return (
      <main className="grid h-screen place-items-center bg-[#090a0d] text-stone-100">
        <div className="grid justify-items-center gap-3 text-center">
          <Satellite className="text-cyan-200" size={28} />
          <p className="text-sm">{backendError ?? "Booting AresSim..."}</p>
          {backendError ? <button className="rounded-md border border-cyan-200/30 px-3 py-1.5 text-xs text-cyan-100 disabled:opacity-40" disabled={backendBusy} onClick={() => void retryBackend()} type="button">Retry backend</button> : null}
        </div>
      </main>
    );
  }

  return (
    <Tooltip.Provider delayDuration={250}>
      <main className="flex h-dvh flex-col overflow-hidden bg-[#090a0d] text-stone-100">
        <TopStatusRibbon onOpenGuide={() => setInfoOpen(true)} onOpenHistory={() => setDrawer("history")} onOpenMission={() => setDrawer("mission")} />

        <section className="relative min-h-0 flex-1 overflow-hidden">
          <GameViewport />

          <div className="pointer-events-none absolute left-4 top-4 z-20 max-md:left-3 max-md:top-3">
            <div className="pointer-events-auto"><MissionHud onOpen={() => setDrawer("mission")} /></div>
            <div className="pointer-events-auto"><AlertHud onOpen={() => setDrawer("alerts")} /></div>
          </div>

          <div className="pointer-events-none absolute bottom-4 right-4 z-30 max-md:hidden">
            <div className="pointer-events-auto"><SelectionHud onOpen={() => setDrawer("inspector")} /></div>
          </div>

          {actionWarning ? (
            <div className={`absolute left-1/2 top-4 z-40 w-[min(520px,calc(100%-2rem))] -translate-x-1/2 rounded-xl border px-4 py-3 text-sm shadow-2xl backdrop-blur-xl ${warningClass(actionWarning.kind)}`} role="alert">
              <div className="text-[9px] font-bold uppercase tracking-[0.18em] opacity-70">{actionWarning.title}</div>
              <div className="mt-1 text-xs">{actionWarning.message}</div>
            </div>
          ) : null}

          {drawer ? <ContextDrawer drawer={drawer} onClose={() => setDrawer(null)} /> : null}
        </section>

        <ActionBar />

        <GameInfoModal open={infoOpen} onOpenChange={setInfoOpen} />
        <AnalyticsModal open={analyticsOpen} onOpenChange={setAnalyticsOpen} />
      </main>
    </Tooltip.Provider>
  );
}

function ContextDrawer({ drawer, onClose }: { drawer: Exclude<Drawer, null>; onClose: () => void }) {
  const config = {
    mission: { title: "Mission & rewards", subtitle: "Objectives, progress, and reward terms", icon: <Target size={17} /> },
    alerts: { title: "Colony systems", subtitle: "Alerts, power, and life support", icon: <AlertTriangle size={17} /> },
    inspector: { title: "Context inspector", subtitle: "Selected terrain, agent, or structure", icon: <Satellite size={17} /> },
    history: { title: "Run timeline", subtitle: "Actions, rewards, and environment events", icon: <Activity size={17} /> },
  }[drawer];
  return (
    <div className="absolute inset-0 z-50 flex justify-end bg-black/20" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <aside className="flex h-full w-[min(390px,92vw)] flex-col border-l border-white/10 bg-[#100e12]/94 shadow-[-28px_0_70px_rgba(0,0,0,0.48)] backdrop-blur-2xl" data-testid={`${drawer}-drawer`}>
        <div className="flex h-16 shrink-0 items-center gap-3 border-b border-white/8 px-4">
          <span className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-200/15 bg-cyan-300/8 text-cyan-100">{config.icon}</span>
          <div className="min-w-0 flex-1"><h2 className="text-sm font-semibold text-white">{config.title}</h2><p className="text-[10px] text-stone-500">{config.subtitle}</p></div>
          <button aria-label="Close drawer" className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 text-stone-400 hover:border-cyan-200/30 hover:text-white" onClick={onClose} type="button"><PanelRightClose size={16} /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-hidden p-3">
          {drawer === "mission" ? <MissionRail /> : null}
          {drawer === "alerts" ? <AlertsSection /> : null}
          {drawer === "inspector" ? <InspectorPanel /> : null}
          {drawer === "history" ? <AgentHistoryPanel /> : null}
        </div>
      </aside>
    </div>
  );
}

function warningClass(kind: NonNullable<ReturnType<typeof useAresStore.getState>["actionWarning"]>["kind"]) {
  if (kind === "terminal") return "border-rose-300/45 bg-rose-950/90 text-rose-50";
  if (kind === "terrain") return "border-yellow-300/35 bg-yellow-950/82 text-yellow-50";
  if (kind === "system") return "border-cyan-300/30 bg-cyan-950/82 text-cyan-50";
  if (kind === "progress") return "border-emerald-300/30 bg-emerald-950/82 text-emerald-50";
  return "border-orange-300/35 bg-orange-950/82 text-orange-50";
}

export default App;
