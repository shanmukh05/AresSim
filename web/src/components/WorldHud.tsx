/** In-world HUD overlays for mission progress and warnings. */

import { AlertTriangle, ArrowRight, Crosshair, Target } from "lucide-react";
import { useAresStore } from "../state/useAresStore";
import type { SelectionTarget, SimSnapshot } from "../types/sim";
import { formatSignedFixed } from "../lib/format";

export function MissionHud({ onOpen }: { onOpen: () => void }) {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const stats = snapshot.objectiveStats;
  const progress = Math.max(stats.habitatBuildProgress, stats.terrainScanned / Math.max(1, stats.rockSitesTotal) * 100);
  return (
    <button
      className="group w-[min(268px,calc(100vw-2rem))] overflow-hidden rounded-lg border border-white/10 bg-[#0b0d0f]/82 text-left shadow-[0_16px_38px_rgba(0,0,0,0.34)] backdrop-blur-xl transition hover:border-cyan-200/30"
      data-testid="mission-hud"
      onClick={onOpen}
      type="button"
    >
      <div className="flex items-center gap-2.5 px-2.5 py-2">
        <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-orange-200/15 bg-orange-300/8 text-orange-100/70"><Target size={13} /></span>
        <span className="min-w-0 flex-1"><span className="block text-[7px] font-bold uppercase tracking-[0.16em] text-stone-600">Current objective</span><span className="mt-0.5 block truncate text-[11px] font-semibold text-white">{snapshot.mission.title}</span></span>
        <span className="text-[10px] font-bold tabular-nums text-emerald-200">{formatSignedFixed(stats.rewardTotals.total)}</span>
        <ArrowRight size={13} className="shrink-0 text-stone-600 transition group-hover:translate-x-0.5 group-hover:text-cyan-100" />
      </div>
      <div className="h-0.5 bg-white/8"><div className="h-full bg-gradient-to-r from-orange-400 to-cyan-300" style={{ width: `${Math.min(100, progress)}%` }} /></div>
    </button>
  );
}

export function AlertHud({ onOpen }: { onOpen: () => void }) {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const count = snapshot.mission.alerts.length + (snapshot.buildPadState.serviceNeeded ? 1 : 0);
  const critical = snapshot.gameStatus === "game_over" || snapshot.resources.battery <= 18;
  const primary = snapshot.buildPadState.serviceNeeded ? "Build pad service required" : snapshot.mission.alerts[0] ?? "All habitat systems nominal";
  return (
    <button
      className={`mt-1.5 flex w-[min(268px,calc(100vw-2rem))] items-center gap-2 rounded-lg border px-2.5 py-2 text-left shadow-xl backdrop-blur-xl transition ${critical ? "border-rose-300/35 bg-rose-950/70 text-rose-100" : "border-white/10 bg-[#0b0d0f]/78 text-stone-300 hover:border-amber-200/30"}`}
      data-testid="alert-hud"
      onClick={onOpen}
      type="button"
    >
      <AlertTriangle size={13} className={critical ? "text-rose-200" : count ? "text-amber-200" : "text-emerald-200"} />
      <span className="min-w-0 flex-1 truncate text-[10px]">{primary}</span>
      <span className="rounded-full bg-white/8 px-1.5 py-0.5 text-[9px] font-bold">{count}</span>
    </button>
  );
}

export function SelectionHud({ onOpen }: { onOpen: () => void }) {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const selected = useAresStore((state) => state.selectedTarget);
  if (!selected) return null;
  const summary = selectionSummary(snapshot, selected);
  return (
    <button className="group flex min-w-52 max-w-64 items-center gap-2 rounded-lg border border-cyan-200/15 bg-[#071216]/86 px-2.5 py-2 text-left shadow-2xl backdrop-blur-xl transition hover:border-cyan-200/35" data-testid="selection-hud" onClick={onOpen} type="button">
      <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-cyan-200/15 bg-cyan-300/8 text-cyan-100"><Crosshair size={13} /></span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[10px] font-semibold text-white">{summary.title}</span>
        <span className="mt-px block truncate text-[9px] text-stone-500">{summary.detail}</span>
      </span>
      <ArrowRight size={12} className="shrink-0 text-stone-600 transition group-hover:translate-x-0.5 group-hover:text-cyan-100" />
    </button>
  );
}

function selectionSummary(snapshot: SimSnapshot, target: SelectionTarget) {
  if (target.kind === "cell") {
    const cell = snapshot.terrain[target.y]?.[target.x];
    return { title: `${cell?.terrain.replace("_", " ") ?? "Cell"} · ${target.x}, ${target.y}`, detail: cell ? `Roughness ${Math.round(cell.roughness * 100)}% · Dust ${Math.round(cell.dust * 100)}%` : "Terrain cell" };
  }
  if (target.kind === "rover") {
    const rover = snapshot.rovers.find((item) => item.id === target.id);
    return { title: rover?.name ?? "Rover", detail: rover ? `${Math.round(rover.battery)}% battery · ${rover.currentTask}` : "Agent" };
  }
  if (target.kind === "structure") {
    const structure = snapshot.structures.find((item) => item.id === target.id);
    return { title: structure?.name ?? "Structure", detail: structure ? `${structure.health}% health · ${structure.status}` : "Habitat structure" };
  }
  if (target.kind === "history") {
    const row = snapshot.history.find((item) => item.id === target.id);
    return { title: row ? `Step ${row.step} · ${row.action}` : "History step", detail: row?.result ?? "Replay event" };
  }
  return { title: "Colony status", detail: snapshot.statusReason };
}
