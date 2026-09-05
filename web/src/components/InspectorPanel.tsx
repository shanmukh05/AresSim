/**
 * Selected-tile inspector. Reads `selectedTarget` plus the snapshot; selection
 * is UI-only until an action is dispatched.
 */

import { Activity, Box, ChevronRight, Crosshair, MapPinned, MousePointer2, Satellite } from "lucide-react";
import { Metric } from "./ui";
import { useAresStore } from "../state/useAresStore";
import type { SimSnapshot } from "../types/sim";
import type { ReactNode } from "react";
import { formatFixed, formatSignedFixed } from "../lib/format";
import { estimateWaitRecharge, getPowerMargin } from "../lib/power";
import { payloadRemainingKg, payloadUsedKg } from "../lib/payload";

type MetricTone = "default" | "good" | "warn" | "bad";
type DetailMetric = { label: string; value: string; tone?: MetricTone };
type Detail = { icon: ReactNode; title: string; description: string; metrics: DetailMetric[] };

export function InspectorPanel() {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const selected = useAresStore((state) => state.selectedTarget);
  const hovered = useAresStore((state) => state.hoveredTarget);
  const target = selected ?? hovered;
  const detail = getDetail(snapshot, target);
  const pinned = !!selected;

  return (
    <section className="min-h-0 overflow-y-auto" data-empty={!target} data-testid="inspector">
      <div className="rounded-xl border border-white/8 bg-gradient-to-b from-cyan-300/[0.055] to-transparent p-3">
        <div className="flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-200/15 bg-cyan-300/8 text-cyan-100">{target ? detail.icon : <Satellite size={16} />}</span>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 text-[8px] font-semibold text-stone-500"><span className={`h-1.5 w-1.5 rounded-full ${pinned ? "bg-cyan-300" : target ? "bg-amber-300" : "bg-stone-600"}`} />{pinned ? "Pinned target" : target ? "Live hover" : "Inspector ready"}</div>
            <h3 className="mt-0.5 truncate text-sm font-semibold text-white">{detail.title}</h3>
          </div>
          {target ? <Crosshair size={14} className="text-stone-600" /> : null}
        </div>
        <p className="mt-3 text-[11px] leading-5 text-stone-400">{detail.description}</p>
      </div>

      {target ? (
        <div className="mt-3 grid grid-cols-2 gap-2">
          {detail.metrics.map((metric) => <div key={metric.label} className="rounded-lg border border-white/[0.07] bg-black/20 p-2.5"><Metric label={metric.label} value={metric.value} tone={metric.tone} /></div>)}
        </div>
      ) : (
        <div className="mt-3 overflow-hidden rounded-xl border border-dashed border-white/10 bg-black/15">
          <InspectorHint icon={<MousePointer2 size={14} />} label="Hover" text="Preview terrain or an entity" />
          <InspectorHint icon={<MapPinned size={14} />} label="Select" text="Pin details while you operate" />
          <InspectorHint icon={<ChevronRight size={14} />} label="Layers" text="Use icons in Environment controls" />
        </div>
      )}
    </section>
  );
}

function InspectorHint({ icon, label, text }: { icon: ReactNode; label: string; text: string }) {
  return <div className="flex items-center gap-3 border-b border-white/[0.06] px-3 py-2.5 last:border-b-0"><span className="text-cyan-100/60">{icon}</span><span className="w-12 text-[10px] font-medium text-stone-200">{label}</span><span className="text-[10px] text-stone-500">{text}</span></div>;
}

function emptyDetail(): Detail {
  return {
    icon: <MapPinned size={16} />,
    title: "Nothing pinned",
    description: "Point at the world for a preview, or select a cell, rover, structure, or timeline event to keep its telemetry here.",
    metrics: [],
  };
}

function cellDetail(snapshot: SimSnapshot, target: { x: number; y: number }): Detail {
  const cell = snapshot.terrain[target.y]?.[target.x];
  return {
    icon: <MapPinned size={16} />,
    title: `Cell ${target.x}, ${target.y}`,
    description: "Terrain channels come from the authoritative engine snapshot.",
    metrics: [
      { label: "Terrain", value: cell.terrain },
      { label: "Ice", value: `${Math.round(cell.ice * 100)}%`, tone: cell.ice > 0.55 ? "good" : "default" },
      { label: "Ore", value: `${Math.round(cell.ore * 100)}%` },
      { label: "Dust", value: `${Math.round(cell.dust * 100)}%`, tone: cell.dust > 0.6 ? "warn" : "default" },
      { label: "Scanned", value: cell.scanned ? "Yes" : "No", tone: cell.scanned ? "good" : "default" },
      { label: "Roughness", value: `${Math.round(cell.roughness * 100)}%` },
    ],
  };
}

function roverDetail(snapshot: SimSnapshot, id: string): Detail {
  const rover = snapshot.rovers.find((item) => item.id === id) ?? snapshot.rovers[0];
  return {
    icon: <Activity size={16} />,
    title: rover.name,
    description: rover.currentTask,
    metrics: [
      { label: "Battery", value: `${Math.round(rover.battery)}%`, tone: rover.battery > 35 ? "good" : "bad" },
      { label: "Health", value: `${Math.round(rover.health)}%`, tone: "good" },
      { label: "Ice cargo", value: `${rover.cargoIce} kg` },
      { label: "Samples", value: `${rover.cargoSamples} kg` },
      { label: "Ore cargo", value: `${rover.cargoOre} kg` },
      { label: "Payload", value: `${payloadUsedKg(rover)} / ${rover.cargoCapacityKg} kg`, tone: payloadRemainingKg(rover) <= 0 ? "warn" : "default" },
    ],
  };
}

function historyDetail(snapshot: SimSnapshot, id: string): Detail {
  const row = snapshot.history.find((item) => item.id === id);
  return {
    icon: <Activity size={16} />,
    title: row ? `Step ${row.step}: ${row.action}` : "History entry",
    description: row?.result ?? "No history entry selected.",
    metrics: row
      ? [
          { label: "Actor", value: row.actor },
          { label: "Reward", value: row.reward.toFixed(2), tone: row.reward >= 0 ? "good" : "bad" },
          { label: "Events", value: `${row.events.length}` },
          { label: "Target", value: row.target ? `${row.target.x}, ${row.target.y}` : "None" },
        ]
      : [],
  };
}

function structureDetail(snapshot: SimSnapshot, id: string): Detail | null {
  const structure = snapshot.structures.find((item) => item.id === id);
  if (!structure) return null;
  return {
    icon: <Box size={16} />,
    title: structure.name,
    description: structure.status,
    metrics: [
      { label: "Type", value: structure.type },
      { label: "Health", value: `${structure.health}%`, tone: structure.health > 50 ? "good" : "bad" },
      { label: "Power", value: structure.powered ? "Connected" : "Offline", tone: structure.powered ? "good" : "warn" },
      { label: "Location", value: `${structure.x}, ${structure.y}` },
    ],
  };
}

function statusDetail(snapshot: SimSnapshot): Detail {
  const margin = getPowerMargin(snapshot);
  const waitCharge = estimateWaitRecharge(snapshot);
  return {
    icon: <Activity size={16} />,
    title: "Status detail",
    description: "Pinned status panel from the command ribbon.",
    metrics: [
      { label: "Generated", value: formatFixed(snapshot.resources.powerGenerated, " kW") },
      { label: "Consumed", value: formatFixed(snapshot.resources.powerConsumed, " kW") },
      { label: "Power margin", value: formatSignedFixed(margin, " kW"), tone: margin >= 0 ? "good" : "bad" },
      { label: "Wait charge", value: formatSignedFixed(waitCharge, "%"), tone: waitCharge > 0 ? "good" : "warn" },
      { label: "Alerts", value: `${snapshot.mission.alerts.length}`, tone: "warn" },
      { label: "Mode", value: snapshot.mode },
    ],
  };
}

function getDetail(snapshot: SimSnapshot | null, target: ReturnType<typeof useAresStore.getState>["selectedTarget"]): Detail {
  if (!snapshot || !target) return emptyDetail();
  if (target.kind === "cell") return cellDetail(snapshot, target);
  if (target.kind === "rover") return roverDetail(snapshot, target.id);
  if (target.kind === "history") return historyDetail(snapshot, target.id);
  if (target.kind === "structure") return structureDetail(snapshot, target.id) ?? statusDetail(snapshot);
  return statusDetail(snapshot);
}
