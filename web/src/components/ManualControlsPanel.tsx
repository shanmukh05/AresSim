/**
 * Closable on-screen Manual Controls panel with camera-aware D-pad and labeled actions.
 * Dispatches player commands through `useAresStore`; legality stays owned by the Python engine.
 */

import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  BatteryCharging,
  Construction,
  Drill,
  Gamepad2,
  PackageOpen,
  Radar,
  Wrench,
  X,
} from "lucide-react";
import { useAresStore } from "../state/useAresStore";
import { movementDeltaForArrow, type MovementArrow } from "../lib/cameraControls";
import type { ActionType } from "../types/sim";

export function ManualControlsPanel({ onClose }: { onClose: () => void }) {
  const snapshot = useAresStore((state) => state.snapshot);
  const cameraView = useAresStore((state) => state.cameraView);
  const roverCameraYaw = useAresStore((state) => state.roverCameraYaw);
  const moveRover = useAresStore((state) => state.moveRover);
  const dispatchAction = useAresStore((state) => state.dispatchAction);
  const selectedTool = useAresStore((state) => state.selectedTool);
  const backendBusy = useAresStore((state) => state.backendBusy);

  if (!snapshot) return null;

  const rover = snapshot.rovers[0];
  const cell = rover ? snapshot.terrain[rover.y]?.[rover.x] : null;
  const disabled = snapshot.gameStatus === "game_over" || backendBusy;
  const hasCargo = rover ? rover.cargoIce + rover.cargoOre + rover.cargoSamples > 0 : false;

  const triggerMove = (arrow: MovementArrow) => {
    const delta = movementDeltaForArrow(arrow, cameraView, roverCameraYaw);
    if (delta) void moveRover(delta[0], delta[1]);
  };

  const quickActions: Array<{
    id: ActionType;
    label: string;
    shortcut: string;
    icon: React.ReactNode;
    highlighted: boolean;
  }> = [
    {
      id: "scan",
      label: "Scan",
      shortcut: "E",
      icon: <Radar size={13} />,
      highlighted: cell?.terrain === "rock" && !cell.scanned,
    },
    {
      id: "extract",
      label: "Extract",
      shortcut: "X",
      icon: <Drill size={13} />,
      highlighted: cell?.terrain === "ice" && !cell.extracted,
    },
    {
      id: "build",
      label: "Build",
      shortcut: "B",
      icon: <Construction size={13} />,
      highlighted: cell?.terrain === "build_pad" && snapshot.objectiveStats.habitatBuildProgress < 100,
    },
    {
      id: "service",
      label: "Service",
      shortcut: "R",
      icon: <Wrench size={13} />,
      highlighted: snapshot.buildPadState.serviceNeeded,
    },
    {
      id: "unload",
      label: "Unload",
      shortcut: "U",
      icon: <PackageOpen size={13} />,
      highlighted: cell?.terrain === "build_pad" && hasCargo,
    },
  ];

  const terrainLabel = cell ? cell.terrain.replace("_", " ") : "unknown";

  return (
    <section
      aria-label="Manual controls panel"
      data-testid="manual-controls-panel"
      className="w-[272px] max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl border border-cyan-300/20 bg-[#0a0d12]/90 shadow-[0_18px_46px_rgba(0,0,0,0.52)] backdrop-blur-xl"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onClose();
        }
      }}
    >
      <div className="flex items-center justify-between border-b border-white/8 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <Gamepad2 size={14} className="shrink-0 text-cyan-300" />
          <div className="min-w-0">
            <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-cyan-100">Manual Controls</div>
            {rover ? (
              <div className="truncate text-[9px] capitalize text-stone-400">
                {terrainLabel} · ({rover.x}, {rover.y})
              </div>
            ) : null}
          </div>
        </div>
        <button
          aria-label="Close manual controls"
          className="grid h-6 w-6 place-items-center rounded-md text-stone-400 transition hover:bg-white/10 hover:text-white"
          onClick={onClose}
          type="button"
        >
          <X size={13} />
        </button>
      </div>

      <div className="grid grid-cols-[auto_1fr] items-center gap-3 p-3">
        <div className="grid grid-cols-3 gap-1" aria-label="Directional movement pad">
          <span />
          <PadButton
            label={cameraView === "rover" ? "Move forward (W / ↑)" : "Move north (W / ↑)"}
            hint="W"
            disabled={disabled}
            onClick={() => triggerMove("ArrowUp")}
          >
            <ArrowUp size={15} />
          </PadButton>
          <span />

          <PadButton
            label={cameraView === "rover" ? "Move left (A / ←)" : "Move west (A / ←)"}
            hint="A"
            disabled={disabled}
            onClick={() => triggerMove("ArrowLeft")}
          >
            <ArrowLeft size={15} />
          </PadButton>
          <PadButton
            label="Wait and charge (Space)"
            hint="␣"
            active={selectedTool === "wait"}
            disabled={disabled}
            onClick={() => void dispatchAction({ type: "wait" })}
          >
            <BatteryCharging size={14} />
          </PadButton>
          <PadButton
            label={cameraView === "rover" ? "Move right (D / →)" : "Move east (D / →)"}
            hint="D"
            disabled={disabled}
            onClick={() => triggerMove("ArrowRight")}
          >
            <ArrowRight size={15} />
          </PadButton>

          <span />
          <PadButton
            label={cameraView === "rover" ? "Move backward (S / ↓)" : "Move south (S / ↓)"}
            hint="S"
            disabled={disabled}
            onClick={() => triggerMove("ArrowDown")}
          >
            <ArrowDown size={15} />
          </PadButton>
          <span />
        </div>

        <div className="grid gap-1" aria-label="Rover action shortcuts">
          {quickActions.map((action) => (
            <button
              key={action.id}
              aria-label={`${action.label} action (${action.shortcut})`}
              disabled={disabled}
              onClick={() => void dispatchAction({ type: action.id })}
              type="button"
              className={`flex h-7 items-center justify-between rounded-md border px-2 text-[10px] font-semibold transition disabled:pointer-events-none disabled:opacity-35 ${
                action.highlighted
                  ? "border-cyan-300/50 bg-cyan-400/15 text-cyan-100 shadow-[0_0_12px_rgba(57,213,232,0.18)] hover:bg-cyan-400/25"
                  : selectedTool === action.id
                    ? "border-white/25 bg-white/10 text-white"
                    : "border-white/[0.07] bg-white/[0.025] text-stone-300 hover:border-white/20 hover:bg-white/[0.07] hover:text-white"
              }`}
            >
              <span className="flex items-center gap-1.5 text-[11px] font-medium">
                <span className={action.highlighted ? "text-cyan-200" : "text-stone-400"}>{action.icon}</span>
                {action.label}
              </span>
              <kbd className="rounded border border-white/10 bg-black/40 px-1 py-px font-mono text-[8px] text-stone-400">
                {action.shortcut}
              </kbd>
            </button>
          ))}
        </div>
      </div>

      <div className="border-t border-white/[0.06] bg-black/25 px-3 py-1.5 text-[9px] text-stone-500">
        Keys: <span className="text-stone-300">WASD / Arrows</span> move · <span className="text-stone-300">Space</span> charge
      </div>
    </section>
  );
}

function PadButton({
  label,
  hint,
  children,
  onClick,
  disabled,
  active = false,
}: {
  label: string;
  hint: string;
  children: React.ReactNode;
  onClick: () => void;
  disabled: boolean;
  active?: boolean;
}) {
  return (
    <button
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      type="button"
      className={`relative grid h-10 w-10 place-items-center rounded-lg border transition active:scale-95 disabled:pointer-events-none disabled:opacity-30 ${
        active
          ? "border-cyan-300/45 bg-cyan-400/15 text-cyan-100"
          : "border-white/10 bg-white/[0.04] text-stone-200 hover:border-cyan-300/40 hover:bg-cyan-400/12 hover:text-white"
      }`}
    >
      {children}
      <span className="pointer-events-none absolute bottom-0.5 right-1 font-mono text-[7px] text-stone-500">{hint}</span>
    </button>
  );
}
