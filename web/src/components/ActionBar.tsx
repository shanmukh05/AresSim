/**
 * Footer Action Bar for Manual, Algorithm, and Replay.
 * Buttons send commands through the store; the Python engine decides legality.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  BatteryCharging,
  Check,
  Construction,
  Cpu,
  Drill,
  FileJson,
  Gamepad2,
  Pause,
  PackageOpen,
  Play,
  Radar,
  RotateCcw,
  Route,
  Save,
  Shuffle,
  StepForward,
  Upload,
  Wrench,
  X,
} from "lucide-react";
import { useAresStore } from "../state/useAresStore";
import type { ActionType, AlgorithmId, LoadedReplay, RunMode, SimAction } from "../types/sim";

const MODE_STYLE: Record<RunMode, { label: string; icon: ReactNode; accent: string; active: string }> = {
  manual: { label: "Manual", icon: <Gamepad2 size={13} />, accent: "#39d5e8", active: "border-cyan-500/40 bg-cyan-500/10 text-cyan-200 shadow-[0_0_12px_rgba(57,213,232,0.15)]" },
  algorithm: { label: "Algorithm", icon: <Cpu size={13} />, accent: "#a78bfa", active: "border-violet-500/40 bg-violet-500/10 text-violet-200 shadow-[0_0_12px_rgba(167,139,250,0.15)]" },
  load: { label: "Replay", icon: <FileJson size={13} />, accent: "#f6b94a", active: "border-amber-500/40 bg-amber-500/10 text-amber-200 shadow-[0_0_12px_rgba(246,185,74,0.15)]" },
};

const ACTIONS: Array<{ id: ActionType; label: string; icon: ReactNode }> = [
  { id: "move", label: "Move", icon: <Route size={16} /> },
  { id: "scan", label: "Scan", icon: <Radar size={16} /> },
  { id: "extract", label: "Extract", icon: <Drill size={16} /> },
  { id: "build", label: "Build", icon: <Construction size={16} /> },
  { id: "service", label: "Service", icon: <Wrench size={16} /> },
  { id: "unload", label: "Unload payload", icon: <PackageOpen size={16} /> },
  { id: "wait", label: "Wait", icon: <BatteryCharging size={16} /> },
];

const ALGORITHMS: Array<{ value: AlgorithmId; label: string }> = [
  { value: "random", label: "Random" },
  { value: "random_valid", label: "Random (valid)" },
  { value: "wait", label: "Wait" },
  { value: "scripted", label: "Scripted" },
  { value: "masked_ppo", label: "Masked PPO" },
];

const PLAY_INTERVAL_MS = 700;

export function ActionBar() {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const runMode = useAresStore((state) => state.runMode);
  const selectedTool = useAresStore((state) => state.selectedTool);
  const loadedReplay = useAresStore((state) => state.loadedReplay);
  const paused = useAresStore((state) => state.paused);
  const speed = useAresStore((state) => state.speed);
  const setRunMode = useAresStore((state) => state.setRunMode);
  const setSpeed = useAresStore((state) => state.setSpeed);
  const dispatchAction = useAresStore((state) => state.dispatchAction);
  const step = useAresStore((state) => state.step);
  const pauseResume = useAresStore((state) => state.pauseResume);
  const resetCurrentRun = useAresStore((state) => state.resetCurrentRun);
  const resetReplay = useAresStore((state) => state.resetReplay);
  const jumpToReplayStep = useAresStore((state) => state.jumpToReplayStep);
  const randomize = useAresStore((state) => state.randomize);
  const start = useAresStore((state) => state.start);
  const loadGameplay = useAresStore((state) => state.loadGameplay);
  const saveRun = useAresStore((state) => state.saveRun);
  const backendBusy = useAresStore((state) => state.backendBusy);

  const [seed, setSeed] = useState(String(snapshot.seed));
  const [playing, setPlaying] = useState(false);
  const [saveOpen, setSaveOpen] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);
  const terminal = snapshot.gameStatus === "game_over";
  const replayEnded = !!loadedReplay && loadedReplay.cursor >= loadedReplay.totalSteps;
  const modeStyle = MODE_STYLE[runMode];
  const controlsDisabled = terminal || backendBusy;

  useEffect(() => setSeed(String(snapshot.seed)), [snapshot.seed]);
  useEffect(() => setPlaying(false), [runMode, snapshot.sessionId]);
  useEffect(() => { if (terminal || replayEnded) setPlaying(false); }, [replayEnded, terminal]);
  useEffect(() => {
    if (!playing || paused || terminal || runMode === "manual" || (runMode === "load" && !loadedReplay)) return;
    let cancelled = false;
    let timer: number | undefined;
    const schedule = () => {
      timer = window.setTimeout(async () => {
        await useAresStore.getState().step();
        if (!cancelled) schedule();
      }, PLAY_INTERVAL_MS / Math.max(1, speed));
    };
    schedule();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [loadedReplay, paused, playing, runMode, speed, terminal]);

  const applySeed = () => {
    const parsed = Number.parseInt(seed, 10);
    if (Number.isFinite(parsed)) void start(Math.abs(parsed) % 100000);
  };
  const changeMode = async (mode: RunMode) => {
    if (paused) await pauseResume();
    setPlaying(false);
    setRunMode(mode);
  };
  const togglePlay = async () => {
    if (!playing) {
      if (paused) await pauseResume();
      setPlaying(true);
    } else {
      await pauseResume();
    }
  };
  const uploadReplay = async (file?: File) => {
    if (!file) return;
    await loadGameplay(await readFile(file), file.name);
    if (uploadRef.current) uploadRef.current.value = "";
  };

  return (
    <footer className="relative flex h-[60px] shrink-0 items-center justify-center border-t border-white/10 bg-[#07090b] px-4 shadow-[0_-12px_34px_rgba(0,0,0,0.34),inset_0_1px_rgba(255,255,255,0.025)]" data-mode={runMode} data-testid="action-bar">
      <span className="absolute inset-x-0 top-0 h-px opacity-90" style={{ background: `linear-gradient(90deg, transparent 22%, ${modeStyle.accent} 50%, transparent 78%)` }} />
      <div className="grid h-12 w-full max-w-[1180px] grid-cols-[1fr_auto_1fr] items-center gap-4" aria-label="Gameplay controls" data-testid="action-bar-controls">
        <section className="relative flex h-12 w-fit justify-self-start items-center gap-2 rounded-lg border border-white/[0.07] bg-[#090b11]/80 px-3 pt-1 shadow-[0_4px_20px_rgba(0,0,0,0.4),inset_0_1px_rgba(255,255,255,0.03)]" aria-label="Mode and source controls" data-testid="action-bar-left">
          <ZoneLabel color={modeStyle.accent}>Mode</ZoneLabel>
          <ModeSwitch active={runMode} onChange={(mode) => void changeMode(mode)} disabled={backendBusy} />
          {runMode === "algorithm" ? <><Divider /><AlgorithmSetup disabled={backendBusy} /></> : null}
          {runMode === "load" ? <><Divider /><ReplaySource loadedReplay={loadedReplay} uploadRef={uploadRef} onUpload={uploadReplay} disabled={backendBusy} /></> : null}
        </section>

        <section className="relative flex h-12 w-fit justify-self-center items-center justify-center gap-2 rounded-lg border border-white/[0.1] bg-[#0c0f16]/95 px-4 pt-1 shadow-[0_4px_20px_rgba(0,0,0,0.5),inset_0_1px_rgba(255,255,255,0.04)]" aria-label="Active game controls" data-testid="action-bar-center" style={{ borderColor: `${modeStyle.accent}30`, boxShadow: `0_4px_20px_rgba(0,0,0,0.5), inset_0_1px_rgba(255,255,255,0.04), 0_0_16px_${modeStyle.accent}08` }}>
          <ZoneLabel color={modeStyle.accent}>{runMode === "manual" ? "Rover commands" : runMode === "algorithm" ? "Run controls" : "Replay controls"}</ZoneLabel>
          {runMode === "manual" ? <><div className="flex gap-1.5" aria-label="Environment actions">{ACTIONS.map((action) => <CommandButton key={action.id} label={action.label} active={selectedTool === action.id} accent={modeStyle.accent} disabled={controlsDisabled} onClick={() => void dispatchAction({ type: action.id })}>{action.icon}</CommandButton>)}</div><Divider /><RunUtilities disabled={backendBusy} onReset={() => void resetCurrentRun()} onSave={() => setSaveOpen(true)} /></> : null}
          {runMode === "algorithm" ? <><Transport playing={playing} paused={paused} disabled={controlsDisabled} stepDisabled={controlsDisabled || (playing && !paused)} onToggle={() => void togglePlay()} onStep={() => void step()} /><SpeedControl speed={speed} onChange={setSpeed} accent={modeStyle.accent} disabled={backendBusy} /><Divider /><RunUtilities disabled={backendBusy} onReset={() => void resetCurrentRun()} onSave={() => setSaveOpen(true)} /></> : null}
          {runMode === "load" ? <ReplayControls loadedReplay={loadedReplay} playing={playing} paused={paused} ended={replayEnded} busy={backendBusy} speed={speed} onToggle={() => void togglePlay()} onStep={() => void step()} onSpeed={setSpeed} onRepeat={() => { setPlaying(false); void resetReplay(); }} onJump={(target) => void jumpToReplayStep(target)} accent={modeStyle.accent} /> : null}
        </section>

        <section className="relative flex h-12 w-fit justify-self-end items-center justify-end rounded-lg border border-white/[0.07] bg-[#090b11]/80 px-3 pt-1 shadow-[0_4px_20px_rgba(0,0,0,0.4),inset_0_1px_rgba(255,255,255,0.03)]" aria-label="Environment controls" data-testid="action-bar-right">
          <ZoneLabel color={runMode === "load" && loadedReplay ? "#78716c" : modeStyle.accent}>World setup</ZoneLabel>
          <EnvironmentTools
            seed={runMode === "load" && loadedReplay ? String(loadedReplay.seed) : seed}
            setSeed={setSeed}
            applySeed={applySeed}
            randomize={randomize}
            disabled={backendBusy || (runMode === "load" && loadedReplay !== null)}
          />
        </section>
      </div>
      <SaveDialog open={saveOpen} sessionId={snapshot.sessionId} onClose={() => setSaveOpen(false)} onSave={(name) => { void saveRun(name); setSaveOpen(false); }} />
    </footer>
  );
}

function ModeSwitch({ active, onChange, disabled = false }: { active: RunMode; onChange: (mode: RunMode) => void; disabled?: boolean }) {
  const widths: Record<RunMode, string> = { manual: "w-[76px]", algorithm: "w-[94px]", load: "w-[76px]" };
  return (
    <nav className="flex gap-1" aria-label="Simulation mode">
      {(["manual", "algorithm", "load"] as RunMode[]).map((mode) => {
        const style = MODE_STYLE[mode];
        return (
          <TooltipButton
            key={mode}
            label={`${style.label} mode`}
            disabled={disabled}
            ariaPressed={active === mode}
            className={`relative flex h-9 ${widths[mode]} items-center justify-center gap-1.5 rounded-lg border px-1.5 font-sans text-[11px] font-bold tracking-wide transition max-[1099px]:w-9 ${active === mode
                ? style.active
                : "border-white/[0.04] bg-white/[0.01] text-stone-400 hover:border-white/12 hover:bg-white/[0.04] hover:text-stone-200"
              }`}
            onClick={() => onChange(mode)}
          >
            {style.icon}
            <span className="max-[1099px]:hidden">{style.label}</span>
          </TooltipButton>
        );
      })}
    </nav>
  );
}

function EnvironmentTools({ seed, setSeed, applySeed, randomize, disabled = false }: { seed: string; setSeed: (seed: string) => void; applySeed: () => void; randomize: () => void; disabled?: boolean }) {
  return (
    <div className="flex items-center gap-1.5" aria-label="Environment setup">
      <input
        aria-label="World seed"
        className="h-9 w-16 rounded-lg border border-white/[0.08] bg-[#0c0f16]/90 px-2 font-mono text-xs text-stone-100 outline-none focus:border-white/20 focus:ring-1 focus:ring-white/10 transition disabled:opacity-40"
        inputMode="numeric"
        value={seed}
        disabled={disabled}
        onChange={(event) => setSeed(event.target.value)}
        onKeyDown={(event) => { if (event.key === "Enter" && !disabled) applySeed(); }}
      />
      <CommandButton label="Apply seed" disabled={disabled} onClick={applySeed}><Check size={15} /></CommandButton>
      <CommandButton label="Randomize world" disabled={disabled} onClick={randomize}><Shuffle size={15} /></CommandButton>
    </div>
  );
}

function Transport({ playing, paused, disabled, stepDisabled, onToggle, onStep }: { playing: boolean; paused: boolean; disabled: boolean; stepDisabled: boolean; onToggle: () => void; onStep: () => void }) {
  return (
    <div className="flex gap-1.5">
      <CommandButton label={!playing ? "Start run" : paused ? "Resume run" : "Pause run"} disabled={disabled} onClick={onToggle}>
        {playing && !paused ? <Pause size={15} /> : <Play size={15} />}
      </CommandButton>
      <CommandButton label="Step simulation" disabled={stepDisabled} onClick={onStep}>
        <StepForward size={15} />
      </CommandButton>
    </div>
  );
}

function AlgorithmSetup({ disabled = false }: { disabled?: boolean }) {
  const algorithmId = useAresStore((state) => state.algorithmId);
  const checkpointPath = useAresStore((state) => state.checkpointPath);
  const rllibAvailable = useAresStore((state) => state.rllibAvailable);
  const policyCapabilitiesLoaded = useAresStore((state) => state.policyCapabilitiesLoaded);
  const setAlgorithmId = useAresStore((state) => state.setAlgorithmId);
  const setCheckpointPath = useAresStore((state) => state.setCheckpointPath);
  const attachCurrentPolicy = useAresStore((state) => state.attachCurrentPolicy);
  const refreshPolicyCapabilities = useAresStore((state) => state.refreshPolicyCapabilities);

  useEffect(() => {
    void refreshPolicyCapabilities();
  }, [refreshPolicyCapabilities]);

  return (
    <div className="flex items-center gap-1.5">
      <label className="flex h-9 items-center gap-2 rounded-lg border border-white/[0.08] bg-[#0c0f16]/90 px-2.5 text-xs text-stone-300 focus-within:border-violet-500/40 transition">
        <Cpu size={13} className="text-violet-300" />
        <select aria-label="Algorithm policy" disabled={disabled} className="max-w-[108px] bg-transparent font-semibold text-white outline-none cursor-pointer text-[11px] disabled:opacity-30" value={algorithmId} onChange={(event) => setAlgorithmId(event.target.value as AlgorithmId)}>
          {ALGORITHMS.map((option) => {
            const blocked = option.value === "masked_ppo" && policyCapabilitiesLoaded && !rllibAvailable;
            return (
              <option key={option.value} value={option.value} disabled={blocked} className="bg-[#0b0e14] text-white">
                {option.label}{blocked ? " (rllib required)" : ""}
              </option>
            );
          })}
        </select>
      </label>
      {algorithmId === "masked_ppo" ? (
        <label className="flex h-9 min-w-[180px] items-center gap-2 rounded-lg border border-white/[0.08] bg-[#0c0f16]/90 px-2.5 text-xs text-stone-300 focus-within:border-violet-500/40 transition">
          <input
            aria-label="Checkpoint path"
            disabled={disabled}
            className="w-full bg-transparent font-mono text-[10px] text-white outline-none placeholder:text-stone-500 disabled:opacity-30"
            placeholder="/absolute/path/to/checkpoint.json"
            value={checkpointPath}
            onChange={(event) => setCheckpointPath(event.target.value)}
            onBlur={() => void attachCurrentPolicy()}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void attachCurrentPolicy();
              }
            }}
          />
        </label>
      ) : null}
      <CommandButton label="Apply checkpoint path" disabled={disabled || algorithmId !== "masked_ppo"} onClick={() => void attachCurrentPolicy()}>
        <Check size={15} />
      </CommandButton>
    </div>
  );
}

function ReplaySource({ loadedReplay, uploadRef, onUpload, disabled = false }: { loadedReplay: LoadedReplay | null; uploadRef: React.RefObject<HTMLInputElement | null>; onUpload: (file?: File) => void; disabled?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <input ref={uploadRef} aria-label="Upload trajectory JSON" className="sr-only" type="file" accept="application/json,.json" onChange={(event) => void onUpload(event.target.files?.[0])} />
      <CommandButton label={loadedReplay ? "Replace replay" : "Choose replay"} disabled={disabled} onClick={() => uploadRef.current?.click()}><Upload size={15} /></CommandButton>
      <span className="max-w-24 truncate font-mono text-[10px] text-stone-400" data-testid="loaded-replay-meta">{loadedReplay?.fileName ?? "No replay"}</span>
    </div>
  );
}

function ReplayControls({ loadedReplay, playing, paused, ended, busy, speed, onToggle, onStep, onSpeed, onRepeat, onJump, accent }: { loadedReplay: LoadedReplay | null; playing: boolean; paused: boolean; ended: boolean; busy: boolean; speed: number; onToggle: () => void; onStep: () => void; onSpeed: (speed: number) => void; onRepeat: () => void; onJump: (step: number) => void; accent: string }) {
  const loaded = !!loadedReplay;
  return (
    <div className="flex items-center gap-2" aria-label="Replay controls">
      <div className="flex gap-1.5">
        <CommandButton label={!playing ? "Play replay" : paused ? "Resume replay" : "Pause replay"} disabled={!loaded || ended || busy} onClick={onToggle}>
          {playing && !paused ? <Pause size={15} /> : <Play size={15} />}
        </CommandButton>
        <CommandButton label="Step replay" disabled={!loaded || ended || busy} onClick={onStep}>
          <StepForward size={15} />
        </CommandButton>
      </div>
      <CommandButton label="Repeat replay" disabled={!loaded || busy} onClick={onRepeat}><RotateCcw size={15} /></CommandButton>
      <SpeedControl speed={speed} onChange={onSpeed} disabled={!loaded || busy} accent={accent} />
      <span className="w-11 text-right font-mono text-[10px] text-stone-300" data-testid="replay-cursor">{loadedReplay?.cursor ?? 0}/{loadedReplay?.totalSteps ?? 0}</span>
      <input aria-label="Replay step" data-testid="step-scrubber" className="h-1 w-32 cursor-pointer rounded-lg bg-white/10 accent-current transition disabled:opacity-20" style={{ accentColor: accent }} type="range" min={0} max={Math.max(1, loadedReplay?.totalSteps ?? 0)} value={Math.min(loadedReplay?.cursor ?? 0, Math.max(1, loadedReplay?.totalSteps ?? 0))} disabled={!loaded || busy} onChange={(event) => onJump(Number(event.target.value))} />
    </div>
  );
}

function SpeedControl({ speed, onChange, accent, disabled = false }: { speed: number; onChange: (speed: number) => void; accent: string; disabled?: boolean }) {
  return (
    <label className={`flex h-9 items-center gap-1.5 rounded-lg border border-white/[0.08] bg-[#0c0f16]/90 px-2.5 ${disabled ? "opacity-25" : ""}`}>
      <input aria-label="Autoplay speed" className="h-1 w-12 cursor-pointer" style={{ accentColor: accent }} type="range" min={1} max={5} step={1} value={speed} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} />
      <span className="w-5 font-mono text-[10px] text-stone-300">{speed}×</span>
    </label>
  );
}

function RunUtilities({ onReset, onSave, disabled = false }: { onReset: () => void; onSave: () => void; disabled?: boolean }) {
  return (
    <div className="flex gap-1.5">
      <CommandButton label="Reset run" disabled={disabled} onClick={onReset}><RotateCcw size={15} /></CommandButton>
      <CommandButton label="Export trajectory" disabled={disabled} onClick={onSave}><Save size={15} /></CommandButton>
    </div>
  );
}

function CommandButton({ label, children, onClick, disabled = false, active = false, accent }: { label: string; children: ReactNode; onClick?: () => void; disabled?: boolean; active?: boolean; accent?: string }) {
  return (
    <TooltipButton
      label={label}
      disabled={disabled}
      className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg border transition disabled:pointer-events-none disabled:opacity-20 ${active
          ? "border-white/30 bg-white/10 text-white"
          : "border-white/[0.06] bg-white/[0.02] text-stone-400 hover:border-white/15 hover:bg-white/[0.06] hover:text-white"
        }`}
      style={active && accent ? { borderColor: accent, color: accent, boxShadow: `0 0 12px ${accent}22` } : undefined}
      onClick={onClick}
    >
      {children}
    </TooltipButton>
  );
}

function TooltipButton({ label, children, className, onClick, disabled = false, ariaPressed, style }: { label: string; children: ReactNode; className: string; onClick?: () => void; disabled?: boolean; ariaPressed?: boolean; style?: React.CSSProperties }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button aria-label={label} aria-pressed={ariaPressed} className={className} disabled={disabled} onClick={onClick} style={style} type="button">
          {children}
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="z-[80] rounded-md border border-white/10 bg-[#0b0e11] px-2 py-1 text-[10px] text-stone-200 shadow-xl" sideOffset={7}>
          {label}
          <Tooltip.Arrow className="fill-[#0b0e11]" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

function Divider() {
  return <span className="mx-1 h-8 border-l border-white/[0.08]" />;
}

function ZoneLabel({ children, color }: { children: ReactNode; color: string }) {
  return (
    <span className="pointer-events-none absolute -top-2 left-4 px-1.5 py-0.5 rounded text-[8px] font-bold uppercase tracking-[0.08em] bg-[#07090b] border border-white/[0.06] text-stone-400 flex items-center gap-1.5 shadow-sm">
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
      {children}
    </span>
  );
}

function SaveDialog({ open, sessionId, onClose, onSave }: { open: boolean; sessionId: string; onClose: () => void; onSave: (name: string) => void }) {
  const [name, setName] = useState("");
  const fallback = `aresim-${sessionId.slice(0, 6)}`;
  useEffect(() => { if (!open) setName(""); }, [open]);
  return <Dialog.Root open={open} onOpenChange={(next) => { if (!next) onClose(); }}><Dialog.Portal><Dialog.Overlay className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm" /><Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[min(400px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-white/10 bg-[#0b0e12] p-4 shadow-2xl"><div className="flex items-center justify-between"><Dialog.Title className="text-sm font-semibold text-white">Export trajectory</Dialog.Title><Dialog.Close aria-label="Close save dialog" className="text-stone-500 hover:text-white"><X size={16} /></Dialog.Close></div><Dialog.Description className="mt-1 text-[10px] text-stone-500">Export the current run as replayable trajectory JSON.</Dialog.Description><input autoFocus aria-label="Trajectory filename" className="mt-4 h-9 w-full rounded-md border border-white/10 bg-black/35 px-3 text-xs text-white outline-none" placeholder={fallback} value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") onSave(name.trim() || fallback); }} /><div className="mt-3 flex justify-end gap-2"><button className="h-8 rounded-md border border-white/10 px-3 text-[10px] text-stone-400" onClick={onClose} type="button">Cancel</button><button className="h-8 rounded-md border border-cyan-200/25 bg-cyan-300/12 px-3 text-[10px] text-cyan-100" onClick={() => onSave(name.trim() || fallback)} type="button">Export JSON</button></div></Dialog.Content></Dialog.Portal></Dialog.Root>;
}

function readFile(file: File) {
  if (typeof file.text === "function") return file.text();
  return new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result ?? "")); reader.onerror = () => reject(reader.error); reader.readAsText(file); });
}
