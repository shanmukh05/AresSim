/**
 * Compact header meters for power, battery, payload, habitat, and livability.
 * Values come from the latest snapshot; this component does not compute rules.
 */

import { Activity, Archive, BarChart3, Battery, BookOpen, Clock3, CloudSun, HeartPulse, History, House, Target, Zap } from "lucide-react";
import type { ReactNode } from "react";
import { useAresStore } from "../state/useAresStore";
import { formatFixed, formatSignedFixed } from "../lib/format";
import { payloadUsedKg } from "../lib/payload";

export function TopStatusRibbon({ onOpenGuide, onOpenHistory, onOpenMission }: { onOpenGuide: () => void; onOpenHistory: () => void; onOpenMission: () => void }) {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const setAnalyticsOpen = useAresStore((state) => state.setAnalyticsOpen);
  const powerMargin = snapshot.resources.powerGenerated - snapshot.resources.powerConsumed;
  const roverHealth = snapshot.rovers[0]?.health ?? 0;
  const rover = snapshot.rovers[0];
  const payload = rover ? payloadUsedKg(rover) : 0;
  const habitatProgress = snapshot.objectiveStats.habitatBuildProgress;
  const statusClass = snapshot.gameStatus === "game_over" ? "text-rose-200" : snapshot.gameStatus === "paused" ? "text-amber-200" : "text-emerald-200";

  return (
    <header className="relative z-50 flex h-[58px] shrink-0 items-center gap-3 border-b border-orange-100/10 bg-gradient-to-r from-[#120b09]/96 via-[#0b0c0e]/96 to-[#071013]/96 px-3 shadow-[0_14px_38px_rgba(0,0,0,0.42)] backdrop-blur-xl sm:px-4">
      <div className="flex min-w-0 items-center gap-2.5">
        <img src="/logo.png" alt="" width={36} height={36} className="h-9 w-9 shrink-0 rounded-lg border border-orange-300/25 object-cover shadow-[0_0_24px_rgba(249,115,22,0.16)]" />
        <div className="hidden leading-none sm:block">
          <div className="text-xs font-black tracking-[0.2em] text-white">ARESIM</div>
          <div className="mt-1.5 text-[8px] font-semibold uppercase tracking-[0.18em] text-orange-100/40">Habitat One</div>
        </div>
      </div>

      <div className="hidden h-7 border-l border-white/10 sm:block" />

      <div className="flex min-w-0 items-center gap-2">
        <div className={`flex items-center gap-1.5 rounded-full border border-white/8 bg-white/[0.035] px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.14em] ${statusClass}`}>
          <span className="h-1.5 w-1.5 rounded-full bg-current shadow-[0_0_9px_currentColor]" />{snapshot.gameStatus.replace("_", " ")}
        </div>
        <div className="hidden items-center gap-1.5 text-[10px] text-stone-400 md:flex"><Clock3 size={13} className="text-orange-200/60" /><span className="font-bold text-stone-100">SOL {snapshot.sol.toString().padStart(3, "0")}</span><span>{snapshot.localTime}</span></div>
        <div className="hidden items-center gap-1.5 text-[10px] text-stone-500 xl:flex"><CloudSun size={13} />{snapshot.weather}</div>
      </div>

      <div className="flex-1" />

      <div className="hidden items-center gap-1.5 lg:flex" aria-label="Grouped mission telemetry">
        <TelemetryGroup label="Rover" testId="header-rover-group">
          <Telemetry icon={<Battery size={13} />} label="Battery" value={formatFixed(snapshot.resources.battery, "%")} tone={snapshot.resources.battery > 35 ? "good" : "bad"} />
          <Telemetry icon={<Activity size={13} />} label="Health" value={formatFixed(roverHealth, "%")} tone={roverHealth > 35 ? "good" : "bad"} />
          <StorageTelemetry capacity={rover?.cargoCapacityKg ?? 12} ice={rover?.cargoIce ?? 0} ore={rover?.cargoOre ?? 0} samples={rover?.cargoSamples ?? 0} used={payload} />
        </TelemetryGroup>
        <TelemetryGroup label="Build pad" testId="header-build-pad-group">
          <Telemetry icon={<Zap size={13} />} label="Power" value={formatSignedFixed(powerMargin, " kW")} tone={powerMargin >= 0 ? "good" : "bad"} />
          <Telemetry icon={<House size={13} />} label="Habitat" value={formatFixed(habitatProgress, "%")} tone={habitatProgress >= 100 ? "good" : "neutral"} />
          <Telemetry icon={<HeartPulse size={13} />} label="Livability" value={formatFixed(snapshot.resources.livability, "%")} tone={snapshot.resources.livability > 30 ? "good" : "bad"} />
        </TelemetryGroup>
      </div>

      <div className="flex items-center gap-1 rounded-lg border border-white/8 bg-black/20 p-1">
        <HeaderButton label="Open mission" onClick={onOpenMission}><Target size={15} /></HeaderButton>
        <HeaderButton label="Open agent history" onClick={onOpenHistory}><History size={15} /></HeaderButton>
        <HeaderButton label="Open analytics" onClick={() => setAnalyticsOpen(true)} testId="analytics-button"><BarChart3 size={15} /></HeaderButton>
        <HeaderButton label="Open game guide" onClick={onOpenGuide}><BookOpen size={15} /></HeaderButton>
      </div>
    </header>
  );
}

function TelemetryGroup({ label, testId, children }: { label: string; testId: string; children: ReactNode }) {
  return (
    <section className="flex h-10 items-center overflow-hidden rounded-lg border border-white/8 bg-black/20" aria-label={`${label} telemetry`} data-testid={testId}>
      <span className="flex h-full shrink-0 items-center border-r border-white/8 px-2 text-[7px] font-black uppercase tracking-[0.16em] text-stone-500">{label}</span>
      {children}
    </section>
  );
}

function Telemetry({ icon, label, value, tone }: { icon: ReactNode; label: string; value: string; tone: "good" | "bad" | "neutral" }) {
  const iconTone = tone === "good" ? "text-emerald-200/70" : tone === "bad" ? "text-rose-200/80" : "text-cyan-200/65";
  const valueTone = tone === "bad" ? "text-rose-200" : "text-stone-100";
  return (
    <div className="flex h-full min-w-[82px] flex-col justify-center border-r border-white/8 px-2 last:border-r-0 xl:min-w-[92px]" data-testid={`header-${label.toLowerCase()}`}>
      <span className="flex items-center gap-1.5 leading-none">
        <span className={`shrink-0 ${iconTone}`}>{icon}</span>
        <span className="whitespace-nowrap text-[8px] font-bold uppercase tracking-[0.08em] text-stone-500">{label}</span>
      </span>
      <span className={`mt-1 whitespace-nowrap pl-[19px] text-[10px] font-bold leading-none tabular-nums ${valueTone}`}>{value}</span>
    </div>
  );
}

function StorageTelemetry({ used, capacity, ice, samples, ore }: { used: number; capacity: number; ice: number; samples: number; ore: number }) {
  const full = used >= capacity;
  return (
    <div
      aria-label={`Rover storage ${used} of ${capacity} kilograms; ${ice} kilograms ice, ${samples} kilograms samples, ${ore} kilograms ore`}
      aria-valuemax={capacity}
      aria-valuemin={0}
      aria-valuenow={used}
      className="flex h-full min-w-[96px] flex-col justify-center px-2 xl:min-w-[106px]"
      data-testid="header-storage"
      role="meter"
    >
      <span className="flex items-center gap-1.5 leading-none">
        <span className={`shrink-0 ${full ? "text-amber-200" : "text-cyan-200/70"}`}><Archive size={13} /></span>
        <span className="whitespace-nowrap text-[8px] font-bold uppercase tracking-[0.08em] text-stone-500">Storage</span>
      </span>
      <span className={`mt-1 whitespace-nowrap pl-[19px] text-[10px] font-bold leading-none tabular-nums ${full ? "text-amber-100" : "text-stone-100"}`}>{used}/{capacity} kg</span>
    </div>
  );
}

function HeaderButton({ label, onClick, children, testId }: { label: string; onClick: () => void; children: ReactNode; testId?: string }) {
  return <button aria-label={label} className="grid h-8 w-8 shrink-0 place-items-center rounded-md text-stone-400 transition hover:bg-cyan-300/10 hover:text-cyan-100" data-testid={testId} onClick={onClick} type="button">{children}</button>;
}
