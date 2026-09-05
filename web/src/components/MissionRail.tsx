/** Side rail for mission progress, selection, and secondary panels. */

import { AlertTriangle, Ban, BatteryCharging, CheckCircle2, Crosshair, Drill, Droplets, Footprints, Gauge, LineChart, PackageOpen, Sun, Target, Trophy, Wrench } from "lucide-react";
import type { ReactNode } from "react";
import { useAresStore } from "../state/useAresStore";
import { formatFixed, formatSignedFixed } from "../lib/format";
import { estimateWaitRecharge } from "../lib/power";

export function MissionRail() {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const setAnalyticsOpen = useAresStore((state) => state.setAnalyticsOpen);
  const stats = snapshot.objectiveStats;
  const iceProgress = percent(stats.iceSitesExtracted, stats.iceSitesTotal);
  const scanProgress = percent(stats.terrainScanned, stats.rockSitesTotal);

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden rounded border border-stone-800 bg-stone-950/35 p-3" data-testid="reward-objectives">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
        <Target size={14} />
        Reward Objectives
      </div>
      <h1 className="text-base font-semibold leading-tight text-white">{snapshot.mission.title}</h1>
      <p className="mt-1.5 text-xs leading-4 text-stone-400">{snapshot.mission.objective}</p>
      <div className="mt-3 grid min-h-0 gap-1.5 overflow-y-auto pr-1">
        <TotalRewardCard reward={stats.rewardTotals.total} stepCount={snapshot.step} onOpen={() => setAnalyticsOpen(true)} />
        <ObjectiveCard icon={<Drill size={14} />} label="Ice Collected" progress={iceProgress} progressLabel="Ice objective progress" reward={stats.rewardTotals.iceCollected} value={`${stats.iceSitesExtracted} / ${stats.iceSitesTotal} sites | ${stats.iceCollected.toFixed(0)} units`} />
        <ObjectiveCard icon={<PackageOpen size={14} />} label="Payload Delivered" reward={stats.rewardTotals.delivered} value={`${stats.iceDelivered.toFixed(0)} kg ice · ${stats.samplesDelivered.toFixed(0)} kg samples · ${stats.unloadCount} ${stats.unloadCount === 1 ? "unload" : "unloads"}`} />
        <ObjectiveCard icon={<Crosshair size={14} />} label="Terrain Scanned" progress={scanProgress} progressLabel="Terrain objective progress" reward={stats.rewardTotals.terrainScanned} value={`${stats.terrainScanned} / ${stats.rockSitesTotal} rock sites`} />
        <ObjectiveCard
          icon={<CheckCircle2 size={14} />}
          label="Habitat Built"
          progress={stats.habitatBuildProgress}
          reward={stats.rewardTotals.habitatBuilt}
          value={`${formatFixed(stats.habitatBuildProgress, "%")} (${stats.habitatBuildCount}/10)`}
        />
        <ObjectiveCard icon={<Wrench size={14} />} label="Serviced" reward={stats.rewardTotals.serviced} value={`${stats.serviceCount} times`} />
        <ObjectiveCard icon={<Footprints size={14} />} label="Traversal Reward" reward={stats.rewardTotals.traversal} value="Low shaping reward" />
        <ObjectiveCard icon={<Ban size={14} />} label="Blocked Penalty" reward={stats.rewardTotals.blockedPenalty} value="Invalid action total" />
      </div>
    </section>
  );
}

export function AlertsSection() {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const selectTarget = useAresStore((state) => state.selectTarget);
  const criticalAlert = snapshot.gameStatus === "game_over" || snapshot.rules.some((rule) => rule.status === "failed") || snapshot.resources.battery <= 18 || snapshot.resources.livability <= 16;
  const alertCount = snapshot.mission.alerts.length + (snapshot.buildPadState.serviceNeeded ? 1 : 0);

  return (
    <section className="min-h-0 flex-1">
      <button className={`flex h-full min-h-0 w-full flex-col overflow-hidden rounded border p-3 text-left transition hover:border-amber-300/40 ${criticalAlert ? "border-rose-400/60 bg-rose-950/25 shadow-[0_0_22px_rgba(244,63,94,0.18)]" : "border-stone-800 bg-stone-950/35"}`} onClick={() => selectTarget({ kind: "status", id: "alerts" })} type="button">
        <div className="mb-3 flex shrink-0 items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-stone-500">
          <span className={`flex items-center gap-2 ${criticalAlert ? "text-rose-100" : ""}`}>
            <AlertTriangle size={14} />
            Alerts
          </span>
          <span className={`rounded border px-1.5 py-0.5 text-[10px] ${criticalAlert ? "border-rose-300/40 bg-rose-400/15 text-rose-100" : "border-amber-300/20 bg-amber-400/10 text-amber-100"}`}>{alertCount}</span>
        </div>
        <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {snapshot.buildPadState.serviceNeeded ? (
            <AlertRow tone="critical" text="Build pad service required" />
          ) : null}
          {snapshot.mission.alerts.length === 0 && !snapshot.buildPadState.serviceNeeded ? <AlertRow tone="clear" text="No active alerts" /> : null}
          <div className="space-y-1.5">
            {snapshot.mission.alerts.map((alert) => (
              <AlertRow key={alert} tone={alert.includes("failed") || alert.includes("ended") || alert.includes("0.00") ? "critical" : "normal"} text={alert} />
            ))}
          </div>
          <div className="grid gap-1.5 border-t border-stone-800/80 pt-2 text-[10px] text-stone-400">
            <AlertMetric icon={<Sun size={12} />} label="Solar" value={formatFixed(snapshot.resources.powerGenerated, " kW")} />
            <AlertMetric icon={<BatteryCharging size={12} />} label="Load" value={formatFixed(snapshot.resources.powerConsumed, " kW")} />
            <AlertMetric icon={<BatteryCharging size={12} />} label="Wait charge" value={formatSignedFixed(estimateWaitRecharge(snapshot), "%")} />
            <AlertMetric icon={<Droplets size={12} />} label="Water chain" value={snapshot.resources.water > 110 ? "Online" : "Building"} />
            <AlertMetric icon={<Gauge size={12} />} label="Pressure" value={snapshot.resources.oxygen > 150 ? "Stable" : "Incomplete"} />
          </div>
        </div>
      </button>
    </section>
  );
}

function percent(value: number, total: number) {
  return total > 0 ? Math.min(100, (value / total) * 100) : 0;
}

function AlertMetric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <span className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1.5 rounded border border-stone-800 bg-black/20 px-2 py-1">
      <span className="text-cyan-100">{icon}</span>
      <span className="truncate text-stone-500">{label}</span>
      <span className="font-semibold text-stone-100">{value}</span>
    </span>
  );
}

function AlertRow({ text, tone }: { text: string; tone: "normal" | "critical" | "clear" }) {
  const style =
    tone === "critical"
      ? "border-rose-400/35 bg-rose-500/10 text-rose-100"
      : tone === "clear"
        ? "border-emerald-300/20 bg-emerald-400/10 text-emerald-100"
        : "border-stone-800 bg-black/20 text-stone-200";
  return <div className={`rounded border px-2 py-1.5 text-xs leading-4 ${style}`}>{text}</div>;
}

function TotalRewardCard({ reward, stepCount, onOpen }: { reward: number; stepCount: number; onOpen: () => void }) {
  const positive = reward >= 0;
  return (
    <button
      type="button"
      onClick={onOpen}
      data-testid="total-reward-card"
      aria-label="Open analytics from reward summary"
      className="grid w-full gap-1 rounded-lg border border-cyan-300/40 bg-gradient-to-br from-cyan-950/80 via-stone-950 to-emerald-950/70 p-3 text-left shadow-[0_0_24px_rgba(34,211,238,0.18)] transition hover:border-cyan-200/80 hover:shadow-[0_0_32px_rgba(34,211,238,0.32)]"
    >
      <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-100/80">
        <span className="flex items-center gap-1.5"><Trophy size={12} /> Total Reward</span>
        <span className="flex items-center gap-1 text-cyan-200/80">{stepCount} steps <LineChart size={12} /></span>
      </div>
      <div className={`text-xl font-bold tabular-nums ${positive ? "text-emerald-200" : "text-rose-200"}`}>
        {formatSignedFixed(reward)}
      </div>
      <div className="text-[10px] text-stone-400">Click to open analytics</div>
    </button>
  );
}

function ObjectiveCard({ icon, label, value, reward, progress, progressLabel }: { icon: ReactNode; label: string; value: string; reward: number; progress?: number; progressLabel?: string }) {
  const clamped = progress == null ? null : Math.max(0, Math.min(100, progress));

  return (
    <div className="rounded border border-stone-800 bg-black/20 px-2 py-1.5">
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded border border-cyan-200/20 bg-cyan-300/10 text-cyan-100">{icon}</span>
        <div className="min-w-0">
          <div className="truncate text-xs font-semibold text-stone-100">{label}</div>
          <div className="text-[10px] text-stone-500">{value}</div>
        </div>
        <div className={`text-right text-[10px] font-semibold ${reward < 0 ? "text-rose-200" : "text-emerald-200"}`}>{formatSignedFixed(reward)}</div>
      </div>
      {clamped != null ? (
        <div aria-label={progressLabel ?? `${label} progress`} aria-valuemax={100} aria-valuemin={0} aria-valuenow={Number(clamped.toFixed(1))} className="mt-1.5 h-1.5 overflow-hidden rounded bg-stone-900" role="progressbar">
          <div className="h-full rounded bg-gradient-to-r from-orange-400 to-cyan-300" style={{ width: `${clamped}%` }} />
        </div>
      ) : null}
    </div>
  );
}
