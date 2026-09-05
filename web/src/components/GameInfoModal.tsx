/** Modal guide that explains Phase 1 mission goals and controls. */

import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import type { ReactNode } from "react";
import { X } from "lucide-react";

export function GameInfoModal({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 grid max-h-[86vh] w-[min(960px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded border border-stone-700 bg-[#121015] text-stone-100 shadow-2xl">
          <div className="flex items-start justify-between gap-4 border-b border-stone-800 px-5 py-4">
            <div>
              <Dialog.Title className="text-xl font-semibold text-white">AresSim Game Guide</Dialog.Title>
              <Dialog.Description className="mt-1 text-sm text-stone-400">How the environment, rewards, and survival rules work. Use this to understand why the simulation behaves the way it does.</Dialog.Description>
            </div>
            <Dialog.Close className="grid h-9 w-9 place-items-center rounded border border-stone-700 bg-stone-950 text-stone-300 hover:border-stone-500" aria-label="Close game guide">
              <X size={18} />
            </Dialog.Close>
          </div>

          <Tabs.Root className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)]" defaultValue="play">
            <Tabs.List className="grid grid-cols-5 border-b border-stone-800 bg-stone-950/45 p-2 text-xs max-sm:grid-cols-2">
              {["play", "terrain", "actions", "rewards", "rules"].map((tab) => (
                <Tabs.Trigger key={tab} className="rounded px-3 py-2 capitalize text-stone-400 data-[state=active]:bg-stone-800 data-[state=active]:text-white" value={tab}>
                  {tab}
                </Tabs.Trigger>
              ))}
            </Tabs.List>

            <div className="min-h-0 overflow-y-auto p-5">
              <Tabs.Content className="grid gap-4 md:grid-cols-2" value="play">
                <GuideSection title="How To Play">
                  Select a map cell, then click an action. The command executes immediately. Explore until rover battery, rover health, or habitat livability fails. There is no fixed mission checklist - useful actions produce reward shaping rather than task completion gates.
                </GuideSection>
                <GuideSection title="Controls">
                  Arrow keys move the rover one cell in Manual mode. The compact footer switches between Manual, Algorithm, and Replay. Manual exposes direct action icons, Algorithm provides policy transport and speed controls, and Replay provides file and timeline controls. Starting Algorithm mode runs automatic steps until paused or terminal.
                </GuideSection>
                <GuideSection title="Camera And Replay">
                  The default 3D Survey uses an angled orthographic camera so terrain height, structures, and routes remain readable while the world rotates through 0-359°. Top view is a separate north-up camera with the environment sides aligned to the window. Fit keeps the full environment centered; in a zoomed 3D or Top view, click the navigation map to pan to any region. Rover movement resumes follow. Mouse wheel and ± control scale. Rover POV uses a level first-person camera with rover-relative arrows: Up always moves forward, Down backward, and Left/Right move to the rover's current sides. After a direction change, the complete arrow frame rotates with the rover. Movement and turns remain smoothly interpolated without vertical bob.
                  The empty space behind the world also reflects local Sol time: an illustrated Sun and corona cross the background during daylight, the horizon warms at dawn and dusk, and a restrained Mars-sky guide appears at night. Phobos is the larger, faster irregular moon; Deimos is smaller and smoother. Earth, Venus, Jupiter, and Saturn use compact recognizable illustrations for orientation, not live astronomical positions. This ambient clock never overlays or recolors gameplay terrain.
                </GuideSection>
                <GuideSection title="Rover Visibility">
                  Policies see a fixed 8x8 square, not the complete 32x32 world. The rover is local cell [3,3], covering offsets -3 through +4; world-edge slots are unknown padding. Use the flashlight icon beside Sound and Cell Boundaries to preview this exact crop in 3D Survey, Top, or Rover POV. The preview is visual only and does not alter actions or rewards.
                </GuideSection>
                <GuideSection title="Determinism And Seeds">
                  Every seed reproduces the same terrain channels, landing build pad, rover start, starter infrastructure, weather, colony resources, and initial events. Different seeds usually produce visibly different maps while still satisfying landing-zone validity. Randomize creates a new deterministic world; Set starts the entered seed.
                </GuideSection>
                <GuideSection title="Save And Load">
                  <RuleLine label="Trajectory" text="Exports a portable aresim.trajectory.episode.v1 JSON file with policy metadata and a complete replay projection: initial snapshot, per-step deltas, checkpoints, and final snapshot." />
                  <RuleLine label="Load" text="Restores seed, snapshots, and replay cursor from a save file. The grid, validation, and rewards are not re-simulated - the uploaded file is the replay authority." />
                  <RuleLine label="Checkpoint" text="Jump directly to an automatic anchor (initial, every 10 steps, significant event, final) without replaying earlier deltas." />
                </GuideSection>
                <GuideSection title="Analytics">
                  The Run Analytics modal (icon in the top ribbon) plots reward, resource, behavior, progress, and environment series for both live and loaded runs from the same underlying data, so saved plays can be inspected exactly like live ones.
                </GuideSection>
              </Tabs.Content>

              <Tabs.Content className="grid gap-4 md:grid-cols-2" value="terrain">
                <GuideSection title="Environment Generation">
                  Terrain is generated as coherent regions: connected ridge bands shaped by a sinusoidal axis, basin-like craters with rough rims, clustered ice deposits biased toward lower terrain, rocky ore regions near ridges, dune bands, and regolith plains between features. The build pad search retries with deterministic seed offsets and falls back to a sanitized safe patch.
                </GuideSection>
                <GuideSection title="Terrain Legend">
                  <Legend color="bg-[#8c4a2f]" label="Regolith" text="Smooth rover traversal, normal battery drain." />
                  <Legend color="bg-[#3a302b]" label="Rock / Ore" text="Dark chunky outcrops with mineral flecks. Scan is valid only here." />
                  <Legend color="bg-[#9fc7d1]" label="Ice" text="Traversable deposit; extraction reduces the cell's ice value." />
                  <Legend color="bg-[#342620]" label="Crater" text="Blocked terrain. Rover movement is not allowed." />
                  <Legend color="bg-[#b56436]" label="Dune" text="Traversable warning terrain with extra battery drain." />
                  <Legend color="bg-[#6f5546]" label="Ridge" text="Connected raised bands with high battery drain." />
                  <Legend color="bg-[#9a6a48]" label="Build Pad" text="Only a small landing-zone area. Build and explicit payload Unload are restricted to this pad; Service works on or near it." />
                </GuideSection>
                <GuideSection title="Cell Channels">
                  Each grid cell carries continuous channels plus a discrete type: <code>height</code>, <code>roughness</code>, <code>ice</code>, <code>ore</code>, <code>dust</code>, <code>scanned</code>, <code>extracted</code>. The grid is always the source of truth; visual smoothing never changes validity or rewards.
                </GuideSection>
                <GuideSection title="Build Pad Rules">
                  The build pad is a contiguous 5x5 flat, low-roughness, non-crater, non-ridge, non-ice, non-ore area. Build is valid only on build-pad cells. Service is valid on or near the pad. The raised deck combines a rover docking lane, utility gantry, solar service rack, habitat assembly zone, ice tank farm, and sample-storage bay as one operating depot.
                </GuideSection>
                <GuideSection title="Build Pad Visuals">
                  Cyan power glow means normal. A persistent red deck, solar rack, perimeter, alarm ring, beam, and beacon means Service is required. Habitat framing grows by 10% per successful Build until the dome is complete. Delivered ice fills the cyan tank gauge and delivered samples populate the violet storage crates. Unload animates separate cyan and violet transfer routes from the rover dock to those destinations.
                </GuideSection>
                <GuideSection title="Build Pad Service Thresholds">
                  The service warning latches only after dust exceeds <code>0.78</code>, infrastructure health falls below <code>62%</code>, or the power margin falls below <code>-8</code> while dust is above <code>0.55</code>. Moderate wear and transient power deficits remain operational. Service repairs the pad, resets dust to <code>0.08</code>, and clears the warning.
                </GuideSection>
                <GuideSection title="Markers">
                  Cyan rings show scanned rock/ore cells. After extraction, the tile becomes normal regolith and an amber X with a downward pointer marks the extraction site. These markers are replay/analysis visuals only and do not change the underlying grid rules.
                </GuideSection>
              </Tabs.Content>

              <Tabs.Content className="grid gap-4 md:grid-cols-2" value="actions">
                <GuideSection title="Action Space">
                  Phase 1 keeps the visible action space small for RL friendliness: <RuleLine label="Move" text="Traverse one target cell. Arrows are world-cardinal in 3D/Top and rover-relative in Rover POV." />
                  <RuleLine label="Scan" text="Inspect an unscanned rock/ore block." />
                  <RuleLine label="Extract" text="Pull ice from valid ice deposits." />
                  <RuleLine label="Build" text="Add 10% habitat progress on build-pad cells." />
                  <RuleLine label="Service" text="Clear build-pad service state, repair infrastructure, reduce dust." />
                  <RuleLine label="Unload" text="Transfer every carried resource to base storage while standing on the build pad." />
                  <RuleLine label="Wait" text="Run one charging cycle." />
                </GuideSection>
                <GuideSection title="Validity Rules">
                  Move is blocked on craters. Move is allowed on ridges and dunes with warnings and extra battery cost. Scan works once per rock/ore block and requires 0.5 kg free payload. Extract requires at least 25% ice and 2 kg free payload. Rock terrain is scan-only in Phase 1 - ore signal is retained for future mechanics. Build requires the landing-zone build pad and is blocked once habitat is complete. Service requires the rover to be on or near the build pad. Unload requires the rover to stand on the build pad with non-empty cargo.
                </GuideSection>
                <GuideSection title="Invalid Action Handling">
                  Invalid actions do not mutate rover position, terrain resources, payload, or colony resources (except explicit penalty mechanics). They append an invalid history entry, emit a blocked warning, and apply zero or negative reward. Examples: moving into a crater, scanning an already-scanned cell, collecting without enough capacity, extracting from a non-ice cell, building off-pad, servicing away from the pad, or unloading off-pad.
                </GuideSection>
                <GuideSection title="Cargo And Conversion">
                  The shared payload bay holds 12 kg: <code>used = ice + ore + samples</code>. Scan collects a 0.5 kg sample and Extract collects 2 kg ice; neither partially collects when the full result cannot fit. Cargo stays aboard when the rover enters the pad. Use Unload explicitly to send samples to the storage vault and ice to the tank farm. Delivered ice conversion: <code>waterGain = cargoIce * 1.8</code>, <code>oxygenGain = cargoIce * 0.8</code>, <code>livabilityGain = cargoIce * 0.08</code>.
                </GuideSection>
                <GuideSection title="Action Loads">
                  Each action contributes a power load on top of the base infrastructure consumption: <RuleLine label="Scan" text="+0.35 kW load." />
                  <RuleLine label="Extract" text="+1.4 kW load." />
                  <RuleLine label="Build" text="+1.8 kW load." />
                  <RuleLine label="Service" text="+0.8 kW load." />
                  <RuleLine label="Unload" text="+0.25 kW load." />
                  <RuleLine label="Wait" text="+0.0 kW load." />
                  <RuleLine label="Move" text="+0.15 kW load." />
                </GuideSection>
              </Tabs.Content>

              <Tabs.Content className="grid gap-4 md:grid-cols-2" value="rewards">
                <GuideSection className="md:col-span-2" title="How Rewards Are Computed">
                  Each timestep returns a structured <code>rewardTerms</code> object. The display reward (<code>reward = sum(rewardTerms)</code>) is added to <code>rewardTotals.total</code>. Terms are kept separate from the total so reward shaping can be debugged per category.
                </GuideSection>
                <GuideSection className="md:col-span-2" title="Per-Action Reward Terms">
                  <div className="grid gap-2 md:grid-cols-2">
                    <Equation label="Move" formula="R = traversal − (battery cost × 0.08) + safety" note="Traversal: +0.01 on regolith or +0.002 on the build pad. Safety: −0.08 on ridge/dune; otherwise +0.01." />
                    <Equation label="Scan" formula="R = 1.10 information + 0.25 exploration − (battery cost × 0.08)" note="Only the first valid scan of a rock/ore cell receives the information and exploration terms." />
                    <Equation label="Extract" formula="R = (ice × 0.42) + (ore × 0.20) − (battery cost × 0.08)" note="The resource term scales with material actually removed from the cell." />
                    <Equation label="Build" formula="R = 1.80 infrastructure + 0.75 livability − (battery cost × 0.08)" note="Awarded only when the build action advances a valid build-pad project." />
                    <Equation label="Service" formula="R = service + 0.35 efficiency − (battery cost × 0.08)" note="Service is +1.20 when required and +0.35 otherwise." />
                    <Equation label="Unload" formula="R = 0.50·ice + 0.40·ore + 0.30·samples + 0.40 logistics − (battery cost × 0.08)" note="Mass values are the resources actually transferred from rover to base." />
                    <Equation label="Wait" formula="R = recovery − (battery cost × 0.04)" note="Recovery is +0.25 with positive power margin; otherwise −0.15." />
                    <Equation label="Invalid" formula="R = −2.00 invalid − 0.10 time" note="Invalid commands do not receive any positive action terms." />
                  </div>
                </GuideSection>
                <GuideSection title="Reward Categories">
                  Terms may include: <code>information</code>, <code>exploration</code>, <code>resource</code>, <code>delivery</code>, <code>logistics</code>, <code>infrastructure</code>, <code>livability</code>, <code>service</code>, <code>efficiency</code>, <code>traversal</code>, <code>energy</code>, <code>safety</code>, <code>invalid</code>, <code>recovery</code>. The Run Analytics modal charts each category separately.
                </GuideSection>
                <GuideSection title="Cumulative Totals">
                  <code>rewardTotals</code> accumulates per objective: <RuleLine label="iceCollected" text="sum of extract-action rewards." />
                  <RuleLine label="terrainScanned" text="sum of scan-action rewards." />
                  <RuleLine label="delivered" text="sum of unload-action delivery and logistics rewards." />
                  <RuleLine label="habitatBuilt" text="sum of build-action rewards." />
                  <RuleLine label="serviced" text="sum of service-action rewards." />
                  <RuleLine label="traversal" text="sum of move-action movement shaping." />
                  <RuleLine label="blockedPenalty" text="sum of invalid-action penalties (also in total)." />
                  <RuleLine label="total" text="sum of every term across every action." />
                </GuideSection>
                <GuideSection title="Reward Principles">
                  Reward useful exploration and information gain. Reward resource collection only when resources exist. Reward build/service only when they improve infrastructure, livability, or power. Give ordinary movement only a tiny shaping reward to prevent move-farming. Penalize invalid actions. Penalize unsafe terrain through battery and safety terms. Keep Wait from being a free reward source.
                </GuideSection>
                <GuideSection title="Analytics Plots">
                  The Analytics modal Rewards tab shows total reward per step, cumulative reward, per-category lines, cumulative stacked area, signed bars per step, and stacked category bars. Use the legend to toggle series on and off.
                </GuideSection>
              </Tabs.Content>

              <Tabs.Content className="grid gap-4 md:grid-cols-2" value="rules">
                <GuideSection title="Terminal Failure Conditions">
                  <RuleLine label="Battery 0%" text="Rover battery depleted. Exploration ends." />
                  <RuleLine label="Health 0%" text="Rover can no longer operate." />
                  <RuleLine label="Livability 0%" text="Habitat livability collapsed." />
                </GuideSection>
                <GuideSection className="md:col-span-2" title="Battery Drain Formula">
                  <Equation label="Battery drain" formula="D = base(action) × e^(0.32 × total stress) + power deficit − recharge" note="The exponential makes several moderate hazards together more costly than one isolated hazard." />
                  <Equation label="Total stress" formula="S = terrain + 0.45·roughness + 0.22·cell dust + weather + action + cargo" note="Every input is evaluated at the rover's current cell and current simulation step." />
                  <Equation label="Cargo stress" formula="cargo = 0.012 × (ice + ore + sample payload kg)" note="Every carried kilogram increases action drain through the same shared mass model." />
                </GuideSection>
                <GuideSection title="Action Base Drain">
                  Move: 0.32. Scan: 0.24. Extract: 0.82. Build: 1.05. Service: 0.52. Unload: 0.18. Wait: 0.0. Difficult terrain multiplies drain via the stress exponent.
                </GuideSection>
                <GuideSection title="Terrain Stress">
                  Build Pad: 0.05. Regolith: 0.08. Rock: 0.20. Ice: 0.16. Dune: 0.34. Ridge: 0.62. Crater: blocked before drain. Roughness, dust, and cargo add further stress.
                </GuideSection>
                <GuideSection title="Power Model">
                  <Equation label="Generation" formula="Pgen = panels × 42 × average health × weather factor × dust factor" note="Dust factor is clamped from 0.30 to 1.00." />
                  <Equation label="Consumption" formula="Puse = 6.5 + 4.2·habitats + 1.1·storage + 1.6·chargers + action load" note="Power margin is generation minus consumption." />
                  <RuleLine label="Weather factors" text="Clear 1.18 · Dusty 0.72 · Dust Front 0.48 · Cold Night 0.22 · Severe Storm 0.36." />
                </GuideSection>
                <GuideSection title="Power Effects">
                  <Equation label="Deficit penalty" formula="min(1.60, |power margin| × 0.055)" note="Applied only when the power margin is negative." />
                  <Equation label="Wait recharge" formula="min(18.0, power margin × 0.85)" note="Applied only with a positive margin." />
                  <Equation label="Passive pad charge" formula="min(4.0, power margin × 0.25)" note="The build-pad charger is intentionally slower than Wait." />
                </GuideSection>
                <GuideSection title="Life-Support Drain">
                  <Equation label="Water / step" formula="0.07 + 0.0018·livability + 0.035·dust intensity" note="Water is a colony resource and drains every step." />
                  <Equation label="Oxygen / step" formula="0.09 + 0.0022·livability + 0.012·max(0, −power margin)" note="A power deficit increases oxygen loss." />
                  <Equation label="Livability / valid action" formula="−0.03 − min(0.30, 0.019·max(0, −power margin)) − empty-reserve penalty + service bonus" note="An empty water or oxygen reserve costs 0.50. Service adds 0.22; Build and delivered-ice gains are applied separately." />
                </GuideSection>
                <GuideSection title="Warning Types">
                  <RuleLine label="Blocked action" text="Orange: invalid command, such as crater movement or off-pad building." />
                  <RuleLine label="Terrain hazard" text="Yellow: allowed risky movement, such as ridge or dune traversal." />
                  <RuleLine label="System warning" text="Cyan/amber: low power margin, dust effects, low battery, or a full rover payload." />
                  <RuleLine label="Exploration ended" text="Red: terminal rover battery, rover health, or livability failure." />
                  <RuleLine label="Mission progress" text="Green/blue: ice collected, new cell scanned, payload unloaded, build pad upgraded, build pad serviced." />
                </GuideSection>
                <GuideSection title="Dust And Weather">
                  Weather drives a stress factor and a dust-intensity delta every step. Clear reduces dust; Dusty and Dust Front increase it; Cold Night slightly reduces it. Service resets dust intensity to 0.08.
                  <Equation label="Solar dust factor" formula="clamp(1 − 0.42·dust intensity, 0.30, 1.00)" note="This factor multiplies solar generation." />
                  <Equation label="Dust stress" formula="0.28 × dust intensity" note="This term is added to battery-drain stress." />
                </GuideSection>
              </Tabs.Content>
            </div>
          </Tabs.Root>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function GuideSection({ title, children, className = "" }: { title: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded border border-stone-800 bg-stone-950/45 p-4 ${className}`}>
      <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-orange-100">{title}</h3>
      <div className="space-y-2 text-sm leading-6 text-stone-300">{children}</div>
    </section>
  );
}

function Legend({ color, label, text }: { color: string; label: string; text: string }) {
  return (
    <div className="flex gap-3">
      <span className={`mt-1 h-4 w-4 shrink-0 rounded border border-white/20 ${color}`} />
      <span>
        <span className="font-medium text-stone-100">{label}</span>
        <span className="text-stone-500"> - {text}</span>
      </span>
    </div>
  );
}

function RuleLine({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <span className="font-medium text-stone-100">{label}</span>
      <span className="text-stone-500"> - {text}</span>
    </div>
  );
}

function Equation({ label, formula, note }: { label: string; formula: string; note: string }) {
  return (
    <div className="overflow-hidden rounded-lg border border-cyan-200/10 bg-black/25">
      <div className="flex items-center gap-2 border-b border-white/[0.06] px-2.5 py-1.5">
        <span className="h-1.5 w-1.5 rounded-full bg-cyan-300/70" />
        <span className="text-[10px] font-semibold text-cyan-100/80">{label}</span>
      </div>
      <div className="px-2.5 py-2">
        <code className="block whitespace-normal font-mono text-[11px] leading-5 text-orange-100">{formula}</code>
        <p className="mt-1 text-[10px] leading-4 text-stone-500">{note}</p>
      </div>
    </div>
  );
}
