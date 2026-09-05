# AresSim Environment Rules Reference

Last updated: 2026-08-24

Status: Canonical Phase 1 simulator contract, implemented by the deterministic Python engine in `engine/`. UI-only presentation behavior is summarized where it touches simulator semantics; the full interface contract lives in [UI Design Reference](ui_design.md), and portable replay behavior lives in [Gameplay Save Format](gameplay_save_format.md).

## 1. Purpose

AresSim is a Mars civilization-building reinforcement-learning environment. Phase 1 runs as a deterministic Python world used by the `web/` UI through REST and by optional in-process Gymnasium/PettingZoo adapters. The backend owns simulation transitions, validation, engine rewards, snapshots, save/replay, local policy observations, legal masks, and RL reward projections. Baselines, seeded rollouts, trajectory replay, and the RLlib masked-PPO training/evaluation pipeline are implemented; additional algorithms and multi-rover learning remain later phases.

The environment should remain:

- Learnable for early RL agents.
- Deterministic from a seed.
- Visually understandable to human players.
- Strict about terrain and action rules.
- Small enough for rapid training iteration.
- Extensible for later colony, multi-agent, scenario, and online deployment work.

## 2. Simulation Scope

Phase 1 models one unmanned rover and one starter landing build pad. The rover explores, scans, extracts resources, services the build pad, builds/upgrades base capability, waits for power recovery, and continues until a terminal survival condition is reached.

The current UI-playable backend does not include:

- RL policy inference or training.
- Gymnasium/PettingZoo observations and adapters, action masks, RLlib rollout collection, or trainers.
- Database or server-side save persistence; downloaded trajectory JSON remains the portable artifact.
- Multiple rovers.
- Human crew simulation.
- Detailed atmospheric, thermal, or mechanical physics.
- Mission checklist victory.
- Hidden step or sol deadline.

Those belong in later `engine/` phases.

## 3. World Size

The default grid is `32 x 32`.

Rationale:

- 32 x 32 gives 1024 cells, which is easier for early RL exploration than a 48 x 48 grid with 2304 cells.
- It is still large enough for resource scouting, terrain avoidance, base-zone planning, and path-history analysis.
- It fits comfortably in the UI at initial zoom.

Coordinates use integer grid cells:

- `x`: horizontal index, 0 to 31.
- `y`: vertical index, 0 to 31.

All action targets are clamped or validated against this grid.

### 3.1 Rover observation window

The canonical world remains `32 x 32`, but an acting rover does not receive the complete map as its policy observation. Phase 1 uses the fixed schema `aresim.obs.local.v1`: an `8 x 8` local square derived from the rover's current position on every reset and step.

- The tensor is always exactly 64 row-major cells.
- The rover occupies local index `[3,3]`; an even window therefore covers world offsets `-3..+4` on each axis.
- `origin = (roverX - 3, roverY - 3)` maps local `[0,0]` to world coordinates.
- In-bounds cells inside that square expose their current terrain and channels.
- Slots falling beyond a world edge remain in place as unknown padding with a zero visibility mask. The crop never shifts to reveal unrelated cells.
- Terrain and resource channels outside the square are not part of the actor observation.
- Phase 1 exposes no occupancy grid or entity table to the actor. The single rover is already represented by `self` and remains anchored at local `[3,3]`; integrated Build Pad components are summarized through terrain and Colony state.
- One categorical `pad_proximity` value replaces separate location booleans: `0` outside service range, `1` within service range, and `2` on the build pad.
- Privileged full state remains available only to the simulator, deterministic replay, evaluation/oracle tooling, and a future centralized critic. It must never enter a decentralized rover policy at inference.

Every Algorithm policy and future RL adapter must consume this local contract (or an explicitly versioned successor), not the browser `SimSnapshot`. Action masks must be derived only from visible facts and public self/colony telemetry; they must not reveal hidden hazards or resources.

## 4. Determinism

The same seed must produce the same:

- Terrain channels.
- Terrain classification.
- Landing build pad.
- Rover starting position.
- Starter infrastructure state.
- Initial weather.
- Initial colony resources.
- Initial reward objective list.
- Initial history events.

Different seeds should usually produce visibly different terrain, resource placement, weather, and starting conditions while still satisfying landing-zone validity.

The Python engine uses deterministic seeded pseudo-random generation and exposes deterministic `reset(seed)` plus stable SHA-256 state checksums as tested contracts.

## 4.1 UI Session Modes

The web UI has a reserved 60 px footer rail for Manual, Algorithm, and Replay. Its three labeled zones separate mode/source selection, active run commands, and world setup. These modes decide which setup and operating controls are visible, but they do not change terrain rules, reward definitions, movement legality, resource mutation, or terminal conditions.

Mode behavior:

- Manual: user actions dispatch directly to the environment through seven commands: Move, Scan, Extract, Build, Service, Unload, and Wait.
- Algorithm: the selected temporary UI-side policy chooses an action; Python validates and executes it as actor `Agent`. Step runs once, while Play schedules the next request only after the previous response completes.
- Replay (`load` in runtime state): uploaded trajectory JSON (or a legacy gameplay file) is parsed and validated by Python. The backend restores checkpoints and applies saved deltas while the UI only presents the returned snapshots.

Cell-boundary visibility, analytical layers, audio mute, 3D Survey/Top/Rover viewpoint, POV yaw and field of view, zoom, fit-to-screen, arbitrary 3D camera angle, mini-map camera centering, and the rover-visibility preview toggle are presentation-only. They must never affect action validity, rewards, terrain channels, or saved environment state. Viewpoints, boundaries, layers, the flashlight preview, and navigation-map panning are controlled from the upper-right Environment panel; they are not extra simulator observation channels or duplicate Inspector tabs. The preview simply darkens everything outside the current `aresim.obs.local.v1` window in all three camera views, making the real policy crop inspectable. Rover POV maps keyboard arrows through its UI-owned yaw before dispatching the same one-cell world movement action: Up/Down are forward/back and Left/Right are lateral relative to the current heading. This input transform does not add orientation to the simulator state or change action legality.

The ambient sky outside rendered world geometry is also presentation-only. Its sun position, dusk/day/night palette, and star opacity are deterministic functions of the snapshot `localTime`; they do not alter weather, solar generation, visibility, terrain lighting, observations, or action outcomes.

In Load mode, the uploaded gameplay is the replay authority. Neither the UI nor backend re-runs action validation or reward formulas while stepping a replay; the backend applies saved timestep deltas or restores checkpoint snapshots.

Only three modes are active. Legacy gameplay metadata may still contain `runMode: "llm"` and `llmAgentId`; those fields are accepted only for replay compatibility and are never emitted by a new gameplay export.

## 5. Terrain Channels

Each cell has continuous channels plus a discrete terrain label.

Required cell fields:

- `x`: cell x coordinate.
- `y`: cell y coordinate.
- `terrain`: discrete terrain type.
- `height`: normalized elevation, 0 to 1.
- `roughness`: normalized traversal difficulty, 0 to 1.
- `ice`: normalized ice signal, 0 to 1.
- `ore`: normalized ore signal, 0 to 1.
- `dust`: normalized cell dust accumulation, 0 to 1.
- `scanned`: whether the cell has already received a successful `Scan`.
- `extracted`: whether ice was extracted from the cell and an extracted marker should be shown.

Important presentation rule:

- Build validity is communicated entirely through the landing build pad terrain and action warnings.
- A successful ice extraction changes the cell to ordinary regolith and retains `extracted: true`. The renderer must keep the full regolith tile visible and add only an amber X/downward extraction pointer; it must never reveal or substitute the brown world foundation.
- Cell boundaries are a UI-only diagnostic option. They are hidden by default so neighboring terrain reads as one continuous surface.
- The build-pad terrain uses a stable very light biscuit material so its manufactured footprint stays distinct from the dark-brown world foundation. A service alarm may temporarily replace that base with the documented red warning treatment.
- Future engine channels must not be added to the docs unless they are present in the current snapshot contract or explicitly marked as future-only.

The 32 x 32 grid is always the authority. Rendering may blend or smooth visuals, but it must not change the terrain cell type, action validity, reward, or failure state.

## 6. Terrain Types

### Regolith

Flat or moderately flat Mars soil.

Rules:

- Rover can move smoothly.
- Normal battery drain.
- May contain low dust and low resource traces.
- Not valid for `Build` unless it is explicitly part of the landing build pad.

Visual role:

- Default rust-colored plain.
- Smooth traversable background.

### Build Pad

Safe landing-zone construction and infrastructure area.

Rules:

- Rover can move smoothly.
- Lowest movement stress.
- `Build` is valid only on build-pad cells.
- `Service` is valid on or near the build pad.
- `Unload` is valid only while the rover itself is on a build-pad cell and its payload is non-empty.
- Habitat, solar array, charging/battery module, and storage are represented as one integrated build-pad site in the UI.
- The build pad should be a compact 5 x 5 area in Phase 1.

Generation constraints:

- Must be on flat regolith.
- Must not be crater, ridge, water/ice-rich, ore-rich, or rough terrain.
- Must have low roughness and safe slope.
- Must be away from crater and ridge hazards.

Visual role:

- Compact landing-zone surface with plates, cable traces, power nodes, and service-state cues.
- It should communicate where infrastructure exists without separate habitat/solar/charger icons.

### Rock / Ore

Rocky or ore-rich terrain.

Rules:

- Rover can move.
- Movement drains more battery than regolith.
- `Scan` is the only resource action valid on rock terrain in Phase 1.
- `Extract` is not valid on rock terrain.
- Ore signal is retained for terrain interpretation and future mechanics, not Phase 1 extraction.

Visual role:

- Dark, chunky outcrop clusters with localized boulder shapes and mineral flecks.
- Should feel like scattered basalt/mineral deposits, not elevated terrain.

### Ice

Subsurface or exposed ice deposit.

Rules:

- Rover can move.
- Movement has moderate battery drain.
- `Extract` is valid when ice is high enough.
- Extraction reduces ice in the cell.
- A successfully extracted ice cell always reclassifies to regolith and retains `extracted: true`, even when its numeric ice channel is not reduced all the way to zero.
- Ice cargo is stored on the rover until it returns to the build pad and explicitly performs `Unload`. Moving onto the pad, waiting, building, or servicing never unloads automatically.

Visual role:

- Blue-gray crystalline deposit patches.

### Crater

Crater basin or unsafe depression.

Rules:

- Rover movement is blocked.
- Attempting to move into a crater emits `Rover cannot move on crater`.
- Build, service, and extraction are invalid unless a future scenario explicitly adds crater-specific mechanics.

Visual role:

- Dark basin with rough rim.
- Must be clearly blocked at a glance.

### Ridge

High roughness elevated terrain band.

Rules:

- Rover movement is allowed.
- Movement emits a terrain hazard warning.
- Movement drains substantially more battery than normal terrain.
- Build pad cannot be generated on or near ridge cells.

Visual role:

- Connected raised rocky bands, not isolated one-cell spikes.
- Uses directional highlight/shadow and longer continuous geometry so it remains visually distinct from Rock/Ore.

### Dune

Loose sand or drift terrain.

Rules:

- Rover movement is allowed.
- Movement emits a terrain hazard warning.
- Movement drains more battery than regolith.
- Build pad should avoid heavy dune zones.

Visual role:

- Coherent bands or patches, not random single cells.

## 7. Terrain Generation Rules

Terrain must feel physically coherent.

Required principles:

- Ridges appear as connected bands.
- Craters appear as contiguous basins with rough rims.
- Ice appears in clustered deposits, often in lower or colder terrain.
- Ore appears in rocky or ridge-adjacent clusters.
- Dunes appear as nearby bands or patches.
- Regolith fills traversable plains between special features.
- Features transition plausibly. A ridge should not abruptly become isolated one-cell regolith without surrounding slope or roughness context.

Current mock generation includes:

- One coherent ridge band shaped by a sinusoidal axis.
- One coherent dune band.
- Multiple deterministic crater basins distributed across the map.
- Multiple deterministic ice clusters, usually biased toward lower/cooler terrain.
- Multiple deterministic ore clusters influenced by rock/ridge proximity.
- Smooth noise for height and roughness.
- Landing pad search with deterministic retries and a sanitized fallback.

Rendering boundary:

- Terrain may be visually smoothed, blended, labeled, textured, or decorated in the UI.
- Texture blending, rocks, dust streaks, glints, object labels, and shadows are presentation only.
- Visual smoothing must never change movement legality, action validation, reward, or terminal conditions.

## 8. Landing Build Pad Rules

Current size:

- Radius 2 around the landing center.
- Total area: 5 x 5 cells.

The build pad must be:

- Contiguous.
- Flat.
- Low roughness.
- Non-crater.
- Non-ridge.
- Not ice-rich.
- Not ore-rich.
- Safe for starter infrastructure.

Starter infrastructure is contained inside the build pad:

- Habitat capacity.
- Solar generation.
- Charging/battery interface.
- Storage.

UI rule:

- The UI renders the build pad as one integrated infrastructure site.
- It should not show separate habitat, solar, or charging icons in Phase 1.
- Service-needed state should be visible on the pad through warning glow, cracks, dim cables, or similar cues.

If the generator cannot find a valid 5 x 5 area, it must:

1. Retry with deterministic seed offsets.
2. Use a bounded retry count.
3. Fall back to a fixed safe landing patch.
4. Sanitize that patch into valid build-pad cells.

## 9. Entities

### Rover

The rover is unmanned.

Rover state:

- `id`
- `name`
- `x`
- `y`
- `battery`
- `health`
- `cargoIce`
- `cargoOre`
- `cargoSamples`
- `cargoCapacityKg`
- `currentTask`

The rover does not consume oxygen or water.

Rover survival resources:

- Battery.
- Health.
- Cargo load effects.
- Shared payload capacity. Phase 1 uses a 12 kg bay across ice, ore, and geological samples.
- Dust and maintenance state in future phases.

### Build Pad Infrastructure

The Python snapshot includes typed structures internally so the engine can model health and power. In the Phase 1 UI, these structures are presented as one build-pad site instead of separate map icons.

Internal infrastructure roles:

- Habitat: consumes power and supports livability.
- Solar array: generates power based on weather, dust, and health.
- Charging/battery module: enables recharge when power margin is positive and the rover is on the pad.
- Storage: receives converted ice resources.

`Service` operates on the build-pad infrastructure as a whole. `Unload` transfers the rover's complete payload to storage; these remain separate actions so maintenance and logistics are independently observable.

Build-pad state:

- `normal`: infrastructure is stable.
- `needs_service`: service is required after dust exceeds `0.78`, any internal structure falls below `62%` health, or a severe power margin below `-8` coincides with dust above `0.55`.
- `habitat_built`: habitat construction reached 100 percent.
- `habitat_built_needs_service`: habitat is complete and infrastructure also needs service.

The service warning is deliberately latched: once one of those conditions is reached, it remains active until a successful `Service` action. Moderate dust, routine wear, and small or transient power deficits do not request service. Infrastructure degrades more gradually each simulation step:

```text
infrastructureDamage = weatherDamage + dustIntensity * 0.025

weatherDamage:
  Dust Front = 0.120
  Dusty      = 0.045
  Clear or Cold Night = 0.015
```

The service-needed state should be represented primarily as a red glowing build-pad visual, not as a generic top-level warning toast.

## 10. Resources

### Rover Battery

Battery is the main rover survival resource.

Rules:

- Movement drains battery.
- Difficult terrain drains more battery.
- Extract, Build, and Service have deterministic action costs. Wait has no direct rover action cost and instead uses the power-margin charging rule.
- Dust, weather, roughness, and carried cargo affect drain.
- Power deficit adds extra drain.
- Positive power margin can recharge the rover when waiting or when the rover is on the build pad.
- Battery at 0 percent causes terminal exploration failure.

Failure message:

`Exploration ended..... Rover battery depleted`

### Rover Health

Health represents rover operability.

Rules:

- Hazardous terrain, high dust/weather stress, and poor power conditions can reduce health.
- Service can provide small recovery.
- Health at 0 percent causes terminal exploration failure.

Failure message:

`Exploration ended..... Rover can no longer operate`

### Colony Water

Water belongs to the base/colony life-support system, not the rover.

Rules:

- Display as Colony Water.
- Drains deterministically every step.
- Ice cargo is converted into water only by a successful `Unload` on the build pad.
- Low water reduces livability over time in future engine phases.

### Colony Oxygen

Oxygen belongs to the base/colony life-support system, not the rover.

Rules:

- Display as Colony O2.
- Drains deterministically every step.
- Ice conversion also contributes to O2 in the Phase 1 mock.
- It is not consumed by the unmanned rover.

### Livability

Livability measures how survivable the starter base is.

Rules:

- Build and Service actions can increase livability.
- Ice delivery can slightly improve livability.
- Stable operation now has a small passive livability decline of `0.03` per valid action; power deficit and colony resource depletion accelerate it.
- Livability at 0 percent causes terminal exploration failure.

Failure message:

`Exploration ended..... Habitat livability collapsed`

### Rover Payload

The rover has one shared mass-constrained payload bay:

```text
payloadCapacityKg = 12
payloadUsedKg = cargoIce + cargoOre + cargoSamples
payloadRemainingKg = max(0, payloadCapacityKg - payloadUsedKg)
```

- A successful `Scan` stores a 0.5 kg geological sample.
- A successful `Extract` stores 2 kg of ice.
- Scan or Extract is blocked before mutation when its full result cannot fit; partial collection is not performed.
- Entering the build pad does not transfer cargo automatically.
- `Unload` transfers all ice, ore, and samples in one atomic action and is valid only when the rover is on the build pad with a non-empty payload.
- Ice is processed into colony water and oxygen during unload. Samples and ore are recorded as delivered inventory but do not yet produce a colony resource.
- Payload mass contributes to rover battery stress.

### Power

Power is a first-class environment system.

Tracked values:

- `powerGenerated`
- `powerConsumed`
- Power margin = generated minus consumed.

Generation formula:

```text
generated = solarPanelCount * 42 * averageSolarHealth * weatherFactor * dustFactor
averageSolarHealth = mean(solarHealthPercent / 100)
dustFactor = clamp(1 - dustIntensity * 0.42, 0.30, 1.00)
```

Weather factors:

- Clear: `1.18`
- Dusty: `0.72`
- Dust Front: `0.48`
- Cold Night: `0.22`
- Severe Storm: `0.36`

Consumption formula:

```text
consumed = 6.5
         + habitatCount * 4.2
         + storageCount * 1.1
         + chargerOrBatteryCount * 1.6
         + actionLoad
```

Action loads:

- Scan: `0.35`
- Extract: `1.4`
- Build: `1.8`
- Service: `0.8`
- Unload: `0.25`
- Wait: `0.0`
- Move: `0.15`

Effects:

- Negative margin adds battery drain: `min(1.6, abs(powerMargin) * 0.055)`.
- Wait runs one charging cycle when power margin is positive: `min(18.0, powerMargin * 0.85)`.
- Passive charging on the build pad is slower than Wait: `min(4.0, powerMargin * 0.25)`.
- Wait has no direct rover battery drain. If power margin is negative, the deficit model can still drain battery and no recharge is applied.
- A single Wait can fully recharge only when the remaining battery deficit is less than the calculated charging-cycle amount. Otherwise, multiple Wait actions are required and each one advances simulation time/history.
- Charger only helps when the build pad has positive usable power.
- Service improves infrastructure health and reduces dust intensity.

## 11. Battery Drain Formula

Battery drain is deterministic and deliberately slower than a fixed large step penalty. It uses exponential stress so rough terrain matters without instantly ending a run.

```text
batteryDrain = actionBaseDrain(action) * exp(totalStress * 0.32) + powerDeficitPenalty - recharge

totalStress = terrainStress
            + roughness * 0.45
            + cellDust * 0.22
            + weatherStress
            + actionStress
            + cargoStress

cargoStress = payloadUsedKg * 0.012
```

Action base drain:

- Move: `0.32`
- Scan: `0.24`
- Extract: `0.82`
- Build: `1.05`
- Service: `0.52`
- Unload: `0.18`
- Wait: `0.00`
- Event/invalid: `0.16`

Terrain stress:

- Build Pad: `0.05`
- Regolith: `0.14`
- Ice: `0.30`
- Rock/Ore: `0.38`
- Dune: `0.55`
- Ridge: `0.82`
- Crater: blocked before drain.

Action stress:

- Extract: `0.58`
- Build: `0.66`
- Service: `0.28`
- Unload: `0.04`
- Scan: `0.08`
- Wait: `0.02`, but Wait currently has no direct drain because its base drain is `0.00`.
- Move: `0.02`

Weather stress:

- Clear: `-0.06 + dustIntensity * 0.28`
- Dusty: `0.22 + dustIntensity * 0.28`
- Dust Front: `0.48 + dustIntensity * 0.28`
- Cold Night: `0.28 + dustIntensity * 0.28`
- Severe Storm: `0.62 + dustIntensity * 0.28`

Relative behavior:

- Regolith and build pad are efficient.
- Ice and ore are moderate.
- Dune and ridge warn and cost more.
- Crater is blocked.
- Carrying ice, ore, or samples slowly increases drain.

## 12. Colony Life-Support Formula

Water and oxygen drain every step. This is deterministic and belongs to the base, not the rover.

```text
waterDrain = 0.07 + livability * 0.0018 + dustIntensity * 0.035
oxygenDrain = 0.09 + livability * 0.0022 + max(0, -powerMargin) * 0.012

livabilityDelta = -0.03
                 - min(0.30, max(0, -powerMargin) * 0.019)
                 - (0.50 if water <= 0 or oxygen <= 0, otherwise 0)
                 + (0.22 if action is Service, otherwise 0)
```

The livability delta applies once per valid action. A successful Build adds `1.10` separately before this delta, and delivered ice adds `cargoIce * 0.08` separately during Unload. This keeps infrastructure and delivery gains meaningful while making unattended habitat decline slightly faster.

Ice conversion occurs when:

- Rover is on the build pad.
- Rover payload is non-empty.
- Current action is an explicitly validated `Unload`.

Conversion:

```text
waterGain = cargoIce * 1.8
oxygenGain = cargoIce * 0.8
livabilityGain = cargoIce * 0.08
cargoIce = 0
cargoOre = 0
cargoSamples = 0
```

Delivered ice, ore, and sample counters are updated in the same atomic transition. This gives collection strategic value while keeping transport and delivery visible to policies and replay viewers.

## 13. Weather And Dust

Weather states:

- Clear.
- Dusty.
- Dust Front.
- Cold Night.
- Severe Storm, reserved for future extension.

Rules:

- Dusty and Dust Front reduce solar generation.
- Cold Night sharply reduces solar generation.
- Dust intensity influences power output and battery stress.
- Dust warnings should appear as system warnings.
- Service reduces dust intensity in the mock system.
- Future storm states can add movement risk, visibility reduction, and health damage.

Top-ribbon rule:

- Weather remains in the top status ribbon.
- Dust is not a top-ribbon metric in Phase 1.
- Cell dust remains visible in the inspector and optional dust analytical layer.

## 14. Action Space

Phase 1 visible action space is intentionally small for RL friendliness.

The policy adapter is one flat masked `Discrete(10)` head: Wait, four cardinal Move outputs, Scan, Extract, Build, Service, and Unload. Phase 1 does not split verb selection and movement direction into conditional heads; each logit maps directly to one stable command for PPO, DQN, replay, and debugging.

Visible actions:

- Move.
- Scan.
- Extract.
- Build.
- Service.
- Unload.
- Wait.

Internal actions:

- Invalid.
- Event.

Removed visible actions:

- Mine Ice.
- Mine Ore.
- Connect.
- Repair.
- Clean.
- Invalid Action Test.

Reason for simplification:

- `Extract` replaces separate mining actions.
- `Service` replaces connect, repair, and clean.
- `Unload` is distinct from movement and build-pad entry so agents can learn explicit logistics and future cargo routing can extend the same action.
- A smaller action space makes early policy learning easier.
- Contextual validation still lets the environment enforce correct behavior.

## 15. Action Rules

### Move

Inputs:

- Arrow-key direction or target cell.

Valid if:

- Target is inside the grid.
- Target is not crater.
- Game is not terminal.

Effects:

- Rover position changes.
- Step count increments.
- Battery drain is computed by the deterministic formula.
- Dune and ridge movement emit terrain hazard warnings.
- Full rover path history is retained for trail rendering and replay.

Blocked movement:

- Emits blocked warning.
- Appends invalid history entry.
- Does not mutate rover position.
- Does not grant normal reward.

### Scan

Inputs:

- Target cell, usually selected cell or suggested target.

Valid if:

- Target is inside the grid.
- Target terrain is `rock`, representing a rock/ore outcrop.
- Target cell has not already been scanned.
- At least 0.5 kg of payload capacity remains.

Effects:

- Sets `cell.scanned = true`.
- Adds a deterministic 0.5 kg geological sample to `cargoSamples`.
- Rewards information gain once per rock/ore outcrop cell.
- Appends history.

Invalid scan:

- Emits blocked warning.
- Appends invalid history entry.
- Does not grant information reward.
- Occurs when the target is already scanned, the target is not a rock/ore outcrop, or the 0.5 kg sample cannot fit.

### Extract

Inputs:

- Target cell.

Valid if:

- Target has enough ice.
- At least 2 kg of payload capacity remains.

Effects on valid ice extraction:

- Reduces the `ice` channel by the deterministic extraction amount.
- Increases rover cargo ice by the deterministic 2 kg cargo amount.
- Sets `cell.extracted = true`.
- Always reclassifies the extracted cell to `regolith`, even if a residual numeric ice channel remains. The original fixed `iceSitesTotal` retains the session objective denominator.
- Renders as the same full-height regolith surface as every other regolith cell, with the amber extraction marker layered above it.
- Appends resource reward and history.

Invalid extraction:

- Emits blocked warning.
- Appends invalid history.
- Occurs on rock/ore terrain even when scan is valid there.
- Occurs when the complete 2 kg extraction cannot fit; extraction is never partial.
- Does not mutate resources.
- Does not grant normal reward.

### Unload

Inputs:

- No remote target. The rover's current position is authoritative.

Valid if:

- Rover is standing on a `build_pad` cell.
- `payloadUsedKg > 0`.

Effects:

- Transfers all carried ice, ore, and samples to the base in one atomic step.
- Clears `cargoIce`, `cargoOre`, and `cargoSamples`.
- Converts delivered ice into water, oxygen, and the documented livability gain.
- Increments delivered-resource counters and `unloadCount`.
- Adds delivery/logistics reward terms, history, audio, and completion feedback.

Invalid unload:

- Is blocked away from the build pad or when payload is empty.
- Does not remove or convert any cargo.

### Build

Inputs:

- Target build-pad cell.

Valid if:

- Target is inside the landing build pad.
- Target terrain is `build_pad`.
- Habitat build progress is below 100 percent.

Invalid if:

- Target is outside the build pad.
- Target is crater, ridge, dune, ice, rock, or ordinary regolith.
- Target has unsafe terrain channels.
- Habitat construction is already complete.

Effects:

- Represents build-pad base upgrade or habitat capability.
- Consumes deterministic battery/power.
- Increases livability.
- Adds infrastructure/progress reward.
- Adds 10 percent habitat progress per successful Build.
- Habitat completion requires 10 successful Build actions.

### Service

`Service` is contextual. It reduces RL complexity by combining connect, repair, and clean.

Inputs:

- Target cell on or near the landing build pad.

Valid if:

- Rover/target is within the service radius of the build pad.
- Game is not terminal.

Contextual effects:

- Cleans/synchronizes solar output.
- Repairs internal structures to `100%` health.
- Reconnects the build-pad power bus.
- Resets dust intensity to `0.08`.
- Clears the latched service warning.
- Can slightly improve livability.

Invalid if:

- No build-pad infrastructure is nearby.

Invalid effects:

- Blocked warning.
- Invalid history entry.
- No resource mutation.
- No normal reward.

### Wait

Inputs:

- None, or current rover position.

Valid if:

- Game is not terminal.

Effects:

- Advances step/time.
- Recharges battery only when power margin is positive.
- Recharge amount is deterministic and based on available power: `min(18.0, powerMargin * 0.85)` battery percentage points.
- Does not have a direct rover action drain; battery can still fall during Wait if the base is in power deficit.
- May expose power deficit or dust system warnings.
- Records a history event showing whether the Wait charging cycle restored battery or failed due to non-positive power margin.
- Appends history.

Wait should not be a free reward source. Reward shaping should prevent idle exploit loops, and backend policies should avoid repeatedly waiting after the battery reaches 100%.

## 16. Validation And Invalid Actions

The environment must validate every action before mutation.

Invalid actions must:

- Not mutate rover position.
- Not mutate terrain resources.
- Not mutate colony resources except for explicit penalty mechanics.
- Append an invalid history entry.
- Emit a clear warning.
- Apply zero or negative reward.

Examples:

- Moving into crater.
- Scanning an already-scanned cell.
- Extracting from a cell without enough ice.
- Scanning or extracting without enough free payload capacity.
- Unloading away from the build pad or with an empty payload.
- Building outside the build pad.
- Servicing away from the build pad.
- Targeting outside the grid.

## 17. Warning Taxonomy

Warnings are typed. The UI must not show every warning as the same generic design.

### Blocked Action

Use for invalid actions.

Style intent:

- Amber/orange.

Examples:

- `Rover cannot move on crater`.
- `Build actions require the landing-zone build pad`.
- `Extract requires an ice deposit`.
- `Service requires the rover to be on or near the build pad`.
- `Scan target already scanned`.
- `Scan requires 0.50 kg free payload; return to the build pad and Unload`.
- `Extract requires 2.00 kg free payload; return to the build pad and Unload`.
- `Unload requires the rover to be on the build pad`.

### Terrain Hazard

Use for allowed risky movement.

Style intent:

- Yellow.

Examples:

- Ridge traversal warning.
- Dune traversal warning.

Behavior:

- Action succeeds.
- Battery drain is higher.
- History includes warning event.

### System Warning

Use for power, weather, dust, low battery, or low resource margin.

Style intent:

- Cyan/amber.

Examples:

- Power deficit.
- Dust reducing solar output.
- Low battery.
- Build pad service needed.

### Exploration Ended

Use for terminal failure.

Style intent:

- Red.
- Larger and more persistent.

Examples:

- Battery reaches 0 percent.
- Rover health reaches 0 percent.
- Habitat livability reaches 0 percent.

### Mission Progress

Use for successful milestone progress.

Style intent:

- Green or blue.

Examples:

- Ice collected.
- New cell scanned.
- Build pad upgraded.
- Build pad serviced.
- Payload unloaded.

## 18. Rewards

Phase 1 rewards are mock shaping values, not final RL reward definitions.

Reward objectives:

- Collecting Ice.
- Returning Payload.
- Scanning Terrain.
- Building Habitat.
- Servicing Build Pad.

Live objective stats:

- `iceCollected`: cumulative ice units extracted.
- `iceDelivered`: cumulative ice mass transferred to the base.
- `samplesCollected`: cumulative geological sample mass collected by Scan.
- `samplesDelivered`: cumulative sample mass transferred to the base.
- `unloadCount`: successful unload actions.
- `iceSitesExtracted`: count of unique ice sites successfully extracted.
- `iceSitesTotal`: count of cells that start as valid extractable ice sites for the generated seed.
- `terrainScanned`: unique terrain cells scanned.
- `rockSitesTotal`: count of cells that start as valid scannable rock/ore sites for the generated seed.
- `habitatBuildProgress`: habitat construction progress, 0 to 100.
- `habitatBuildCount`: successful build actions, capped by the 10-step habitat rule.
- `serviceCount`: successful service actions.
- `rewardTotals`: cumulative reward totals for ice collection, payload delivery, terrain scanning, habitat building, servicing, traversal shaping, blocked-action penalties, and total reward.

Principles:

- Reward useful exploration and information gain.
- Reward resource collection only when resources exist.
- Reward build/service only when they improve infrastructure, livability, or power conditions.
- Give ordinary movement only a very small shaping reward so policies do not learn to move only for reward farming.
- Penalize invalid actions.
- Penalize unsafe terrain through battery and safety terms.
- Avoid making Wait an exploit.
- Keep reward terms separate from total reward for analysis.

Current reward-term categories may include:

- `information`
- `exploration`
- `resource`
- `delivery`
- `logistics`
- `infrastructure`
- `livability`
- `service`
- `efficiency`
- `traversal`
- `energy`
- `safety`
- `invalid`
- `recovery`

Invalid actions should receive negative or zero reward terms and no normal success reward.

Movement reward:

- Normal traversal reward is intentionally tiny: around `0.01` before energy/safety terms.
- Build-pad movement is even smaller: around `0.002`.
- Ridge and dune movement may still produce a negative net reward because hazard/safety and energy costs dominate.
- Movement exists primarily to let the agent reach useful objectives, not as a main reward source.

Blocked penalty:

- Invalid or blocked actions contribute to `rewardTotals.blockedPenalty`.
- Blocked penalty also contributes to `rewardTotals.total`, so total reward reflects mistakes as well as successes.

## 19. Reward Objectives

Phase 1 no longer uses a mission checklist or checklist victory. The environment is open-ended:

- Explore the 32 x 32 Mars map.
- Scan each rock/ore outcrop at most once.
- Collect ice when valid.
- Manage the shared 12 kg payload bay and explicitly Unload at the build pad to deliver samples and support colony water/O2.
- Build habitat capability on the build pad.
- Service the build pad when dust, damage, or power margin requires it.
- Continue until rover battery, rover health, or habitat livability reaches 0.

Presentation markers:

- Scanned rock/ore cells show cyan analytical rings.
- A successful extraction immediately changes that cell to ordinary full-height regolith, while an amber X, downward pointer, and amber ring identify the extraction site.
- These markers are replay/analysis visuals only and do not change terrain rules.

Presentation feedback:

- Each new history action can trigger a short target pulse/beam and a synthesized action-specific sound, including a distinct Unload confirmation cue.
- A transition into build-pad service-needed state has its own request cue and persistent red world-space warning treatment.
- Manual actions, Algorithm steps/autoplay, and Replay steps use the same history-driven feedback path.
- Muting audio suppresses the cue but does not suppress the history event, warning, animation, reward, or environment mutation.

Objective progress:

- Ice Collected progress is `iceSitesExtracted / iceSitesTotal`.
- Terrain Scanned progress is `terrainScanned / rockSitesTotal`.
- Denominators are computed at session creation from the seed-generated map and remain fixed throughout the run.
- Extracted ice cells become regolith after extraction, but this does not reduce `iceSitesTotal`.
- Scanned rock cells remain part of `rockSitesTotal`; repeated scans are invalid and do not increase progress.

Step count is elapsed simulation time only. There is no hidden movement limit and no hard sol window in Phase 1.

## 20. Terminal Failure Conditions

Current terminal failures:

- Rover battery reaches 0 percent.
- Rover health reaches 0 percent.
- Habitat livability reaches 0 percent.

Removed Phase 1 conditions:

- Step deadline.
- Hidden movement limit.
- Checklist victory.

Future failures may include:

- Critical colony oxygen depletion.
- Critical colony water depletion.
- Solar system destroyed during storm.
- Rover stranded without recharge path.
- Scenario-specific deadline, only if explicitly configured.

## 21. Step Count And Time

Step count is elapsed simulation steps.

Rules:

- Each valid action increments step.
- Invalid attempted actions may also append history and increment step if the environment treats them as consumed decisions.
- Step count is not a movement limit.
- Local time and sol advance from step count.
- The gameplay state contains no remaining-step or remaining-Sol budget, and neither value belongs in the Phase 1 actor observation.

An RL training or evaluation wrapper may impose a configured `max_episode_steps` safeguard. This is external to simulator rules: reaching it returns `truncated = True`, not terminal failure, and does not create an in-game deadline. The limit belongs in the experiment configuration rather than WorldState, Scenario state, or `self`. Sol is derived from step count, so an explicit future timed task must define a single deadline basis rather than simultaneous step and Sol budgets.

UI placement:

- Elapsed simulation steps belong in Agent Step History, not the top ribbon.

## 22. Action History

Every action or event should append a history record.

Required fields:

- Step number.
- Actor.
- Action type.
- Target, if any.
- Result.
- Reward.
- Reward terms.
- Resource delta.
- Event labels or notes.

History supports:

- Replay analysis.
- RL debugging.
- Reward inspection.
- Human review of policy behavior.
- Click-to-highlight target in the map.
- Full rover-path trail rendering.
- Directional arrows on the rover path at regular intervals.

Invalid action history must be explicit and filterable.

## 23. Snapshot Contract

The current frontend expects snapshots with:

- Session metadata:
  - Session id.
  - Seed.
  - Step.
  - Sol.
  - Local time.
  - Mode.
- Game status:
  - Running.
  - Paused.
  - Game over.
  - Status reason.
- Environment:
  - Weather.
  - Dust intensity.
  - Terrain size.
- Resources:
  - Power generated.
  - Power consumed.
  - Battery.
  - Colony water.
  - Colony oxygen.
  - Livability.
- Mission/open-exploration metadata:
  - Objective.
  - Reward objectives.
  - Alerts.
- Objective stats:
  - Ice collected.
  - Ice and samples delivered, samples collected, and unload count.
  - Terrain scanned.
  - Habitat build progress/count.
  - Service count.
  - Reward totals.
- Build-pad state:
  - Service-needed boolean.
  - Build-pad status.
- Rules:
  - Survival, power, movement, service, and warning rules.
- Terrain:
  - 32 x 32 cells.
  - Scanned/extracted state flags.
- Entities:
  - Rover.
  - Internal build-pad structures.
- Action history:
  - Newest-first snapshot entries; save export converts them to chronological `steps[]`.

UI numeric formatting:

- Power margin uses two decimals.
- Battery uses two decimals.
- Colony Water, Colony O2, Livability, solar generation, and load use two decimals when shown as numeric status values.

`integrations/ui.py` preserves these concepts as the active camel-case REST snapshot contract. A breaking field change requires an explicit compatibility migration.

## 24. Gameplay Save And Replay Contract

Portable exports use the unified `aresim.trajectory.episode.v1` schema documented in [Trajectory Episode and Replay Projection](gameplay_save_format.md). Legacy `aresim.gameplay.v1` files remain importable.

Required trajectory/replay content:

- Trajectory schema version and episode metadata.
- Optional policy trace (`null` for UI-authored sessions).
- Replay metadata, initial snapshot, chronological step deltas, automatic checkpoints, final snapshot, and integrity counters.

Step deltas must track:

- Action.
- Actor.
- Target.
- Result.
- Events.
- Reward and reward terms.
- Resource delta.
- Terrain/entity/resource/objective/build-pad mutations for that timestep.
- Appended history entry.

Automatic checkpoints must include:

- Step 0 Initial checkpoint.
- Every 10 steps.
- Significant event steps.
- Final checkpoint.

Significant replay events include:

- Invalid or blocked action.
- Terrain hazard.
- Extract success.
- Unload success.
- Habitat build progress change.
- Service success.
- Low battery or system warning.
- Game over.

Checkpoint snapshots are intentionally full snapshots. This lets Load mode jump directly to a point in the run without replaying every previous delta.

Replay semantics:

- Loading a gameplay file restores its embedded seed and replay metadata.
- Jumping to a checkpoint sets the visible environment to that checkpoint snapshot.
- Stepping after a checkpoint applies the next saved timestep delta.
- Repeat replay returns to the Initial checkpoint.
- Replay mode does not re-simulate actions, validation, rewards, or terrain rules.
- Replay never exposes seed application, randomization, manual actions, or live-run saving; the uploaded file remains authoritative.
- Legacy saves with LLM metadata remain replayable, but LLM is not an active mode.
- Legacy `{ savedAt, snapshot }` and raw `SimSnapshot` files may be upgraded in memory, but intermediate replay quality depends on available history.

Save files are part of the active backend contract. Python exports and imports this shape; transport, compression, or storage may evolve later without changing replay meaning.

## 25. Active Backend Boundary

The implemented `engine/` provides:

- deterministic reset and step through `AresEngine`;
- frozen grouped configuration and one immutable `DEFAULT_ENGINE_CONFIG` in `defaults.py`;
- typed action validation, mutation, named reward terms, warnings/events, and terminal rules;
- camel-case UI snapshots through `integrations/ui.py`;
- gameplay deltas, initial/interval/significant/final checkpoints, export, legacy normalization, and replay reconstruction;
- one serialized in-memory live session and one loaded replay through `AresService`;
- a local FastAPI transport with one typed error envelope.
- a framework-neutral `AresEnvironment` composition over the same canonical engine;
- the full rover-centered `aresim.obs.local.v1` numerical observation with fixed edge padding;
- the fixed `aresim.action.rover.v1` ten-action adapter and validation-derived legal mask;
- open-exploration shaped-training and sparse-evaluation reward projections;
- exactly-one-rover Gymnasium and PettingZoo Parallel adapters with transition parity.

Active REST routes:

- `GET /api/health`
- `POST /api/sessions`
- `GET /api/sessions/{sessionId}`
- `POST /api/sessions/{sessionId}/actions`
- `POST /api/sessions/{sessionId}/pause`
- `POST /api/sessions/{sessionId}/resume`
- `POST /api/sessions/{sessionId}/save`
- `POST /api/replays`
- `POST /api/replays/{replayId}/step`
- `POST /api/replays/{replayId}/jump`
- `POST /api/replays/{replayId}/reset`

The Gymnasium and PettingZoo adapters are optional in-process Python APIs; they do not add REST routes or alter frontend snapshot/save contracts. Their policy input is always local and their RL-facing shaped/sparse rewards remain separate from engine/UI rewards. External truncation, seeded baseline rollouts, unified UI-loadable trajectories, fixed evaluation splits, and RLlib masked-PPO training are implemented through the same environment boundary. W&B owns training metrics, while checkpoints, fixed-seed evaluation, trajectories, and Jupyter analysis remain reproducibility artifacts. The project still has no model-serving endpoint, database, authentication, WebSocket, server-side save library, multi-rover mechanics, or additional learned algorithm. Future additions must reuse the canonical environment instead of duplicating its rules and are described in [Agent Data, RL, and LLM Architecture Proposal](../rl/agent_data_rl_llm_proposal.md) and [RL Algorithms, Training, and Evaluation](../rl/rl_quickstart.md).

## 26. Update Checklist

When environment behavior changes, update this document and the implementation together:

- Terrain types or terrain channels.
- Grid size.
- Build-pad generation.
- Entity state.
- Action set.
- Action validation.
- Reward terms.
- Resource model.
- Power model.
- Battery formula.
- Life-support formula.
- Warning taxonomy.
- Open-exploration objectives.
- Terminal conditions.
- Snapshot contract.
- Replay/history schema.
- Analytics series shape (`AnalyticsSeriesPoint`) and live/loaded parity.
- Gameplay save/checkpoint schema.
