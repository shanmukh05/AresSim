# AresSim UI Design Reference

Last updated: 2026-08-02

Status: Canonical implemented specification for the orthographic 3D simulator shell, renderer boundary, extension direction, and multi-agent UI readiness. Simulator semantics live in [Environment Rules Reference](environment_rules.md), and portable replay serialization lives in [Gameplay Save Format](gameplay_save_format.md). This document supersedes the separate historical 3D redesign proposals.

## 1. Product Goal

AresSim should open directly into a playable Mars habitat simulation. The environment is a deterministic 2D grid, but the default presentation is an orthographic 3D world with terrain elevation, structures, rovers, paths, deposits, shadows, and optional analytical overlays.

The interface must serve two audiences without becoming a dashboard wall:

- players operating a Mars habitat;
- researchers watching, replaying, and debugging policies.

The world is the primary surface. Mission, alert, action, inspection, replay, and analytics interfaces appear as compact HUD elements or contextual surfaces rather than permanent side columns.

## 2. Current Technology

- React 19 + TypeScript + Vite.
- Python 3.12 + FastAPI for authoritative local sessions, transitions, saves, and replay reconstruction.
- Three.js through React Three Fiber for the world renderer.
- `@react-three/drei` for scene utilities and GPU instancing helpers.
- Zustand for asynchronous REST-client state plus presentation-only UI state.
- Tailwind CSS for layout and visual tokens.
- Radix UI for accessible dialogs, tabs, tooltips, selects, sliders, and popovers.
- Recharts for run analytics.
- Vitest/Testing Library and Playwright for automated verification.

The simulator remains the source of truth. Three.js is a presentation dependency, not part of environment rules.

## 3. Layout

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ ARESIM · STATUS · SOL/TIME · WEATHER    POWER · ROVER · HABITAT · HUD  │
├─────────────────────────────────────────────────────────────────────────┤
│ Mission HUD                                      Environment controls   │
│ Alert chip                                                              │
│                                                                         │
│                    ORTHOGRAPHIC 3D MARS WORLD                           │
│                                                                         │
│ Context launcher                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ MODE & SOURCE       ROVER/RUN/REPLAY COMMANDS       WORLD SETUP          │
└─────────────────────────────────────────────────────────────────────────┘
                                                  ┌───────────────────────┐
                                                  │ Context drawer        │
                                                  │ mission / inspector   │
                                                  │ history / systems     │
                                                  └───────────────────────┘
```

The application has three permanent regions:

1. A 58 px compact gameplay header.
2. The flexible middle viewport, owned by the world.
3. A reserved 60 px Action Bar footer below the world.

Everything else inside the world region is floating, contextual, or transient. The footer is a sibling of the world viewport, so world geometry can never sit behind it.

## 4. Command Header

The header provides only information and operations that must remain globally reachable.

Left:

- AresSim identity.
- Habitat/scenario identity.

Center:

- sol and local time;
- running, paused, or terminal state;
- weather on larger screens.

Right:

- power margin;
- rover battery;
- rover health;
- habitat construction progress;
- habitat livability;
- mission drawer;
- history drawer;
- analytics;
- game guide.

Responsive priority:

- Power and status remain visible longest.
- The complete Power/Battery/Health/Habitat/Livability telemetry group appears on larger screens. Habitat is construction completion, while Livability is the colony survival resource; they are intentionally separate values.
- Water and oxygen live in the Colony Systems drawer rather than competing for header space.
- Icon operations always retain accessible names.

Run mode is intentionally not shown as a header badge. Mode selection belongs to the unified bottom command deck.

Only warnings and failures use saturated colors. Healthy resource values use restrained green.

## 5. World Renderer

### 5.1 Projection and camera

- 3D Survey is the default view. It uses an angled orthographic camera with a 45-degree north-east starting angle so terrain elevation, structures, deposits, routes, and rover activity remain spatially legible.
- Rotating 3D Survey orbits the angled camera through the complete 0-359 degree range without changing simulation coordinates.
- Top is a separate directly overhead orthographic view. It is north-up and fixes the square environment's sides parallel to the browser window instead of presenting a 45-degree diamond.
- Fit mode frames the entire map on initial load and session reset.
- Button or mouse-wheel zoom switches 3D Survey or Top to manual zoom and shows the mini navigation view.
- A continuous 0-359 degree rotation slider allows any horizontal viewing angle.
- The current degree button resets 3D Survey to the 45-degree north-east angle. Top intentionally exposes no rotation control.
- Fit returns to the complete-world view without changing simulation state.
- Zoom and camera angle persist across simulation actions, automatic steps, and replay snapshots. The render loop reapplies the stored orthographic target and zoom, so a new simulator snapshot cannot restore the Canvas bootstrap camera.
- In manual zoom, the camera initially follows the active rover. Clicking the mini navigation map pans the camera to the selected world region and clamps the viewport within terrain bounds. The next rover movement resumes follow without rotating the 3D Survey or Top alignment or resetting zoom.
- Fit mode always targets the mathematical center of the world. Rover movement, a new seed, and randomization cannot translate the fitted environment.
- Each view owns its fit calculation: 3D Survey includes the angled terrain footprint, while Top fits the exact window-aligned world rectangle.
- Terrain is seamless by default. An optional cell-boundary control in the Environment panel introduces narrow gaps for debugging exact grid coordinates.

The empty world margin is a time-aware ambient backdrop derived from snapshot `localTime`. An illustrated Sun uses a bright photosphere, corona, sparse rays, and dust-colored glow as it travels from the lower-left dawn horizon through a noon apex to the lower-right sunset horizon. The background interpolates between restrained Martian day, dawn/dusk, and night colors; a static sparse star field fades in only at night. The WebGL canvas remains transparent outside rendered geometry, placing this layer behind—never over—the terrain, HUDs, selection markers, or controls. Ambient transitions occur only when simulator time changes, last 1.4 seconds, and are disabled under reduced-motion preferences. Scene lighting and terrain colors remain stable so time-of-day presentation cannot obscure gameplay state.

The night guide is informed by NASA observations. Mars has two moons: Phobos is the larger, irregular, heavily cratered moon and completes roughly three orbits per day; Deimos is smaller, smoother, and takes about 30 hours ([NASA Mars Moons](https://science.nasa.gov/mars/moons/), [NASA moon facts](https://science.nasa.gov/mars/moons/facts/)). From the surface, Phobos appears about one-third the apparent diameter of Earth's Moon, while Deimos resembles a bright star ([NASA CRISM comparison](https://science.nasa.gov/resource/crism-views-phobos-and-deimos/)). Their CSS illustrations therefore use different sizes, textures, directions, and orbital rates. NASA's Curiosity imagery confirms Earth and Venus appear as bright points through Martian dust ([NASA Earth and Venus from Mars](https://science.nasa.gov/photojournal/curiosity-rover-finds-earth-and-venus-in-the-martian-sky/)); Earth can even be accompanied by its Moon as a separate point ([NASA bright evening star](https://science.nasa.gov/resource/bright-evening-star-seen-from-mars-is-earth-annotated/)). Earth, Venus, Jupiter, and Saturn are given compact recognizable illustrations to make the solar-system context legible, but their positions and apparent sizes are intentionally schematic—not an ephemeris or scientific sky simulator.

Rover POV is a level first-person perspective just above the rover. Discrete cell movement is interpolated horizontally with a fixed eye height, so elevation changes and snapshot updates cannot create vertical bob or pitch oscillation. Keyboard arrows are rover-relative in this view: Up is forward, Down is backward, Left is the current left side, and Right is the current right side. After a successful side move, the rover faces that travel direction and the entire arrow frame rotates with it, so Up immediately continues straight along the new heading. Horizontal pointer dragging also updates the relative input frame. Cardinal 3D Survey and Top controls remain world-relative. Quaternion interpolation turns the POV smoothly instead of snapping. Mouse wheel and ± adjust its 42-82 degree field of view. Fit from POV returns to fitted 3D Survey.

The complete navigation surface is one upper-right Environment cluster. It contains a three-button 3D Survey/Top/Rover POV viewpoint row, camera zoom, angle readout/reset and continuous rotation in 3D Survey, contextual guidance, a dedicated three-icon audio/cell-boundary/rover-visibility row, icon-only analytical layers, and the interactive mini navigation map while manually zoomed. No analytical layer is selected by default; clicking the active layer icon again returns to the unmodified terrain surface. The mini map displays the current visible rectangle and accepts pointer clicks to pan the zoomed camera.

The flashlight icon toggles a diagnostic preview of `aresim.obs.local.v1` in every camera view. When active, an 8×8 square around the rover remains illuminated while four dark volumetric masks suppress the rest of the terrain and structures. A fine pale perimeter and restrained rover-centered light make the boundary legible without adding a checkerboard. The preview follows the rover after every movement and exposes its active bounds through viewport diagnostics. It is off by default and changes no simulator state, observation content, action, reward, or replay. The active policy schema uses `self[10]`, categorical `pad_proximity`, no occupancy/entity tensors, and one flat masked `Discrete(10)` action head; remaining training steps or Sols are not UI or policy telemetry.

Camera state is UI-only.

### 5.2 Coordinate contract

Every simulator cell `(x, y)` maps to world position:

```text
worldX = x - width / 2 + 0.5
worldZ = y - height / 2 + 0.5
worldY = presentation elevation
```

Ray/pointer selection must resolve back to the simulator cell or stable entity id. Visual smoothing, height, shadows, or rotation must never change action targeting.

### 5.3 Terrain language

- Regolith: one stable rust/orange material and elevation token.
- Ridge: connected raised band with clearly visible height.
- Crater: recessed dark cells.
- Dune: warm raised drift.
- Ice: pale blue/white cells with crystalline props.
- Rock/ore: one stable dark basalt material and fixed low-poly marker form.
- Build pad: very light biscuit manufactured tiles distinct from both natural terrain and the dark world foundation; service-required state may override them with red.
- Extracted ice: a full-height standard regolith cell with an amber X and downward extraction pointer plus an amber analytical ring; extraction must never expose the brown world foundation or replace the regolith surface.

Terrain is rendered through GPU instances grouped by terrain category. Batches are recreated for each seed and whenever category membership changes, and they retain transition headroom for extracted cells. Randomization and extraction therefore cannot leave an obsolete ice instance, exceed a stale instance buffer, drop newly assigned regolith tiles, or expose the world foundation. Every terrain type has one stable base color, height, and marker design so policy observations do not acquire cosmetic sub-classes. Analytical overlays are the only intentional color modulation.

### 5.4 Entities

Rovers use a readable low-poly silhouette:

- chassis;
- wheels;
- solar/body surface;
- camera mast;
- grounded shadow;
- selection ring.

Structures expose:

- stable silhouettes by type;
- power/health status beacons;
- selection rings;
- deterministic position from their simulator coordinates.

The scene renders all entries in `rovers[]` and `structures[]`; it does not limit rendering to one entity.

### 5.5 Paths and markers

- The recorded action path renders as a cyan world-space line.
- The path contains valid movement entries only; scan, build, extraction, and service targets never create false travel segments.
- Each bright triangular arrowhead derives its quaternion independently from the preceding and current rover positions, so turns visibly point in their actual direction.
- Scanned cells use cyan rings.
- Extracted cells use amber rings.
- History selection can highlight its target cell.
- Markers are analytical presentation only.

### 5.6 Overlays

Current overlays:

- none;
- ice;
- ore;
- dust;
- roughness.

Overlays recolor terrain instances using normalized cell channels. They do not change terrain geometry or action rules.

### 5.7 Build pad state language

The build pad is a single integrated world-space logistics installation rather than a collection of generic structure props. A raised metal deck and panel seams establish the manufactured footprint. The installation contains:

- a marked rover docking lane;
- a utility gantry and status mast;
- a solar service rack;
- a progressively assembled habitat foundation, framing, and dome;
- a two-vessel ice tank farm with a cyan delivered-mass gauge;
- a sample-storage rack whose violet crates represent delivered 0.5 kg samples;
- permanent deck routes between the dock, tank farm, and sample vault.

Its operating states are:

- Normal: restrained amber perimeter and status mast.
- Building: foundations, uprights, and habitat framing progressively increase the installation silhouette.
- Habitat built: completed habitat dome and green status treatment.
- Service needed: after the canonical severe-dust, health, or sustained power-stress threshold is crossed, the deck, solar rack, perimeter, pulsing alarm ring, vertical alarm beam, and status beacon switch to persistent red warning treatment. Moderate wear does not activate this state.
- Serviced: cyan status treatment when at least one service action has completed and no warning remains.

The warning remains latched until Service succeeds. Exact trigger values and the slower infrastructure-degradation rates are defined in `environment_rules.md`. These cues sit in the 3D world and remain visible without opening a drawer.

Unload uses destination-specific motion rather than a generic success pulse. Cyan particles and a transfer beam travel from the rover dock to the ice tanks; violet packets and a second beam travel to the sample vault. A short elevated status label reports the exact transferred masses. The persistent tank and vault labels display delivered totals when the camera is manually zoomed, so the depot reads clearly without cluttering the fitted overview.

Successful actions produce a short world-space pulse and beam at their target. Move, scan, extract, build, service, unload, wait, invalid actions, and service requests have distinct synthesized Web Audio cues. Unload uses a rising transfer confirmation. The same history-driven feedback path is used by manual play, algorithm autoplay, and loaded replay. Audio is opt-out through the mute button in the upper-right Environment cluster and no external audio files are required. Browsers may require the first pointer interaction before sound can start.

## 6. Floating HUD

### 6.0 Grouped status header

The top status ribbon separates metrics by system ownership:

- Rover: Battery, Health, and Storage.
- Build pad: Power, Habitat, and Livability.

Every metric uses the same compact two-line alignment: `icon + short label` on the first row and the value on the second row, aligned beneath the label. The group names sit inline at the start of their respective groups. Labels remain deliberately short: Battery, Health, Storage, Power, Habitat, and Livability.

Storage is a live `used / 12 kg` value whose accessible label breaks the load into ice, geological samples, and ore. It changes to warning treatment at capacity. Keeping it beside Battery and Health makes the payload constraint visible in every mode without consuming command-bar space. Build-pad production and colony metrics remain visually grouped but distinct from the rover.

### 6.1 Mission HUD

The upper-left mission strip is 268 px wide and shows:

- compact objective label/icon;
- mission title;
- cumulative reward;
- a two-pixel combined progress line.

Clicking opens the Mission & Rewards drawer. Detailed objective cards are not permanently visible.

### 6.2 Alert HUD

The alert strip directly below the mission strip uses the same width, radius, surface, spacing, and typography system. It shows:

- the highest-priority current alert;
- alert count;
- severity color.

Clicking opens the Colony Systems drawer. System metrics such as generation, load, wait charge, water chain, and pressure remain inside that drawer.

### 6.3 Selection HUD

Clicking terrain, a rover, a structure, or a history row pins a selection. A narrow lower-right context launcher shows identity plus two critical facts without presenting another large tablet over the world. Clicking it opens the Context Inspector drawer.

Hover information remains transient and must not replace the pinned inspector.

## 7. Context Drawers

Drawers overlay the right side of the world and do not reserve permanent width.

Available drawers:

- Mission & Rewards.
- Colony Systems/Alerts.
- Context Inspector.
- Run Timeline/Agent History.

Rules:

- Only one drawer is open at a time.
- Clicking the backdrop or Close Drawer dismisses it.
- Drawers retain bounded internal scrolling.
- Opening a drawer must not alter simulation state.
- Arrow-key rover movement is disabled while a drawer or guide is open.

### 7.1 Mission & Rewards

Contains total reward, step count, objective progress, category rewards, and penalties. Clicking Total Reward opens analytics.

### 7.2 Context Inspector

The inspector is a single selection-aware telemetry view, not a tabbed tablet. Its header distinguishes a pinned target, a live hover, and the empty ready state. Cell details show terrain, ice, ore, dust, scan state, and roughness; rover details show battery, health, ice, samples, ore, and total payload against capacity; structure details use stable ids. The empty state gives short instructions instead of fake placeholder metrics. Analytical layers belong only to the upper-right Environment controls and are not duplicated here.

### 7.3 Run Timeline

Contains:

- newest/oldest ordering;
- action filter;
- elapsed steps;
- actor/action/result/reward rows;
- reward-term and event summaries;
- target highlighting.

## 8. Mode and Gameplay Controls

The Action Bar is a reserved 60 px footer below the world, so it never obscures environment geometry. Its opaque carbon rail uses compact command slots and three individually bordered, subtly tinted zones rather than a floating card. Each zone has a persistent micro-label and mode-colored status dot so ownership is clear without adding large headings.

- Left: mode selection plus Algorithm policy/parameter configuration or Replay file source.
- Center: Manual action controls, Algorithm transport/speed, or Replay transport/timeline.
- Right: seed, apply, and randomize controls for Manual and Algorithm. In Replay, the `World setup` zone label remains for stable spatial grouping, but the zone contains no environment controls because the file is authoritative.

The mode row contains:

- three mode buttons: cyan Manual, violet Algorithm, and amber Replay, using compact 10 px sentence-case labels, normal tracking, and a strong active baseline;
- mode names remain visible on wide desktop layouts and collapse to accessible icons below 1100 px so Algorithm configuration cannot intrude into the center transport zone.

Mode-specific setup follows these contracts:

- Manual: deterministic seed, icon-only apply and randomize, seven icon-only actions, reset, and save.
- Algorithm: policy selector, reserved parameter icon, deterministic seed, icon-only apply and randomize, play/pause, step, speed, reset, and save.
- Replay: icon-only upload, filename, play/pause, step, speed, repeat, cursor, and timeline. Seed controls and environment randomization are unavailable.

The reserved parameters control is intentionally disabled and labeled as forthcoming. It defines a stable location for policy-specific and agent-specific configuration without inventing parameters before their schemas exist.

All command buttons except mode names and essential data inputs are icon-only. Tooltips and accessible names carry their text labels. The rail uses normal compact text, with monospace reserved for seed, speed, filename, and replay position.

Manual actions:

- Move.
- Scan.
- Extract.
- Build.
- Service.
- Unload.
- Wait.

Payload belongs to the grouped Rover metrics in the header, not to the Action Bar. Full capacity changes the header meter to amber, while the upper-left alert strip surfaces the payload-rule warning. Scan and Extract remain selectable when full so validation can explain why they are blocked. Unload is explicit: it succeeds only on the build pad with non-empty cargo and is never triggered by entering the pad. Each Scan adds 0.5 kg and each Extract adds 2 kg.

Agent and replay transport controls:

- Start.
- Pause/resume.
- Step.
- Speed.
- Repeat/restart.
- Save for live Algorithm runs.
- Replay cursor and scrubber.

Actions dispatch immediately. There is no separate Execute button. Invalid actions produce a visible warning with a reason.

## 9. Warnings

Warnings appear as transient centered HUD toasts near the top of the world.

Kinds retain distinct treatments:

- blocked;
- terrain;
- system;
- progress;
- terminal.

Non-terminal warnings dismiss automatically. Terminal warnings remain visible.

## 10. Analytics and Guide

The existing Run Analytics dialog remains a separate focused workspace with Rewards, Resources, Behavior, Progress, and Environment tabs.

The guide remains a modal reference for:

- how to play;
- terrain;
- actions;
- rewards;
- rules and formulas.

Rewards and Rules use labeled equation cards: each card separates the formula from a short interpretation of its variables, conditions, or caps. Long per-action reward definitions use a responsive two-column card grid instead of inline code paragraphs.

Both surfaces are opened from the command header.

## 11. State Ownership

Simulator/backend owned:

- terrain and entity state;
- resources;
- rewards;
- actions and validity;
- history/events;
- terminal states;
- deterministic seed results.

UI owned:

- open drawer;
- guide/analytics visibility;
- selected/hovered target;
- active analytical layer, with `none` as the default;
- cell-boundary visibility, hidden by default;
- rover-visibility preview, hidden by default; this visualizes but does not define the always-local 8×8 policy observation;
- audio mute;
- camera zoom mode and target;
- continuous camera angle;
- active 3D Survey/Top/Rover viewpoint, Rover look yaw, and Rover POV field of view;
- active Manual/Algorithm/Replay mode and Algorithm policy selection;
- live/replay playback, pause, and speed controls;
- history filter/order;
- local warning visibility.

These presentation preferences are not required in `aresim.trajectory.episode.v1`. Loading a trajectory initializes the default fitted 3D Survey and clears selection, while later replay steps preserve any viewpoint and zoom the viewer chooses.

The renderer consumes a `WorldPresentation` produced by `snapshotToWorld`. It does not call the backend or mutate snapshots.

## 12. Responsive Behavior

Desktop:

- full header clusters;
- 268 px mission/alert strips;
- right drawer up to 390 px;
- one reserved 60 px footer command rail below the world.

Tablet:

- secondary header metrics collapse;
- drawers use up to 92% viewport width;
- mode labels collapse below 1100 px while icons and tooltips remain available;
- world remains visible behind contextual surfaces.

Small screens:

- identity text and noncritical metrics collapse;
- the context launcher may be hidden while the inspector remains reachable through selection and future compact navigation;
- landscape is preferred for manual play.

## 13. Accessibility

- Essential controls remain DOM elements outside the canvas.
- All icon buttons have accessible names.
- Tooltips do not replace accessible labels.
- Drawers expose a labeled close operation.
- Warning messages use `role="alert"`.
- Color is not the only status signal; labels and icons are retained.
- Keyboard movement ignores inputs and is disabled while modal/contextual surfaces are open.
- Reduced-motion support must be applied to future camera tweening and particle effects.

## 14. Performance Rules

- Terrain categories use GPU instancing.
- Repeated props remain low-poly.
- Device pixel ratio is capped.
- The default profile avoids postprocessing.
- Fog, shadows, props, and antialiasing must degrade cleanly under future quality profiles.
- Scene components consume the presentation adapter rather than subscribing independently to the complete backend client.

Current build size includes Three.js and analytics. Future work should lazy-load analytics and split Three.js vendor chunks.

## 15. Testing Expectations

Unit/component tests cover:

- snapshot-to-presentation mapping;
- deterministic terrain/elevation mapping;
- compact header and HUD presence;
- drawers hidden initially and opened contextually;
- modes/actions/save/load behavior;
- warnings and analytics.
- payload capacity, Scan/Extract blocking, explicit Unload, and legacy replay hydration;

Browser tests cover:

- visible nonblank WebGL canvas;
- full initial fit;
- camera zoom persistence, fit, and arbitrary-angle rotation;
- default angled 3D Survey plus a north-up, window-aligned square Top projection;
- local-time ambient phase selection, sun position, night stars, and background-only layering;
- mouse-wheel zoom in 3D Survey, Top, and Rover POV;
- Rover POV switching, field-of-view limits, rover-relative four-arrow remapping after every heading change, movement-aligned smooth yaw, level interpolated translation, Fit return, and transition overlay;
- mini-map click navigation, terrain-bound clamping, and rover-follow resumption after movement;
- integrated mini navigation;
- contextual mission/history/inspector flows;
- keyboard movement without camera zoom mutation;
- seamless terrain plus the optional cell-boundary toggle;
- icon-only analytical-layer toggling with no layer active by default;
- extracted-regolith replacement and stale GPU-instance removal;
- markers, build-pad state, replay, uploads, and analytics;
- live payload telemetry, build-pad-only unload feedback, and payload resource charts;
- desktop and tablet layouts, including zero Action Bar zone overflow.

## 16. Renderer and Packaging Boundary

The implemented renderer data flow is:

```text
Python REST session/replay service
              -> SimSnapshot
              -> snapshotToWorld
              -> WorldPresentation
              -> React Three Fiber scene
```

`WorldPresentation` contains dimensions, terrain, all rovers, all structures, build-pad state/progress/service history, and path points. The renderer does not call transports or mutate snapshots.

The intended future package split, after the interfaces stabilize, is:

```text
@aresim/protocol        snapshots, actions, events, rewards, capabilities
@aresim/client-core     future reusable sessions, REST transport, replay, selectors
@aresim/renderer-three  optional orthographic renderer
@aresim/ui-react        HUD, drawers, controls, timeline, analytics
@aresim/web             official composed application
```

Planned extension registries include entity renderers, overlays, action presentations, inspector sections, and policy diagnostics. Registry definitions require stable ids, compatible protocol versions, accessible labels, optional configuration schemas, and cleanup behavior. Unknown entity types must receive a generic presentation instead of crashing the scene.

## 17. Multi-Agent Readiness

Array position is not agent identity. Future agent state, actions, history, rewards, events, and replay records require stable `agent_id` values.

The training boundary uses one composed `AresEnvironment`. `AresGymEnv` exposes one rover to RLlib, while PettingZoo Parallel remains the future multi-rover API. No training-framework state enters the gameplay UI. The frontend consumes authoritative snapshots and optional policy diagnostics; training dashboards and checkpoint comparisons remain analysis artifacts defined in [RL Algorithms, Training, and Evaluation](../rl/rl_quickstart.md).

The scene already renders every rover and structure entry. Future multi-agent UI adds:

- roster search/filter;
- single and multi-select;
- follow and frame operations by stable id;
- deterministic color plus shape/icon distinction;
- level-of-detail label behavior;
- per-agent paths, observation areas, proposed actions, action masks, policy diagnostics, and reward contributions;
- history filtering by actor id, team, action, event, and reward term.

Canvas selection must always have an equivalent roster or history operation for keyboard users.

## 18. Documentation Update Checklist

When UI behavior changes, update:

- this document, which is the single current UI/renderer/extension design source;
- component and Playwright tests;
- package dependencies and browser requirements.
