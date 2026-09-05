/**
 * Orthographic 3D viewport: Survey, Top, and Rover POV.
 * Reads `snapshot` from the store and maps it through `snapshotToWorld`.
 * Clicks select targets; they do not apply actions until the Action Bar / keys do.
 */

import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { Html, Instance, Instances, Line, PerspectiveCamera as DreiPerspectiveCamera } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import { Color, Group, MathUtils, Matrix4, Quaternion, type OrthographicCamera, type PerspectiveCamera, Vector3 } from "three";
import { Compass, Eye, Flashlight, Gem, Grid3X3, LocateFixed, Map as MapIcon, Maximize, Minus, Mountain, PackageCheck, Plus, Snowflake, Volume2, VolumeX, Wind } from "lucide-react";
import { snapshotToWorld } from "../presentation/adapters/snapshotToWorld";
import type { TerrainPresentation, WorldPresentation } from "../presentation/types";
import { useAresStore } from "../state/useAresStore";
import type { CameraView, OverlayMode, SelectionTarget, TerrainType } from "../types/sim";
import { ambientTimeFor, type AmbientTimeState } from "../lib/ambientTime";
import { isInsideRoverObservation, roverObservationBounds, ROVER_OBSERVATION_SIZE } from "../lib/roverObservation";

const TERRAIN_COLORS: Record<TerrainType, string> = {
  regolith: "#a64f2d",
  rock: "#51413a",
  ice: "#9ccbd1",
  crater: "#2a1915",
  dune: "#c56a32",
  build_pad: "#dcc49a",
  ridge: "#6d4d3e",
};

const TERRAIN_ORDER: TerrainType[] = ["regolith", "dune", "ridge", "crater", "rock", "ice", "build_pad"];

/** Main 3D canvas. Requires `snapshot` to already be loaded in the store. */
export function GameViewport() {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const overlay = useAresStore((state) => state.overlayMode);
  const selectedTarget = useAresStore((state) => state.selectedTarget);
  const hoveredTarget = useAresStore((state) => state.hoveredTarget);
  const highlightedCell = useAresStore((state) => state.highlightedCell);
  const zoomMode = useAresStore((state) => state.viewportZoomMode);
  const zoomScale = useAresStore((state) => state.viewportZoomScale);
  const viewportCenter = useAresStore((state) => state.viewportCenter);
  const zoomIn = useAresStore((state) => state.zoomIn);
  const zoomOut = useAresStore((state) => state.zoomOut);
  const fitViewport = useAresStore((state) => state.fitViewport);
  const showGrid = useAresStore((state) => state.showGrid);
  const showRoverVisibility = useAresStore((state) => state.showRoverVisibility);
  const selectTarget = useAresStore((state) => state.selectTarget);
  const hoverTarget = useAresStore((state) => state.hoverTarget);
  const [rotation, setRotation] = useState(45);
  const cameraView = useAresStore((state) => state.cameraView);
  const setCameraView = useAresStore((state) => state.setCameraView);
  const [roverFov, setRoverFov] = useState(62);
  const roverYaw = useAresStore((state) => state.roverCameraYaw);
  const setRoverYaw = useAresStore((state) => state.setRoverCameraYaw);
  const [viewTransition, setViewTransition] = useState(0);
  const previousSessionId = useRef(snapshot.sessionId);
  const previousRover = useRef<{ seed: number; x: number; y: number } | null>(null);
  const lastWheelAt = useRef(0);
  const lookDrag = useRef<{ pointerId: number; x: number } | null>(null);

  const world = useMemo(() => snapshotToWorld(snapshot), [snapshot]);
  const ambientTime = useMemo(() => ambientTimeFor(snapshot.localTime, snapshot.sol), [snapshot.localTime, snapshot.sol]);
  const selectedCell = selectedTarget?.kind === "cell" ? selectedTarget : null;
  const targetCell = highlightedCell ?? selectedCell;
  const rover = world.rovers[0];
  const scannedMarkerCount = world.terrain.filter((cell) => cell.scanned).length;
  const extractedMarkerCount = world.terrain.filter((cell) => cell.extracted).length;
  const pathArrowCount = Math.max(0, Math.ceil((world.path.length - 1) / 3));
  const roverCenter = rover ? { x: rover.x + 0.5, y: rover.y + 0.5 } : null;
  const observationBounds = rover ? roverObservationBounds(rover) : null;
  const roverHeading = useMemo(() => ({ x: Math.sin(roverYaw), z: -Math.cos(roverYaw) }), [roverYaw]);
  const worldCenter = { x: world.dimensions.width / 2, y: world.dimensions.height / 2 };
  const visibleWidth = zoomMode === "fit" ? world.dimensions.width : Math.min(world.dimensions.width, world.dimensions.width / zoomScale);
  const visibleHeight = zoomMode === "fit" ? world.dimensions.height : Math.min(world.dimensions.height, world.dimensions.height / zoomScale);
  const requestedCenter = cameraView === "rover" && roverCenter ? roverCenter : zoomMode === "manual" ? viewportCenter ?? roverCenter ?? worldCenter : worldCenter;
  const center = cameraView !== "rover" && zoomMode === "manual" ? {
    x: Math.max(visibleWidth / 2, Math.min(world.dimensions.width - visibleWidth / 2, requestedCenter.x)),
    y: Math.max(visibleHeight / 2, Math.min(world.dimensions.height - visibleHeight / 2, requestedCenter.y)),
  } : requestedCenter;
  const visibleX = zoomMode === "fit" ? 0 : Math.max(0, Math.min(world.dimensions.width - visibleWidth, center.x - visibleWidth / 2));
  const visibleY = zoomMode === "fit" ? 0 : Math.max(0, Math.min(world.dimensions.height - visibleHeight, center.y - visibleHeight / 2));
  const visibleRectData = `${visibleX.toFixed(2)},${visibleY.toFixed(2)},${visibleWidth.toFixed(2)},${visibleHeight.toFixed(2)}`;

  useEffect(() => {
    if (previousSessionId.current === snapshot.sessionId) return;
    previousSessionId.current = snapshot.sessionId;
    setCameraView("survey");
    setRoverFov(62);
    setRoverYaw(0);
  }, [snapshot.sessionId]);

  useEffect(() => {
    if (!rover) return;
    const previous = previousRover.current;
    previousRover.current = { seed: world.seed, x: rover.x, y: rover.y };
    if (!previous || previous.seed !== world.seed) return;
    const dx = rover.x - previous.x;
    const dz = rover.y - previous.y;
    if (dx || dz) setRoverYaw(Math.atan2(dx, -dz));
  }, [rover?.x, rover?.y, world.seed]);

  const changeCameraView = (view: CameraView) => {
    if (view === cameraView) return;
    setCameraView(view);
    setViewTransition((value) => value + 1);
  };
  const cameraZoomIn = () => cameraView !== "rover" ? zoomIn() : setRoverFov((value) => Math.max(42, value - 4));
  const cameraZoomOut = () => cameraView !== "rover" ? zoomOut() : setRoverFov((value) => Math.min(82, value + 4));
  const cameraFit = () => {
    if (cameraView === "rover") changeCameraView("survey");
    fitViewport();
  };
  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest("button, input, select, textarea, [role='dialog']")) return;
    event.preventDefault();
    const now = Date.now();
    if (now - lastWheelAt.current < 80 || Math.abs(event.deltaY) < 2) return;
    lastWheelAt.current = now;
    if (event.deltaY < 0) cameraZoomIn();
    else cameraZoomOut();
  };
  const startRoverLook = (event: React.PointerEvent<HTMLDivElement>) => {
    if (cameraView !== "rover" || event.button !== 0 || (event.target as HTMLElement).closest("button, input, select, textarea, [role='dialog']")) return;
    lookDrag.current = { pointerId: event.pointerId, x: event.clientX };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const moveRoverLook = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = lookDrag.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - drag.x;
    drag.x = event.clientX;
    setRoverYaw((value) => value - deltaX * 0.007);
  };
  const stopRoverLook = (event: React.PointerEvent<HTMLDivElement>) => {
    if (lookDrag.current?.pointerId !== event.pointerId) return;
    lookDrag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  return (
    <div
      className="relative h-full w-full overflow-hidden bg-[#160906]"
      onWheel={handleWheel}
      onPointerDown={startRoverLook}
      onPointerMove={moveRoverLook}
      onPointerUp={stopRoverLook}
      onPointerCancel={stopRoverLook}
      data-build-pad-status={snapshot.buildPadState.status}
      data-build-progress={snapshot.objectiveStats.habitatBuildProgress}
      data-service-count={snapshot.objectiveStats.serviceCount}
      data-camera-center={`${center.x.toFixed(2)},${center.y.toFixed(2)}`}
      data-camera-rotation={rotation}
      data-camera-view={cameraView}
      data-rover-heading={`${roverHeading.x.toFixed(2)},${roverHeading.z.toFixed(2)}`}
      data-rover-fov={roverFov}
      data-rover-motion="level-smoothed"
      data-camera-projection={cameraView === "survey" ? "angled-orthographic-3d" : cameraView === "top" ? "window-aligned-top" : "first-person-perspective"}
      data-survey-projection={cameraView === "survey" ? "angled-orthographic-3d" : undefined}
      data-top-projection={cameraView === "top" ? "window-aligned-square" : undefined}
      data-grid-visible={showGrid}
      data-observation-preview={showRoverVisibility ? "local-8" : "off"}
      data-observation-window={observationBounds ? `${observationBounds.minX},${observationBounds.minY},${ROVER_OBSERVATION_SIZE},${ROVER_OBSERVATION_SIZE}` : undefined}
      data-extracted-markers={extractedMarkerCount}
      data-path-arrows={pathArrowCount}
      data-scanned-markers={scannedMarkerCount}
      data-testid="game-viewport"
      data-visible-rect={visibleRectData}
    >
      <AmbientTimeBackdrop localTime={snapshot.localTime} sol={snapshot.sol} state={ambientTime} />
      <Canvas
        className="relative z-10"
        dpr={[1, 1.75]}
        orthographic
        shadows
        camera={{ position: [26, 28, 26], near: 0.1, far: 160, zoom: 18 }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      >
        <fog attach="fog" args={[showRoverVisibility ? "#050609" : "#35130c", 58, 112]} />
        <ambientLight intensity={showRoverVisibility ? 0.3 : 2.15} color="#ffcfad" />
        <hemisphereLight args={["#e8aa7c", "#24100c", showRoverVisibility ? 0.22 : 1.55]} />
        <directionalLight
          castShadow
          color="#ffd2a4"
          intensity={showRoverVisibility ? 0.25 : 3.1}
          position={[-18, 34, 12]}
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-far={90}
          shadow-camera-left={-28}
          shadow-camera-right={28}
          shadow-camera-top={28}
          shadow-camera-bottom={-28}
        />
        {cameraView === "survey" ? <SurveyCameraRig dimensions={world.dimensions} focus={center} rotation={rotation} /> : null}
        {cameraView === "top" ? <TopCameraRig dimensions={world.dimensions} focus={center} /> : null}
        {cameraView === "rover" && rover ? <RoverCameraRig fov={roverFov} heading={roverHeading} rover={rover} world={world} /> : null}
        <MarsWorld
          highlightedCell={targetCell}
          hideRovers={cameraView === "rover"}
          overlay={overlay}
          showGrid={showGrid}
          showRoverVisibility={showRoverVisibility}
          selectedTarget={selectedTarget}
          world={world}
          onHover={hoverTarget}
          onSelect={selectTarget}
        />
      </Canvas>

      <WorldNavigation
        cameraView={cameraView}
        changeCameraView={changeCameraView}
        center={center}
        fitViewport={cameraFit}
        height={world.dimensions.height}
        rotation={rotation}
        rover={rover ? { x: rover.x, y: rover.y } : null}
        setRotation={setRotation}
        roverFov={roverFov}
        terrain={world.terrain}
        visibleHeight={visibleHeight}
        visibleWidth={visibleWidth}
        width={world.dimensions.width}
        zoomIn={cameraZoomIn}
        zoomMode={zoomMode}
        zoomOut={cameraZoomOut}
        zoomScale={zoomScale}
      />

      {viewTransition ? <div key={viewTransition} className="camera-view-transition pointer-events-none absolute inset-0 z-10" aria-hidden="true" /> : null}

      <HoverReadout target={hoveredTarget} world={world} />
      <div className="pointer-events-none absolute bottom-4 right-4 rounded-full border border-white/10 bg-black/40 px-2.5 py-1 text-[9px] uppercase tracking-[0.18em] text-stone-400 backdrop-blur">
        WebGL · Balanced
      </div>
    </div>
  );
}

function AmbientTimeBackdrop({ localTime, sol, state }: { localTime: string; sol: number; state: AmbientTimeState }) {
  const style = {
    "--ambient-sky-top": state.skyTop,
    "--ambient-sky-horizon": state.skyHorizon,
    "--ambient-stars-opacity": state.stars.toFixed(3),
    "--ambient-sun-x": `${state.sunX.toFixed(2)}%`,
    "--ambient-sun-y": `${state.sunY.toFixed(2)}%`,
    "--ambient-sun-opacity": state.sunOpacity.toFixed(3),
    "--ambient-night-objects": state.nightObjects.toFixed(3),
    "--phobos-x": `${state.phobosX.toFixed(2)}%`,
    "--phobos-y": `${state.phobosY.toFixed(2)}%`,
    "--deimos-x": `${state.deimosX.toFixed(2)}%`,
    "--deimos-y": `${state.deimosY.toFixed(2)}%`,
    "--planet-drift": `${state.planetDrift.toFixed(2)}%`,
  } as React.CSSProperties;

  return (
    <div
      aria-hidden="true"
      className="ambient-time-sky pointer-events-none absolute inset-0 z-0 overflow-hidden"
      data-local-time={localTime}
      data-night-objects={state.nightObjects.toFixed(3)}
      data-sol={sol}
      data-solar-progress={(state.sunX / 100).toFixed(3)}
      data-time-phase={state.phase}
      data-testid="ambient-time-sky"
      style={style}
    >
      <div className="ambient-stars absolute inset-0" />
      <div className="ambient-sun absolute" />
      <div className="ambient-night-objects absolute inset-0">
        <div className="ambient-moon ambient-phobos absolute" data-celestial-object="Phobos"><i /><span>Phobos</span></div>
        <div className="ambient-moon ambient-deimos absolute" data-celestial-object="Deimos"><i /><span>Deimos</span></div>
        <div className="ambient-planet ambient-earth absolute" data-celestial-object="Earth"><i /><span>Earth</span></div>
        <div className="ambient-planet ambient-venus absolute" data-celestial-object="Venus"><i /><span>Venus</span></div>
        <div className="ambient-planet ambient-jupiter absolute" data-celestial-object="Jupiter"><i /><span>Jupiter</span></div>
        <div className="ambient-planet ambient-saturn absolute" data-celestial-object="Saturn"><i /><span>Saturn</span></div>
      </div>
      <div className="ambient-horizon absolute inset-x-0 bottom-0 h-1/2" />
    </div>
  );
}

function SurveyCameraRig({ dimensions, rotation, focus }: { dimensions: WorldPresentation["dimensions"]; rotation: number; focus: { x: number; y: number } }) {
  const { camera, size } = useThree();
  const zoomMode = useAresStore((state) => state.viewportZoomMode);
  const zoomScale = useAresStore((state) => state.viewportZoomScale);
  const targetX = focus.x - dimensions.width / 2;
  const targetZ = focus.y - dimensions.height / 2;
  const desired = useMemo(() => {
    const radians = MathUtils.degToRad(rotation);
    const projectedWidth = Math.abs(Math.cos(radians)) * dimensions.width + Math.abs(Math.sin(radians)) * dimensions.height;
    const projectedHeight = Math.abs(Math.sin(radians)) * dimensions.width + Math.abs(Math.cos(radians)) * dimensions.height;
    const fitZoom = Math.max(7, Math.min(size.width / (projectedWidth + 7), size.height / (projectedHeight * 0.72 + 15)) * 0.88);
    return { radians, zoom: fitZoom * (zoomMode === "manual" ? zoomScale : 1) };
  }, [dimensions.height, dimensions.width, rotation, size.height, size.width, zoomMode, zoomScale]);

  useFrame(() => {
    const orthographic = camera as OrthographicCamera;
    orthographic.position.set(targetX + Math.cos(desired.radians) * 44, 42, targetZ + Math.sin(desired.radians) * 44);
    orthographic.up.set(0, 1, 0);
    orthographic.lookAt(targetX, 0, targetZ);
    if (Math.abs(orthographic.zoom - desired.zoom) > 0.001) {
      orthographic.zoom = desired.zoom;
      orthographic.updateProjectionMatrix();
    }
  });

  return null;
}

function TopCameraRig({ dimensions, focus }: { dimensions: WorldPresentation["dimensions"]; focus: { x: number; y: number } }) {
  const { camera, size } = useThree();
  const zoomMode = useAresStore((state) => state.viewportZoomMode);
  const zoomScale = useAresStore((state) => state.viewportZoomScale);
  const targetX = focus.x - dimensions.width / 2;
  const targetZ = focus.y - dimensions.height / 2;
  const fitZoom = useMemo(() => Math.max(7, Math.min(size.width / (dimensions.width + 5), size.height / (dimensions.height + 5)) * 0.88), [dimensions.height, dimensions.width, size.height, size.width]);

  useFrame(() => {
    const orthographic = camera as OrthographicCamera;
    orthographic.position.set(targetX, 52, targetZ);
    orthographic.up.set(0, 0, -1);
    orthographic.lookAt(targetX, 0, targetZ);
    const desiredZoom = fitZoom * (zoomMode === "manual" ? zoomScale : 1);
    if (Math.abs(orthographic.zoom - desiredZoom) > 0.001) {
      orthographic.zoom = desiredZoom;
      orthographic.updateProjectionMatrix();
    }
  });

  return null;
}

function RoverCameraRig({ fov, heading, rover, world }: {
  fov: number;
  heading: { x: number; z: number };
  rover: WorldPresentation["rovers"][number];
  world: WorldPresentation;
}) {
  const cameraRef = useRef<PerspectiveCamera>(null);
  const [x, z] = cellToWorld(rover.x, rover.y, world.dimensions);
  const eyeHeight = 1.78;
  const desiredPosition = useMemo(() => new Vector3(x, eyeHeight, z), [x, z]);
  const desiredTarget = useRef(new Vector3());
  const lookMatrix = useRef(new Matrix4());
  const targetQuaternion = useRef(new Quaternion());
  const initialized = useRef(false);

  useFrame((_, delta) => {
    const active = cameraRef.current;
    if (!active) return;
    if (!initialized.current) {
      active.position.copy(desiredPosition);
      active.fov = fov;
      active.updateProjectionMatrix();
      desiredTarget.current.set(active.position.x + heading.x * 10, eyeHeight - 0.08, active.position.z + heading.z * 10);
      active.lookAt(desiredTarget.current);
      initialized.current = true;
    }
    active.position.x = MathUtils.damp(active.position.x, desiredPosition.x, 8.5, delta);
    active.position.y = eyeHeight;
    active.position.z = MathUtils.damp(active.position.z, desiredPosition.z, 8.5, delta);
    desiredTarget.current.set(active.position.x + heading.x * 10, eyeHeight - 0.08, active.position.z + heading.z * 10);
    lookMatrix.current.lookAt(active.position, desiredTarget.current, active.up);
    targetQuaternion.current.setFromRotationMatrix(lookMatrix.current);
    active.quaternion.slerp(targetQuaternion.current, 1 - Math.exp(-11 * delta));
    const nextFov = MathUtils.damp(active.fov, fov, 8, delta);
    if (Math.abs(nextFov - active.fov) > 0.001) {
      active.fov = nextFov;
      active.updateProjectionMatrix();
    }
  });

  return <DreiPerspectiveCamera ref={cameraRef} makeDefault near={0.04} far={150} />;
}

function MarsWorld({
  world,
  overlay,
  selectedTarget,
  highlightedCell,
  hideRovers,
  showGrid,
  showRoverVisibility,
  onSelect,
  onHover,
}: {
  world: WorldPresentation;
  overlay: OverlayMode;
  selectedTarget: SelectionTarget | null;
  highlightedCell: { x: number; y: number } | null;
  hideRovers: boolean;
  showGrid: boolean;
  showRoverVisibility: boolean;
  onSelect: (target: SelectionTarget | null) => void;
  onHover: (target: SelectionTarget | null) => void;
}) {
  const grouped = useMemo(() => {
    const result = new Map<TerrainType, TerrainPresentation[]>();
    TERRAIN_ORDER.forEach((terrain) => result.set(terrain, []));
    world.terrain.forEach((cell) => result.get(cell.terrain)!.push(cell));
    return result;
  }, [world.terrain]);

  return (
    <group>
      <mesh position={[0, -0.43, 0]} receiveShadow>
        <boxGeometry args={[world.dimensions.width + 2.2, 0.7, world.dimensions.height + 2.2]} />
        <meshStandardMaterial color="#32140e" roughness={1} />
      </mesh>
      <mesh position={[0, -0.78, 0]} receiveShadow>
        <boxGeometry args={[world.dimensions.width + 2.8, 0.12, world.dimensions.height + 2.8]} />
        <meshStandardMaterial color="#0d0909" roughness={1} />
      </mesh>

      {TERRAIN_ORDER.map((terrain) => (
        <TerrainBatch
          key={`${world.seed}-${terrain}-${grouped.get(terrain)?.length ?? 0}`}
          cells={grouped.get(terrain) ?? []}
          dimensions={world.dimensions}
          overlay={overlay}
          showGrid={showGrid}
          buildPadStatus={world.buildPadStatus}
          onHover={onHover}
          onSelect={onSelect}
          terrain={terrain}
        />
      ))}

      <TerrainDecorations world={world} />
      <WorldMarkers world={world} />
      <PathTrail world={world} />
      <BuildPadInstallation world={world} />
      <ActionPulse key={world.lastAction?.id ?? "no-action"} world={world} />
      {highlightedCell ? <SelectionMarker cell={highlightedCell} world={world} highlighted={!!useAresStore.getState().highlightedCell} /> : null}

      {!hideRovers ? world.rovers.map((rover) => (
        <Rover key={rover.id} rover={rover} world={world} selected={selectedTarget?.kind === "rover" && selectedTarget.id === rover.id} onSelect={onSelect} onHover={onHover} />
      )) : null}
      {world.structures.filter((structure) => terrainAt(world, structure.x, structure.y)?.terrain !== "build_pad").map((structure) => (
        <Structure key={structure.id} structure={structure} world={world} selected={selectedTarget?.kind === "structure" && selectedTarget.id === structure.id} onSelect={onSelect} onHover={onHover} />
      ))}
      {showRoverVisibility && world.rovers[0] ? <RoverVisibilityMask rover={world.rovers[0]} world={world} /> : null}
    </group>
  );
}

function RoverVisibilityMask({ rover, world }: { rover: WorldPresentation["rovers"][number]; world: WorldPresentation }) {
  const bounds = roverObservationBounds(rover);
  const worldLeft = -world.dimensions.width / 2;
  const worldRight = world.dimensions.width / 2;
  const worldTop = -world.dimensions.height / 2;
  const worldBottom = world.dimensions.height / 2;
  const openingLeft = Math.max(worldLeft, bounds.minX - world.dimensions.width / 2);
  const openingRight = Math.min(worldRight, bounds.maxXExclusive - world.dimensions.width / 2);
  const openingTop = Math.max(worldTop, bounds.minY - world.dimensions.height / 2);
  const openingBottom = Math.min(worldBottom, bounds.maxYExclusive - world.dimensions.height / 2);
  const [roverX, roverZ] = cellToWorld(rover.x, rover.y, world.dimensions);
  const hiddenCells = world.terrain.filter((cell) => !isInsideRoverObservation(cell.x, cell.y, rover));
  const perimeter: Array<[number, number, number]> = [
    [openingLeft, 0.34, openingTop],
    [openingRight, 0.34, openingTop],
    [openingRight, 0.34, openingBottom],
    [openingLeft, 0.34, openingBottom],
    [openingLeft, 0.34, openingTop],
  ];

  return (
    <group name="rover-local-observation-mask">
      <Instances limit={hiddenCells.length + 8} renderOrder={25}>
        <boxGeometry args={[1.018, 0.035, 1.018]} />
        <meshBasicMaterial color="#020306" depthWrite={false} opacity={0.78} transparent />
        {hiddenCells.map((cell) => {
          const [x, z] = cellToWorld(cell.x, cell.y, world.dimensions);
          return <Instance key={`hidden-${cell.id}`} position={[x, cell.elevation + 0.31, z]} />;
        })}
      </Instances>
      <Line color="#d9f6ff" lineWidth={1.6} opacity={0.68} points={perimeter} transparent />
      <pointLight color="#d8f5ff" decay={1.7} distance={7.2} intensity={7.5} position={[roverX, 4.2, roverZ]} />
    </group>
  );
}

function TerrainBatch({ cells, dimensions, terrain, overlay, showGrid, buildPadStatus, onSelect, onHover }: {
  cells: TerrainPresentation[];
  dimensions: WorldPresentation["dimensions"];
  terrain: TerrainType;
  overlay: OverlayMode;
  showGrid: boolean;
  buildPadStatus: WorldPresentation["buildPadStatus"];
  onSelect: (target: SelectionTarget) => void;
  onHover: (target: SelectionTarget | null) => void;
}) {
  return (
    <Instances limit={Math.max(1, Math.min(dimensions.width * dimensions.height, cells.length + 64))} castShadow={terrain === "ridge" || terrain === "rock" || terrain === "build_pad"} receiveShadow>
      <boxGeometry args={[1, 1, 1]} />
      <meshLambertMaterial color="#ffffff" />
      {cells.map((cell) => {
        const height = Math.max(0.08, cell.elevation + 0.28);
        const [x, z] = cellToWorld(cell.x, cell.y, dimensions);
        const target = { kind: "cell", x: cell.x, y: cell.y } as const;
        return (
          <Instance
            key={cell.id}
            color={colorForCell(cell, overlay, buildPadStatus)}
            position={[x, terrain === "crater" ? -0.18 : height / 2 - 0.04, z]}
            scale={[showGrid ? 0.96 : 1.012, height, showGrid ? 0.96 : 1.012]}
            onClick={(event: ThreeEvent<MouseEvent>) => { event.stopPropagation(); onSelect(target); }}
            onPointerMove={(event: ThreeEvent<PointerEvent>) => { event.stopPropagation(); onHover(target); }}
            onPointerOut={() => onHover(null)}
          />
        );
      })}
    </Instances>
  );
}

function TerrainDecorations({ world }: { world: WorldPresentation }) {
  const decorated = world.terrain.filter((cell) => cell.extracted || cell.terrain === "rock" || cell.terrain === "ice");
  return (
    <group>
      {decorated.map((cell) => {
        const [x, z] = cellToWorld(cell.x, cell.y, world.dimensions);
        if (cell.extracted) {
          return (
            <group key={cell.id} position={[x, cell.elevation + 0.21, z]}>
              <mesh receiveShadow position={[0, -0.03, 0]} scale={[1.012, 0.16, 1.012]}>
                <boxGeometry />
                <meshStandardMaterial color={TERRAIN_COLORS.regolith} roughness={1} />
              </mesh>
              <mesh position={[0, 0.075, 0]} rotation={[0, Math.PI / 4, 0]} scale={[0.52, 0.055, 0.09]}>
                <boxGeometry />
                <meshBasicMaterial color="#fbbf24" />
              </mesh>
              <mesh position={[0, 0.08, 0]} rotation={[0, -Math.PI / 4, 0]} scale={[0.52, 0.055, 0.09]}>
                <boxGeometry />
                <meshBasicMaterial color="#fbbf24" />
              </mesh>
              <mesh position={[0, 0.42, 0]} rotation={[0, 0, Math.PI]}>
                <coneGeometry args={[0.13, 0.32, 4]} />
                <meshBasicMaterial color="#fff1b8" />
              </mesh>
            </group>
          );
        }
        if (cell.terrain === "ice") {
          return (
            <mesh key={cell.id} castShadow position={[x, cell.elevation + 0.45, z]} rotation={[0.1, 0.35, -0.12]} scale={[0.22, 0.52, 0.22]}>
              <octahedronGeometry args={[0.62, 0]} />
              <meshStandardMaterial color="#b8f0f3" emissive="#4aa7b0" emissiveIntensity={0.18} roughness={0.42} />
            </mesh>
          );
        }
        return (
          <mesh key={cell.id} castShadow position={[x, cell.elevation + 0.25, z]} rotation={[0.18, 0.45, 0.1]} scale={[0.28, 0.3, 0.3]}>
            <dodecahedronGeometry args={[0.7, 0]} />
            <meshStandardMaterial color={cell.terrain === "rock" ? "#2e2827" : "#6b3425"} roughness={1} />
          </mesh>
        );
      })}
    </group>
  );
}

function Rover({ rover, world, selected, onSelect, onHover }: {
  rover: WorldPresentation["rovers"][number];
  world: WorldPresentation;
  selected: boolean;
  onSelect: (target: SelectionTarget) => void;
  onHover: (target: SelectionTarget | null) => void;
}) {
  const [x, z] = cellToWorld(rover.x, rover.y, world.dimensions);
  const elevation = terrainAt(world, rover.x, rover.y)?.elevation ?? 0.1;
  const target = { kind: "rover", id: rover.id } as const;
  return (
    <group
      position={[x, elevation + 0.54, z]}
      rotation={[0, -0.55, 0]}
      onClick={(event) => { event.stopPropagation(); onSelect(target); }}
      onPointerMove={(event) => { event.stopPropagation(); onHover(target); }}
      onPointerOut={() => onHover(null)}
    >
      {selected ? <mesh position={[0, -0.45, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.66, 0.78, 32]} /><meshBasicMaterial color="#67e8f9" transparent opacity={0.85} /></mesh> : null}
      <mesh castShadow scale={[0.9, 0.3, 0.58]}><boxGeometry /><meshStandardMaterial color="#d9e2e5" metalness={0.48} roughness={0.34} /></mesh>
      <mesh castShadow position={[0, 0.24, 0]} scale={[0.5, 0.08, 0.42]}><boxGeometry /><meshStandardMaterial color="#245b73" metalness={0.55} roughness={0.28} /></mesh>
      <mesh castShadow position={[0.14, 0.55, 0]} scale={[0.06, 0.55, 0.06]}><cylinderGeometry args={[0.5, 0.65, 1, 8]} /><meshStandardMaterial color="#bbc8cc" metalness={0.52} /></mesh>
      <mesh castShadow position={[0.15, 0.82, 0]} scale={[0.14, 0.1, 0.14]}><boxGeometry /><meshStandardMaterial color="#111b20" emissive="#55d9ff" emissiveIntensity={0.18} /></mesh>
      {[-0.34, 0.34].flatMap((wheelX) => [-0.36, 0.36].map((wheelZ) => (
        <mesh key={`${wheelX}-${wheelZ}`} castShadow position={[wheelX, -0.19, wheelZ]} rotation={[Math.PI / 2, 0, 0]} scale={[0.16, 0.11, 0.16]}>
          <cylinderGeometry args={[1, 1, 1, 12]} /><meshStandardMaterial color="#181718" roughness={1} />
        </mesh>
      )))}
    </group>
  );
}

function Structure({ structure, world, selected, onSelect, onHover }: {
  structure: WorldPresentation["structures"][number];
  world: WorldPresentation;
  selected: boolean;
  onSelect: (target: SelectionTarget) => void;
  onHover: (target: SelectionTarget | null) => void;
}) {
  const [x, z] = cellToWorld(structure.x, structure.y, world.dimensions);
  const elevation = terrainAt(world, structure.x, structure.y)?.elevation ?? 0.1;
  const target = { kind: "structure", id: structure.id } as const;
  const color = structure.type === "solar" ? "#315c70" : structure.type === "habitat" ? "#d1c1aa" : "#9e806b";
  return (
    <group
      position={[x, elevation + 0.22, z]}
      onClick={(event) => { event.stopPropagation(); onSelect(target); }}
      onPointerMove={(event) => { event.stopPropagation(); onHover(target); }}
      onPointerOut={() => onHover(null)}
    >
      {selected ? <mesh position={[0, -0.12, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.62, 0.76, 32]} /><meshBasicMaterial color="#67e8f9" transparent opacity={0.85} /></mesh> : null}
      <mesh castShadow position={[0, 0.35, 0]} scale={[0.72, 0.7, 0.72]}>
        {structure.type === "habitat" ? <sphereGeometry args={[0.7, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2]} /> : <boxGeometry />}
        <meshStandardMaterial color={color} metalness={0.34} roughness={0.48} emissive={structure.powered ? "#174453" : "#3b0909"} emissiveIntensity={0.15} />
      </mesh>
      <mesh position={[0.31, 0.75, 0.32]} scale={0.08}><sphereGeometry /><meshBasicMaterial color={structure.powered ? "#6ee7b7" : "#fb7185"} /></mesh>
    </group>
  );
}

function WorldMarkers({ world }: { world: WorldPresentation }) {
  return (
    <group>
      {world.terrain.filter((cell) => cell.scanned || cell.extracted).map((cell) => {
        const [x, z] = cellToWorld(cell.x, cell.y, world.dimensions);
        return (
          <mesh key={`marker-${cell.id}`} position={[x, cell.elevation + 0.56, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.25, 0.34, 18]} />
            <meshBasicMaterial color={cell.extracted ? "#fbbf24" : "#67e8f9"} transparent opacity={0.85} depthWrite={false} />
          </mesh>
        );
      })}
    </group>
  );
}

function PathTrail({ world }: { world: WorldPresentation }) {
  if (world.path.length < 2) return null;
  const points = world.path.map((point) => {
    const [x, z] = cellToWorld(point.x, point.y, world.dimensions);
    return [x, (terrainAt(world, point.x, point.y)?.elevation ?? 0.1) + 0.62, z] as [number, number, number];
  });
  const arrows = points.slice(1).map((point, index) => {
    if (index % 2 !== 0) return null;
    const previous = points[index];
    return <PathArrow key={`path-arrow-${index}`} point={point} previous={previous} />;
  });
  return <group><Line points={points} color="#67e8f9" lineWidth={2} transparent opacity={0.78} />{arrows}</group>;
}

function PathArrow({ point, previous }: { point: [number, number, number]; previous: [number, number, number] }) {
  const quaternion = useMemo(() => {
    const direction = new Vector3(point[0] - previous[0], 0, point[2] - previous[2]).normalize();
    return new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), direction);
  }, [point, previous]);
  return (
    <mesh position={point} quaternion={quaternion}>
      <coneGeometry args={[0.14, 0.36, 3]} />
      <meshBasicMaterial color="#baf7ff" transparent opacity={0.95} depthWrite={false} />
    </mesh>
  );
}

function BuildPadInstallation({ world }: { world: WorldPresentation }) {
  const zoomMode = useAresStore((state) => state.viewportZoomMode);
  const zoomScale = useAresStore((state) => state.viewportZoomScale);
  const pad = world.terrain.filter((cell) => cell.terrain === "build_pad");
  if (!pad.length) return null;
  const minX = Math.min(...pad.map((cell) => cell.x));
  const maxX = Math.max(...pad.map((cell) => cell.x));
  const minY = Math.min(...pad.map((cell) => cell.y));
  const maxY = Math.max(...pad.map((cell) => cell.y));
  const centerCell = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  const [x, z] = cellToWorld(centerCell.x, centerCell.y, world.dimensions);
  const needsService = world.buildPadStatus.includes("needs_service");
  const built = world.habitatBuildProgress >= 100;
  const serviced = world.serviceCount > 0 && !needsService;
  const statusColor = needsService ? "#fb7185" : built ? "#6ee7b7" : serviced ? "#67e8f9" : "#fbbf24";
  const unload = world.lastAction?.action === "unload" ? world.lastAction : null;
  const unloadIce = Math.abs(unload?.resourceDelta.ice ?? 0);
  const unloadSamples = Math.abs(unload?.resourceDelta.samples ?? 0);
  const showLabels = zoomMode === "manual" && zoomScale >= 1.35;
  const unloadOrigin: [number, number, number] = unload
    ? [unload.target.x - centerCell.x, 0.48, unload.target.y - centerCell.y]
    : [0, 0.48, 1.55];
  return (
    <group position={[x, 0.45, z]} userData={{ buildPadStatus: world.buildPadStatus }}>
      <PadDeck statusColor={statusColor} needsService={needsService} />
      <DockingLane />
      <UtilityGantry statusColor={statusColor} />
      <SolarServiceRack needsService={needsService} />
      <IceTankFarm delivered={world.iceDelivered} showLabel={showLabels} />
      <SampleStorageBay delivered={world.samplesDelivered} showLabel={showLabels} />
      <HabitatAssembly built={built} progress={world.habitatBuildProgress} statusColor={statusColor} />
      <Line points={[[0, 0.18, 1.35], [0.8, 0.18, 0.8], [1.35, 0.18, -0.55]]} color="#73d7e8" lineWidth={1.8} transparent opacity={0.68} />
      <Line points={[[0, 0.2, 1.35], [-0.72, 0.2, 0.82], [-1.42, 0.2, -0.5]]} color="#c4b5fd" lineWidth={1.6} transparent opacity={0.62} />
      {needsService ? <ServiceBeacon color={statusColor} /> : null}
      {unload && (unloadIce > 0 || unloadSamples > 0) ? <UnloadSignal key={unload.id} from={unloadOrigin} ice={unloadIce} samples={unloadSamples} /> : null}
    </group>
  );
}

function PadDeck({ statusColor, needsService }: { statusColor: string; needsService: boolean }) {
  return (
    <group>
      <mesh castShadow receiveShadow position={[0, 0.02, 0]}><boxGeometry args={[4.82, 0.13, 4.82]} /><meshStandardMaterial color={needsService ? "#7a3f3f" : "#dcc49a"} metalness={0.42} roughness={0.5} /></mesh>
      {[-1.6, -0.8, 0, 0.8, 1.6].map((offset) => <mesh key={`deck-x-${offset}`} position={[offset, 0.095, 0]}><boxGeometry args={[0.018, 0.012, 4.7]} /><meshBasicMaterial color="#1c2427" /></mesh>)}
      {[-1.6, -0.8, 0, 0.8, 1.6].map((offset) => <mesh key={`deck-z-${offset}`} position={[0, 0.096, offset]}><boxGeometry args={[4.7, 0.012, 0.018]} /><meshBasicMaterial color="#1c2427" /></mesh>)}
      <Line points={[[-2.35, 0.13, -2.35], [2.35, 0.13, -2.35], [2.35, 0.13, 2.35], [-2.35, 0.13, 2.35], [-2.35, 0.13, -2.35]]} color={statusColor} lineWidth={needsService ? 4 : 2} transparent opacity={needsService ? 0.95 : 0.72} />
      {[-1.8, -1.2, -0.6, 0.6, 1.2, 1.8].map((offset, index) => <mesh key={`hazard-${offset}`} position={[offset, 0.115, 2.28]} rotation={[0, index % 2 ? 0.45 : -0.45, 0]}><boxGeometry args={[0.32, 0.018, 0.08]} /><meshBasicMaterial color={index % 2 ? "#171a1b" : "#f6b94a"} /></mesh>)}
    </group>
  );
}

function DockingLane() {
  return (
    <group position={[0, 0.13, 1.25]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.54, 0.68, 32]} /><meshBasicMaterial color="#d4d4d0" transparent opacity={0.75} /></mesh>
      <mesh position={[0, 0.015, 0]}><boxGeometry args={[0.7, 0.025, 0.08]} /><meshBasicMaterial color="#f6b94a" /></mesh>
      {[-0.8, 0.8].map((side) => <mesh key={side} position={[side, 0.1, 0]}><boxGeometry args={[0.15, 0.2, 1.05]} /><meshStandardMaterial color="#272d30" metalness={0.75} roughness={0.3} /></mesh>)}
    </group>
  );
}

function UtilityGantry({ statusColor }: { statusColor: string }) {
  return (
    <group position={[0, 0.15, 1.2]}>
      {[-0.9, 0.9].map((side) => <mesh key={side} castShadow position={[side, 0.68, 0]}><boxGeometry args={[0.1, 1.35, 0.12]} /><meshStandardMaterial color="#9ba3a5" metalness={0.8} roughness={0.28} /></mesh>)}
      <mesh castShadow position={[0, 1.34, 0]}><boxGeometry args={[1.92, 0.12, 0.16]} /><meshStandardMaterial color="#858f92" metalness={0.8} roughness={0.28} /></mesh>
      <mesh position={[0, 1.27, 0.1]}><boxGeometry args={[0.42, 0.13, 0.1]} /><meshBasicMaterial color={statusColor} /></mesh>
    </group>
  );
}

function IceTankFarm({ delivered, showLabel }: { delivered: number; showLabel: boolean }) {
  const fill = Math.min(1, delivered / 24);
  return (
    <group position={[1.42, 0.12, -0.62]}>
      <mesh position={[0, 0.05, 0]}><boxGeometry args={[1.35, 0.12, 1.25]} /><meshStandardMaterial color="#323b3e" metalness={0.72} roughness={0.35} /></mesh>
      {[-0.34, 0.34].map((offset) => <group key={offset} position={[offset, 0.68, 0]}>
        <mesh castShadow><cylinderGeometry args={[0.27, 0.3, 1.18, 20]} /><meshStandardMaterial color="#b7c4c7" metalness={0.72} roughness={0.26} /></mesh>
        <mesh position={[0, -0.51 + fill * 0.48, 0.275]} scale={[0.11, Math.max(0.05, fill * 0.9), 0.018]}><boxGeometry /><meshBasicMaterial color="#67e8f9" transparent opacity={0.9} /></mesh>
        <mesh position={[0, 0.62, 0]}><cylinderGeometry args={[0.12, 0.12, 0.14, 12]} /><meshStandardMaterial color="#4b5558" metalness={0.8} /></mesh>
      </group>)}
      <Line points={[[0, 1.35, 0], [0, 1.48, 0], [-0.7, 1.48, 0]]} color="#8fa3a8" lineWidth={2} />
      {showLabel ? <Html center position={[0, 1.9, 0]} style={{ pointerEvents: "none" }}><WorldLabel icon={<Snowflake size={10} />} tone="ice">ICE TANK · {delivered.toFixed(1)} KG</WorldLabel></Html> : null}
    </group>
  );
}

function SampleStorageBay({ delivered, showLabel }: { delivered: number; showLabel: boolean }) {
  const filled = Math.min(6, Math.ceil(delivered / 0.5));
  return (
    <group position={[-1.42, 0.12, -0.62]}>
      <mesh position={[0, 0.58, -0.18]}><boxGeometry args={[1.28, 1.12, 0.12]} /><meshStandardMaterial color="#43484b" metalness={0.72} roughness={0.34} /></mesh>
      {[-0.45, 0, 0.45].map((offset) => <mesh key={`post-${offset}`} position={[offset, 0.58, 0]}><boxGeometry args={[0.06, 1.15, 0.76]} /><meshStandardMaterial color="#777f82" metalness={0.8} roughness={0.3} /></mesh>)}
      {[0.12, 0.56, 1].map((height) => <mesh key={`shelf-${height}`} position={[0, height, 0]}><boxGeometry args={[1.25, 0.06, 0.76]} /><meshStandardMaterial color="#7a8284" metalness={0.8} roughness={0.28} /></mesh>)}
      {Array.from({ length: 6 }, (_, index) => {
        const active = index < filled;
        return <mesh key={index} castShadow position={[-0.24 + (index % 2) * 0.48, 0.33 + Math.floor(index / 2) * 0.43, 0.04]}><boxGeometry args={[0.36, 0.27, 0.48]} /><meshStandardMaterial color={active ? "#9a8ad1" : "#2c3033"} metalness={0.45} roughness={0.46} emissive={active ? "#33255d" : "#000000"} emissiveIntensity={active ? 0.3 : 0} /></mesh>;
      })}
      {showLabel ? <Html center position={[0, 2.2, 0]} style={{ pointerEvents: "none" }}><WorldLabel icon={<PackageCheck size={10} />} tone="sample">SAMPLE VAULT · {delivered.toFixed(1)} KG</WorldLabel></Html> : null}
    </group>
  );
}

function SolarServiceRack({ needsService }: { needsService: boolean }) {
  return (
    <group position={[0, 0.12, -1.72]}>
      {[-0.58, 0, 0.58].map((offset) => <mesh key={offset} castShadow position={[offset, 0.45, 0]} rotation={[-0.34, 0, 0]}><boxGeometry args={[0.48, 0.04, 0.72]} /><meshStandardMaterial color={needsService ? "#6f343d" : "#244653"} metalness={0.5} roughness={0.25} emissive={needsService ? "#4a151c" : "#123746"} emissiveIntensity={0.22} /></mesh>)}
      <mesh position={[0, 0.15, 0]}><boxGeometry args={[1.8, 0.1, 0.12]} /><meshStandardMaterial color="#697174" metalness={0.75} /></mesh>
    </group>
  );
}

function HabitatAssembly({ built, progress, statusColor }: { built: boolean; progress: number; statusColor: string }) {
  const supports = Math.min(4, Math.ceil(progress / 25));
  return (
    <group position={[0, 0.13, -0.18]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.62, 0.78, 32]} /><meshStandardMaterial color="#9da3a1" metalness={0.62} roughness={0.36} /></mesh>
      {Array.from({ length: supports }, (_, index) => { const angle = index / 4 * Math.PI * 2 + Math.PI / 4; return <mesh key={index} castShadow position={[Math.cos(angle) * 0.58, 0.45, Math.sin(angle) * 0.58]} rotation={[0, -angle, 0]}><boxGeometry args={[0.12, 0.9, 0.12]} /><meshStandardMaterial color="#aeb4b1" metalness={0.68} roughness={0.32} emissive={statusColor} emissiveIntensity={0.08} /></mesh>; })}
      {built ? <mesh castShadow position={[0, 0.62, 0]} scale={[1.05, 0.82, 1.05]}><sphereGeometry args={[0.72, 24, 14, 0, Math.PI * 2, 0, Math.PI / 2]} /><meshStandardMaterial color="#d9ddd5" metalness={0.35} roughness={0.38} emissive="#17463d" emissiveIntensity={0.12} /></mesh> : <mesh position={[0, 0.08, 0]}><cylinderGeometry args={[0.52, 0.58, 0.16, 20]} /><meshStandardMaterial color="#6d7475" metalness={0.7} roughness={0.35} /></mesh>}
    </group>
  );
}

function UnloadSignal({ from, ice, samples }: { from: [number, number, number]; ice: number; samples: number }) {
  const ref = useRef<Group>(null);
  const elapsed = useRef(0);
  useFrame((_, delta) => { elapsed.current += delta; if (ref.current) ref.current.visible = elapsed.current < 2.25; });
  return (
    <group ref={ref}>
      {ice > 0 ? <TransferStream from={from} to={[1.42, 0.72, -0.62]} color="#67e8f9" shape="sphere" /> : null}
      {samples > 0 ? <TransferStream from={from} to={[-1.42, 0.72, -0.62]} color="#c4b5fd" shape="box" /> : null}
      {ice > 0 ? <Line points={[from, [1.42, 0.72, -0.62]]} color="#67e8f9" lineWidth={4} transparent opacity={0.76} /> : null}
      {samples > 0 ? <Line points={[from, [-1.42, 0.72, -0.62]]} color="#c4b5fd" lineWidth={4} transparent opacity={0.76} /> : null}
      <Html center position={[0, 3.15, 0.35]} style={{ pointerEvents: "none" }}><WorldLabel tone="active">UNLOADING · {ice.toFixed(1)} KG ICE · {samples.toFixed(1)} KG SAMPLES</WorldLabel></Html>
    </group>
  );
}

function TransferStream({ from, to, color, shape }: { from: [number, number, number]; to: [number, number, number]; color: string; shape: "sphere" | "box" }) {
  const ref = useRef<Group>(null);
  const elapsed = useRef(0);
  useFrame((_, delta) => {
    elapsed.current += delta;
    ref.current?.children.forEach((child, index) => {
      const t = (elapsed.current * 1.5 + index * 0.25) % 1;
      child.position.set(MathUtils.lerp(from[0], to[0], t), MathUtils.lerp(from[1], to[1], t) + Math.sin(t * Math.PI) * 0.3, MathUtils.lerp(from[2], to[2], t));
    });
  });
  return <group ref={ref}>{[0, 1, 2].map((index) => <mesh key={index}>{shape === "sphere" ? <sphereGeometry args={[0.12, 10, 8]} /> : <boxGeometry args={[0.19, 0.19, 0.19]} />}<meshBasicMaterial color={color} /></mesh>)}</group>;
}

function WorldLabel({ children, icon, tone }: { children: React.ReactNode; icon?: React.ReactNode; tone: "neutral" | "ice" | "sample" | "good" | "active" }) {
  const styles: Record<typeof tone, string> = {
    ice: "border-cyan-200/35 bg-cyan-950/90 text-cyan-100",
    sample: "border-violet-200/35 bg-violet-950/90 text-violet-100",
    good: "border-emerald-200/35 bg-emerald-950/90 text-emerald-100",
    active: "border-white/35 bg-black/90 text-white",
    neutral: "border-white/15 bg-[#111416]/90 text-stone-200",
  };
  return <span className={`flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-1 font-mono text-[7px] font-bold tracking-[0.08em] shadow-xl ${styles[tone]}`}>{icon}{children}</span>;
}

function ServiceBeacon({ color }: { color: string }) {
  const ref = useRef<Group>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const pulse = 1 + Math.sin(clock.elapsedTime * 5) * 0.12;
    ref.current.scale.setScalar(pulse);
  });
  return (
    <group ref={ref}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.07, 0]}><ringGeometry args={[1.55, 1.75, 48]} /><meshBasicMaterial color={color} transparent opacity={0.82} depthWrite={false} /></mesh>
      <mesh position={[0, 2.2, 0]}><sphereGeometry args={[0.18, 12, 8]} /><meshBasicMaterial color={color} /></mesh>
      <mesh position={[0, 1.22, 0]} scale={[0.035, 2, 0.035]}><cylinderGeometry /><meshBasicMaterial color={color} transparent opacity={0.65} /></mesh>
    </group>
  );
}

function ActionPulse({ world }: { world: WorldPresentation }) {
  const ref = useRef<Group>(null);
  const elapsed = useRef(0);
  const action = world.lastAction;
  useFrame(({ clock }, delta) => {
    if (!ref.current) return;
    elapsed.current += delta;
    ref.current.visible = elapsed.current < 1.25;
    const pulse = 1 + Math.sin(clock.elapsedTime * 8) * 0.16;
    ref.current.scale.setScalar(pulse);
  });
  if (!action) return null;
  const [x, z] = cellToWorld(action.target.x, action.target.y, world.dimensions);
  const elevation = terrainAt(world, action.target.x, action.target.y)?.elevation ?? 0.1;
  const colors: Record<string, string> = { service: "#fb7185", extract: "#fbbf24", unload: "#c4b5fd", build: "#6ee7b7" };
  const color = colors[action.action] ?? "#67e8f9";
  return (
    <group ref={ref} position={[x, elevation + 0.55, z]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.42, 0.58, 32]} /><meshBasicMaterial color={color} transparent opacity={0.92} depthWrite={false} /></mesh>
      <mesh position={[0, 0.34, 0]} scale={[0.025, 0.68, 0.025]}><cylinderGeometry /><meshBasicMaterial color={color} transparent opacity={0.58} /></mesh>
    </group>
  );
}

function SelectionMarker({ cell, world, highlighted }: { cell: { x: number; y: number }; world: WorldPresentation; highlighted: boolean }) {
  const [x, z] = cellToWorld(cell.x, cell.y, world.dimensions);
  const elevation = terrainAt(world, cell.x, cell.y)?.elevation ?? 0.1;
  return (
    <mesh position={[x, elevation + 0.4, z]}>
      <boxGeometry args={[1.06, 0.08, 1.06]} />
      <meshBasicMaterial color={highlighted ? "#facc15" : "#67e8f9"} transparent opacity={0.42} wireframe />
    </mesh>
  );
}

function WorldNavigation({ width, height, terrain, rover, center, visibleWidth, visibleHeight, rotation, setRotation, cameraView, changeCameraView, roverFov, zoomMode, zoomScale, zoomIn, zoomOut, fitViewport }: {
  width: number;
  height: number;
  terrain: TerrainPresentation[];
  rover: { x: number; y: number } | null;
  center: { x: number; y: number };
  visibleWidth: number;
  visibleHeight: number;
  rotation: number;
  setRotation: (rotation: number) => void;
  cameraView: CameraView;
  changeCameraView: (view: CameraView) => void;
  roverFov: number;
  zoomMode: "fit" | "manual";
  zoomScale: number;
  zoomIn: () => void;
  zoomOut: () => void;
  fitViewport: () => void;
}) {
  const audioMuted = useAresStore((state) => state.audioMuted);
  const toggleAudioMuted = useAresStore((state) => state.toggleAudioMuted);
  const showGrid = useAresStore((state) => state.showGrid);
  const toggleGrid = useAresStore((state) => state.toggleGrid);
  const showRoverVisibility = useAresStore((state) => state.showRoverVisibility);
  const toggleRoverVisibility = useAresStore((state) => state.toggleRoverVisibility);
  const overlayMode = useAresStore((state) => state.overlayMode);
  const setOverlayMode = useAresStore((state) => state.setOverlayMode);
  const setViewportCenter = useAresStore((state) => state.setViewportCenter);
  const normalized = Math.round(((rotation % 360) + 360) % 360);
  const layers: Array<{ mode: Exclude<OverlayMode, "none">; label: string; icon: React.ReactNode }> = [
    { mode: "ice", label: "Ice layer", icon: <Snowflake size={14} /> },
    { mode: "ore", label: "Ore layer", icon: <Gem size={14} /> },
    { mode: "dust", label: "Dust layer", icon: <Wind size={14} /> },
    { mode: "roughness", label: "Roughness layer", icon: <Mountain size={14} /> },
  ];
  return (
    <div className="absolute right-4 top-4 z-20 w-48 overflow-hidden rounded-xl border border-white/10 bg-[#0b0d0f]/88 shadow-2xl backdrop-blur-xl" aria-label="Environment controls">
      <div className="flex items-center justify-between border-b border-white/8 px-2.5 py-2">
        <span className="flex items-center gap-1.5 text-[9px] font-semibold text-stone-300"><Compass size={13} className="text-cyan-200/70" /> Environment</span>
        {cameraView === "survey" ? <span className="flex items-center gap-1">
          <label className="flex items-center rounded-md border border-white/10 bg-black/20 px-1.5 py-1 text-[9px] font-bold tabular-nums text-cyan-100"><input aria-label="World angle" className="w-9 bg-transparent text-right outline-none" inputMode="numeric" maxLength={3} onChange={(event) => { const next = Number(event.target.value); if (Number.isFinite(next)) setRotation(Math.max(0, Math.min(359, next))); }} type="text" value={normalized} />°</label>
          <button aria-label="Reset view angle" className="grid h-6 w-6 place-items-center rounded-md border border-white/10 text-stone-500 hover:border-cyan-200/30 hover:text-cyan-100" onClick={() => setRotation(45)} type="button"><LocateFixed size={12} /></button>
        </span> : <span className={`rounded-md border px-2 py-1 text-[8px] font-semibold ${cameraView === "top" ? "border-cyan-200/15 bg-cyan-300/8 text-cyan-100" : "border-amber-200/15 bg-amber-300/8 text-amber-100"}`}>{cameraView === "top" ? "TOP VIEW" : "ROVER POV"}</span>}
      </div>
      <div className="border-b border-white/8 px-2 pb-2 pt-1.5" aria-label="Environment viewpoints">
        <div className="mb-1 text-[8px] font-medium text-stone-600">Viewpoint</div>
        <div className="grid grid-cols-3 gap-1.5">
          <ViewButton label="3D survey view" active={cameraView === "survey"} onClick={() => changeCameraView("survey")}><Compass size={15} /></ViewButton>
          <ViewButton label="Top view" active={cameraView === "top"} onClick={() => changeCameraView("top")}><MapIcon size={15} /></ViewButton>
          <ViewButton label="Rover point of view" active={cameraView === "rover"} onClick={() => changeCameraView("rover")}><Eye size={15} /></ViewButton>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-1.5 p-2" aria-label="Camera zoom controls">
        <ViewButton label="Zoom out" onClick={zoomOut}><Minus size={15} /></ViewButton>
        <ViewButton label={cameraView === "rover" ? "Return to fitted 3D view" : "Fit environment"} active={cameraView !== "rover" && zoomMode === "fit"} onClick={fitViewport}><Maximize size={15} /></ViewButton>
        <ViewButton label="Zoom in" onClick={zoomIn}><Plus size={15} /></ViewButton>
      </div>
      {cameraView === "survey" ? <label className="block border-t border-white/8 px-2.5 py-2 text-[8px] font-semibold uppercase tracking-[0.14em] text-stone-500">
        Rotate view
        <input aria-label="Rotate world" className="mt-1.5 h-1.5 w-full cursor-pointer accent-cyan-300" max={359} min={0} onChange={(event) => setRotation(Number(event.target.value))} type="range" value={normalized} />
      </label> : <div className="border-t border-white/8 px-2.5 py-2 text-[8px] leading-4 text-stone-500">{cameraView === "top" ? "North-up and window-aligned. Wheel or ± changes scale." : "Arrow keys follow rover facing: ↑ forward, ↓ back, ←/→ side movement. Dragging updates that frame."}</div>}
      <div className="grid grid-cols-3 gap-1.5 border-t border-white/8 p-2" aria-label="Environment display controls">
        <ViewButton label={audioMuted ? "Unmute simulation audio" : "Mute simulation audio"} active={audioMuted} onClick={toggleAudioMuted}>{audioMuted ? <VolumeX size={15} /> : <Volume2 size={15} />}</ViewButton>
        <ViewButton label={showGrid ? "Hide cell boundaries" : "Show cell boundaries"} active={showGrid} onClick={toggleGrid}><Grid3X3 size={15} /></ViewButton>
        <ViewButton label={showRoverVisibility ? "Hide rover visibility" : "Show rover visibility"} active={showRoverVisibility} onClick={toggleRoverVisibility}><Flashlight size={15} /></ViewButton>
      </div>
      <div className="border-t border-white/8 px-2 pb-2 pt-1.5" aria-label="Environment data layers">
        <div className="mb-1 text-[8px] font-medium text-stone-600">Data layers</div>
        <div className="grid grid-cols-4 gap-1.5">
          {layers.map((layer) => <ViewButton key={layer.mode} label={layer.label} active={overlayMode === layer.mode} onClick={() => setOverlayMode(overlayMode === layer.mode ? "none" : layer.mode)}>{layer.icon}</ViewButton>)}
        </div>
      </div>
      {cameraView !== "rover" && zoomMode === "manual" ? <div className="border-t border-white/8 p-2"><MiniView center={center} height={height} onNavigate={setViewportCenter} rover={rover} terrain={terrain} visibleHeight={visibleHeight} visibleWidth={visibleWidth} width={width} /></div> : null}
      <div className="border-t border-white/8 px-2.5 py-1.5 text-center text-[8px] font-semibold uppercase tracking-[0.12em] text-stone-600" data-testid={cameraView === "rover" ? "rover-pov-readout" : zoomMode === "manual" ? "zoom-readout" : undefined}>{cameraView === "rover" ? `${roverFov}° field of view` : zoomMode === "manual" ? `${Math.round(zoomScale * 100)}% zoom` : cameraView === "top" ? "North-up · aligned" : `${rotationLabel(rotation)} 3D survey`}</div>
    </div>
  );
}

function ViewButton({ label, onClick, children, active = false }: { label: string; onClick: () => void; children: React.ReactNode; active?: boolean }) {
  return <button aria-label={label} title={label} className={`grid h-8 place-items-center rounded-md border transition ${active ? "border-cyan-200/40 bg-cyan-300/15 text-cyan-100 shadow-[inset_0_0_12px_rgba(103,232,249,0.08)]" : "border-white/10 bg-white/[0.025] text-stone-400 hover:border-cyan-200/25 hover:text-cyan-100"}`} onClick={onClick} type="button">{children}</button>;
}

function MiniView({ width, height, terrain, rover, center, visibleWidth, visibleHeight, onNavigate }: {
  width: number;
  height: number;
  terrain: TerrainPresentation[];
  rover: { x: number; y: number } | null;
  center: { x: number; y: number };
  visibleWidth: number;
  visibleHeight: number;
  onNavigate: (center: { x: number; y: number }) => void;
}) {
  const navigate = (event: React.MouseEvent<HTMLButtonElement>) => {
    const svg = event.currentTarget.querySelector("svg");
    if (!svg) return;
    const bounds = svg.getBoundingClientRect();
    onNavigate({
      x: ((event.clientX - bounds.left) / bounds.width) * width,
      y: ((event.clientY - bounds.top) / bounds.height) * height,
    });
  };
  return (
    <button
      aria-label="Rover navigation overview"
      className="relative block h-28 w-full cursor-crosshair overflow-hidden rounded-lg border border-orange-200/15 bg-[#170b08]/85 p-2 text-left transition hover:border-cyan-200/35 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-200/70"
      data-testid="mini-view"
      onClick={navigate}
      title="Click to pan the zoomed camera"
      type="button"
    >
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="h-full w-full" role="img" aria-hidden="true">
        {terrain.map((cell) => <rect key={cell.id} x={cell.x} y={cell.y} width="1.08" height="1.08" fill={TERRAIN_COLORS[cell.terrain]} />)}
        <rect x={center.x - visibleWidth / 2} y={center.y - visibleHeight / 2} width={visibleWidth} height={visibleHeight} fill="none" stroke="#67e8f9" strokeWidth="0.7" />
        {rover ? <circle cx={rover.x + 0.5} cy={rover.y + 0.5} r="1.2" fill="#e0f2fe" stroke="#0891b2" strokeWidth="0.55" /> : null}
      </svg>
      <span className="pointer-events-none absolute left-2 top-1.5 text-[7px] font-semibold uppercase tracking-[0.12em] text-cyan-100/65">Click to pan</span>
      <span className="pointer-events-none absolute bottom-1 right-1 text-cyan-100"><LocateFixed size={12} /></span>
    </button>
  );
}

function hoverSummary(target: SelectionTarget, world: WorldPresentation) {
  if (target.kind === "cell") {
    const cell = terrainAt(world, target.x, target.y);
    if (!cell) return null;
    return { title: `${cell.terrain.replace("_", " ")} · ${cell.x}, ${cell.y}`, detail: `rough ${Math.round(cell.roughness * 100)}% · dust ${Math.round(cell.dust * 100)}%` };
  }
  if (target.kind === "rover") {
    const rover = world.rovers.find((item) => item.id === target.id);
    return { title: rover?.name ?? "Rover", detail: rover ? `${Math.round(rover.battery)}% battery · ${rover.currentTask}` : "Open inspector for details" };
  }
  if (target.kind === "structure") {
    const structure = world.structures.find((item) => item.id === target.id);
    return { title: structure?.name ?? "Structure", detail: structure ? `${structure.health}% health · ${structure.status}` : "Open inspector for details" };
  }
  return { title: "Selection", detail: "Open inspector for details" };
}

function HoverReadout({ target, world }: { target: SelectionTarget | null; world: WorldPresentation }) {
  if (!target) return null;
  const summary = hoverSummary(target, world);
  if (!summary) return null;
  return (
    <div className="pointer-events-none absolute bottom-4 left-1/2 z-20 -translate-x-1/2 rounded-lg border border-white/10 bg-black/55 px-3 py-2 text-center shadow-xl backdrop-blur-md">
      <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-stone-100">{summary.title}</div>
      <div className="mt-0.5 text-[10px] text-stone-400">{summary.detail}</div>
    </div>
  );
}

function cellToWorld(x: number, y: number, dimensions: WorldPresentation["dimensions"]): [number, number] {
  return [x - dimensions.width / 2 + 0.5, y - dimensions.height / 2 + 0.5];
}

function terrainAt(world: WorldPresentation, x: number, y: number) {
  return world.terrain[y * world.dimensions.width + x];
}

const OVERLAY_COLORS: Record<Exclude<OverlayMode, "none">, { low: string; high: string; value: (cell: TerrainPresentation) => number }> = {
  ice: { low: "#1e293b", high: "#a5f3fc", value: (cell) => cell.ice },
  ore: { low: "#2a1b16", high: "#fbbf24", value: (cell) => cell.ore },
  dust: { low: "#431407", high: "#fb7185", value: (cell) => cell.dust },
  roughness: { low: "#102a2a", high: "#67e8f9", value: (cell) => cell.roughness },
};

function padColorForStatus(buildPadStatus: WorldPresentation["buildPadStatus"]) {
  if (buildPadStatus.includes("needs_service")) return "#8f303b";
  if (buildPadStatus === "habitat_built") return "#d7cda8";
  return "#dcc49a";
}

function colorForCell(cell: TerrainPresentation, overlay: OverlayMode, buildPadStatus: WorldPresentation["buildPadStatus"]) {
  const padColor = padColorForStatus(buildPadStatus);
  const base = new Color(cell.terrain === "build_pad" ? padColor : TERRAIN_COLORS[cell.terrain]);
  if (overlay === "none") return base;
  const palette = OVERLAY_COLORS[overlay];
  return new Color(palette.low).lerp(new Color(palette.high), MathUtils.clamp(palette.value(cell), 0, 1));
}

function rotationLabel(rotation: number) {
  const normalized = ((rotation % 360) + 360) % 360;
  if (normalized === 45) return "NE";
  if (normalized === 135) return "SE";
  if (normalized === 225) return "SW";
  if (normalized === 315) return "NW";
  return `${normalized}°`;
}
