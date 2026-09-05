/**
 * Map a backend snapshot into renderer-friendly world data.
 * Elevation and path decoration are presentation only; they do not change rules.
 */

import type { SimSnapshot, TerrainCell } from "../../types/sim";
import type { TerrainPresentation, WorldPresentation } from "../types";

function elevationFor(cell: TerrainCell) {
  if (cell.extracted) return 0.16;
  if (cell.terrain === "ridge") return 0.92;
  if (cell.terrain === "dune") return 0.38;
  if (cell.terrain === "rock") return 0.3;
  if (cell.terrain === "build_pad") return 0.2;
  if (cell.terrain === "ice") return 0.16;
  if (cell.terrain === "crater") return -0.18;
  return 0.16;
}

/** Flatten terrain, rover path, and last-action markers for the R3F scene. */
export function snapshotToWorld(snapshot: SimSnapshot): WorldPresentation {
  const terrain: TerrainPresentation[] = snapshot.terrain.flat().map((cell) => ({
    id: `cell-${cell.x}-${cell.y}`,
    x: cell.x,
    y: cell.y,
    terrain: cell.terrain,
    elevation: elevationFor(cell),
    ice: cell.ice,
    ore: cell.ore,
    dust: cell.dust,
    roughness: cell.roughness,
    scanned: cell.scanned,
    extracted: cell.extracted,
  }));

  return {
    seed: snapshot.seed,
    dimensions: snapshot.terrainSize,
    terrain,
    rovers: snapshot.rovers.map((rover) => ({ ...rover, kind: "rover" })),
    structures: snapshot.structures.map((structure) => ({ ...structure, kind: "structure" })),
    buildPadStatus: snapshot.buildPadState.status,
    habitatBuildProgress: snapshot.objectiveStats.habitatBuildProgress,
    habitatBuildCount: snapshot.objectiveStats.habitatBuildCount,
    serviceCount: snapshot.objectiveStats.serviceCount,
    iceDelivered: snapshot.objectiveStats.iceDelivered,
    samplesDelivered: snapshot.objectiveStats.samplesDelivered,
    path: [...snapshot.history]
      .reverse()
      .filter((entry) => entry.action === "move" && entry.target)
      .map((entry) => ({ x: entry.target!.x, y: entry.target!.y })),
    lastAction: snapshot.history[0]?.target ? {
      id: snapshot.history[0].id,
      action: snapshot.history[0].action,
      target: snapshot.history[0].target,
      resourceDelta: snapshot.history[0].resourceDelta,
    } : undefined,
  };
}
