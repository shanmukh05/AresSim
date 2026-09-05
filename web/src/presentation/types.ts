/**
 * Mesh/HUD types derived from a snapshot.
 * Keep this thinner than `SimSnapshot`; do not add gameplay fields the engine does not send.
 */

import type { ActionType, BuildPadStatus, RoverEntity, StructureEntity, TerrainType } from "../types/sim";

export interface TerrainPresentation {
  id: string;
  x: number;
  y: number;
  terrain: TerrainType;
  elevation: number;
  ice: number;
  ore: number;
  dust: number;
  roughness: number;
  scanned: boolean;
  extracted: boolean;
}

export interface RoverPresentation extends RoverEntity {
  kind: "rover";
}

export interface StructurePresentation extends StructureEntity {
  kind: "structure";
}

export interface WorldPresentation {
  seed: number;
  dimensions: { width: number; height: number };
  terrain: TerrainPresentation[];
  rovers: RoverPresentation[];
  structures: StructurePresentation[];
  buildPadStatus: BuildPadStatus;
  habitatBuildProgress: number;
  habitatBuildCount: number;
  serviceCount: number;
  iceDelivered: number;
  samplesDelivered: number;
  path: Array<{ x: number; y: number }>;
  lastAction?: { id: string; action: ActionType; target: { x: number; y: number }; resourceDelta: { ice?: number; ore?: number; samples?: number } };
}
