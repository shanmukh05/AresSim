import { TerrainType } from './types';

export const GRID_SIZE = 15;
export const MAX_STATS = 100;
export const TICK_RATE_MS = 800; // Speed of simulation

export const INITIAL_AGENT_STATE = {
  health: 100,
  energy: 100,
  oxygen: 100,
  bodyTemp: 37,
  inventory: {
    ice: 0,
    samples: 0,
    oxygenTanks: 3,
    rechargePacks: 2,
  },
};

export const TERRAIN_COLORS: Record<TerrainType, string> = {
  [TerrainType.BASE]: 'bg-blue-500/20 border-blue-500',
  [TerrainType.FLAT]: 'bg-orange-900/20 border-orange-900/40',
  [TerrainType.ROCKY]: 'bg-stone-700/40 border-stone-600',
  [TerrainType.SANDY]: 'bg-amber-700/30 border-amber-800',
  [TerrainType.CRATER]: 'bg-red-900/30 border-red-800',
  [TerrainType.ICE]: 'bg-cyan-300/20 border-cyan-400/50',
};

// Movement costs
export const COST_MOVE_FLAT = 1;
export const COST_MOVE_ROCKY = 3;
export const COST_MOVE_SANDY = 2;
export const OXYGEN_DECAY = 0.5;
export const TEMP_NIGHT = -80;
export const TEMP_DAY = -20;