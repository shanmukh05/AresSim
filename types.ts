export enum TerrainType {
  FLAT = 'FLAT',
  ROCKY = 'ROCKY',
  CRATER = 'CRATER', // Hazard
  SANDY = 'SANDY',
  BASE = 'BASE',
  ICE = 'ICE', // Resource
}

export enum AgentAction {
  MOVE_NORTH = 'MOVE_NORTH',
  MOVE_SOUTH = 'MOVE_SOUTH',
  MOVE_EAST = 'MOVE_EAST',
  MOVE_WEST = 'MOVE_WEST',
  MINE_ICE = 'MINE_ICE',
  USE_OXYGEN_TANK = 'USE_OXYGEN_TANK',
  CHARGE_BATTERY = 'CHARGE_BATTERY',
  REPAIR_SUIT = 'REPAIR_SUIT',
  BUILD_SOLAR = 'BUILD_SOLAR',
  IDLE = 'IDLE',
}

export interface Coordinates {
  x: number;
  y: number;
}

export interface AgentState {
  position: Coordinates;
  health: number; // 0-100
  energy: number; // 0-100
  oxygen: number; // 0-100
  bodyTemp: number; // Celsius
  inventory: {
    ice: number;
    samples: number;
    oxygenTanks: number;
    rechargePacks: number;
  };
}

export interface EnvironmentState {
  sol: number; // Day count
  timeOfDay: number; // 0-24
  temperature: number; // Celsius
  radiationLevel: number; // sieverts/hr (abstracted 0-10)
  dustStormIntensity: number; // 0-1 (0 = clear, 1 = severe storm)
  grid: TerrainType[][];
  solarPanels: Coordinates[];
}

export interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'warning' | 'danger' | 'success';
  reward?: number;
}