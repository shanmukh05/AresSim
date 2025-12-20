import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  AgentState, 
  EnvironmentState, 
  TerrainType, 
  AgentAction, 
  Coordinates, 
  LogEntry 
} from '../types';
import { 
  GRID_SIZE, 
  INITIAL_AGENT_STATE, 
  COST_MOVE_FLAT, 
  COST_MOVE_ROCKY, 
  COST_MOVE_SANDY,
  OXYGEN_DECAY,
  MAX_STATS
} from '../constants';

const generateGrid = (): TerrainType[][] => {
  const grid: TerrainType[][] = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(TerrainType.FLAT));
  
  // Seed random terrain
  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const rand = Math.random();
      if (x === Math.floor(GRID_SIZE / 2) && y === Math.floor(GRID_SIZE / 2)) {
        grid[y][x] = TerrainType.BASE;
      } else if (rand > 0.92) {
        grid[y][x] = TerrainType.ICE;
      } else if (rand > 0.8) {
        grid[y][x] = TerrainType.ROCKY;
      } else if (rand > 0.7) {
        grid[y][x] = TerrainType.CRATER;
      } else if (rand > 0.5) {
        grid[y][x] = TerrainType.SANDY;
      }
    }
  }
  return grid;
};

export const useMarsSurvival = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [agent, setAgent] = useState<AgentState>({
    ...INITIAL_AGENT_STATE,
    position: { x: Math.floor(GRID_SIZE / 2), y: Math.floor(GRID_SIZE / 2) }
  });

  const [environment, setEnvironment] = useState<EnvironmentState>({
    sol: 1,
    timeOfDay: 8.0, 
    temperature: -30,
    radiationLevel: 1.2,
    dustStormIntensity: 0,
    grid: generateGrid(),
    solarPanels: []
  });

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [totalReward, setTotalReward] = useState(0);
  const [lastReward, setLastReward] = useState<{ value: number; source: string } | null>(null);

  const addLog = (message: string, type: LogEntry['type'] = 'info', reward: number = 0) => {
    const newLog: LogEntry = {
      id: Math.random().toString(36).substring(7),
      timestamp: new Date().toLocaleTimeString(),
      message,
      type,
      reward
    };
    setLogs(prev => [newLog, ...prev].slice(0, 50));
    
    if (reward !== 0) {
      setTotalReward(prev => prev + reward);
      setLastReward({ value: reward, source: message });
    }
  };

  const moveAgent = useCallback((dx: number, dy: number) => {
    setAgent(prev => {
      const newX = prev.position.x + dx;
      const newY = prev.position.y + dy;

      if (newX < 0 || newX >= GRID_SIZE || newY < 0 || newY >= GRID_SIZE) {
        addLog("Cannot move outside mission zone.", 'warning', -1);
        return prev;
      }

      const terrain = environment.grid[newY][newX];
      const terrainName = terrain.charAt(0) + terrain.slice(1).toLowerCase();
      let energyCost = COST_MOVE_FLAT;
      let moveMsg = `Moved agent (${terrainName})`;
      
      if (terrain === TerrainType.ROCKY) energyCost = COST_MOVE_ROCKY;
      if (terrain === TerrainType.SANDY) energyCost = COST_MOVE_SANDY;
      
      let moveReward = -energyCost;

      if (terrain === TerrainType.CRATER) {
        addLog(`Warning: Entered unstable ${terrainName} terrain!`, 'danger', -10);
        moveReward = -10;
        moveMsg = `Entered ${terrainName} Hazard`;
        if (Math.random() > 0.5) {
             return {
                ...prev,
                position: { x: newX, y: newY },
                energy: Math.max(0, prev.energy - energyCost * 2),
                health: Math.max(0, prev.health - 5)
            };
        }
      } else {
         addLog(moveMsg, 'info', moveReward);
      }

      return {
        ...prev,
        position: { x: newX, y: newY },
        energy: Math.max(0, prev.energy - energyCost),
      };
    });
  }, [environment.grid]);

  const performAction = useCallback((action: AgentAction) => {
    switch (action) {
      case AgentAction.MOVE_NORTH: moveAgent(0, -1); break;
      case AgentAction.MOVE_SOUTH: moveAgent(0, 1); break;
      case AgentAction.MOVE_EAST: moveAgent(1, 0); break;
      case AgentAction.MOVE_WEST: moveAgent(-1, 0); break;
      case AgentAction.MINE_ICE:
        const { x, y } = agent.position;
        if (environment.grid[y][x] === TerrainType.ICE) {
          setAgent(prev => ({
            ...prev,
            energy: Math.max(0, prev.energy - 10),
            inventory: { ...prev.inventory, ice: prev.inventory.ice + 1 }
          }));
          
          setEnvironment(prevEnv => {
            const newGrid = prevEnv.grid.map(row => [...row]);
            newGrid[y][x] = TerrainType.FLAT;
            return { ...prevEnv, grid: newGrid };
          });

          addLog("Mined water ice sample. Deposit depleted.", 'success', 50);
        } else {
          addLog("No resources to mine here.", 'warning', -2);
        }
        break;
      case AgentAction.USE_OXYGEN_TANK:
        if (agent.inventory.oxygenTanks > 0) {
            setAgent(prev => ({
                ...prev,
                oxygen: Math.min(MAX_STATS, prev.oxygen + 50),
                health: 100,
                inventory: { ...prev.inventory, oxygenTanks: prev.inventory.oxygenTanks - 1 }
            }));
            addLog("Oxygen tank utilized. Health restored.", 'success', 10);
        } else {
            addLog("Inventory empty: No oxygen tanks.", 'warning');
        }
        break;
      case AgentAction.CHARGE_BATTERY:
         const { x: cx, y: cy } = agent.position;
         if (environment.grid[cy][cx] === TerrainType.BASE) {
             if (agent.inventory.rechargePacks > 0) {
                 setAgent(prev => ({
                     ...prev,
                     energy: Math.min(MAX_STATS, prev.energy + 50),
                     health: 100,
                     inventory: { ...prev.inventory, rechargePacks: prev.inventory.rechargePacks - 1 }
                 }));
                 addLog("Energy pack used. Health restored.", 'success', 10);
             } else {
                 addLog("Cannot recharge: No Recharge Packs.", 'warning');
             }
         } else {
             addLog("Must be at Base Station to utilize recharge packs.", 'warning');
         }
         break;
    }
  }, [agent, environment, moveAgent]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setEnvironment(prevEnv => {
        let newTime = prevEnv.timeOfDay + 0.5;
        let newSol = prevEnv.sol;
        if (newTime >= 24) {
          newTime = 0;
          newSol += 1;
          addLog(`Sol ${newSol} began.`, 'info');
        }
        
        const isNight = newTime < 6 || newTime > 20;
        const newTemp = isNight ? -80 : -20 + (Math.random() * 5);
        
        // Fluctuating Radiation
        let newRad = prevEnv.radiationLevel + (Math.random() * 0.4 - 0.2);
        if (Math.random() > 0.98) newRad += 4.0; // Radiation burst
        newRad = Math.max(0.1, Math.min(10.0, newRad));

        // Dust Storm logic
        let newDust = prevEnv.dustStormIntensity;
        if (Math.random() > 0.95) {
            newDust = Math.min(1, newDust + 0.2);
            if (newDust > 0.5) addLog("Dust storm intensifying!", 'danger');
        } else {
            newDust = Math.max(0, newDust - 0.05);
        }

        return {
          ...prevEnv,
          timeOfDay: newTime,
          sol: newSol,
          temperature: newTemp,
          radiationLevel: newRad,
          dustStormIntensity: newDust
        };
      });

      setAgent(prevAgent => {
        let newOxy = Math.max(0, prevAgent.oxygen - OXYGEN_DECAY);
        let newEnergy = Math.max(0, prevAgent.energy - 0.1);
        let newHealth = prevAgent.health;

        // 1. Temperature-based health drain
        if (environment.temperature < -60) {
            newHealth -= 0.5; // Moderate cold
        }
        if (environment.temperature < -75) {
            newHealth -= 1.5; // Extreme cold
        }

        // 2. Radiation-based health drain
        if (environment.radiationLevel > 5.0) {
            newHealth -= 0.5; // Elevated radiation
        }
        if (environment.radiationLevel > 8.0) {
            newHealth -= 2.0; // Dangerous radiation
            if (Math.random() > 0.9) addLog("Heavy radiation detected! Suit shielding failing.", "danger");
        }

        // 3. Weather (Dust Storm) health drain
        if (environment.dustStormIntensity > 0.5) {
            newHealth -= 1.0; // High winds abrasion
        }
        if (environment.dustStormIntensity > 0.9) {
            newHealth -= 2.0; // Severe storm structural shearing
        }

        // 4. Depletion Rules (Old Rules)
        if (newOxy <= 0) {
            newHealth -= 5;
            if (prevAgent.oxygen > 0) addLog("CRITICAL: OXYGEN DEPLETED", 'danger', -50);
        }
        if (newEnergy <= 0) {
            newHealth -= 5;
            if (prevAgent.energy > 0) addLog("CRITICAL: ENERGY DEPLETED", 'danger', -50);
        }

        newHealth = Math.max(0, newHealth);

        if (newHealth <= 0 && prevAgent.health > 0) {
            setIsRunning(false);
            addLog("SIGNAL LOST: Agent expired due to environmental exposure.", 'danger', -100);
        }

        return {
          ...prevAgent,
          oxygen: newOxy,
          energy: newEnergy,
          health: newHealth
        };
      });
      
      setTotalReward(r => r + 0.1);

    }, 800);

    return () => clearInterval(interval);
  }, [isRunning, environment.temperature, environment.dustStormIntensity, environment.radiationLevel]);

  return {
    agent,
    environment,
    performAction,
    logs,
    totalReward,
    lastReward,
    isRunning,
    setIsRunning
  };
};