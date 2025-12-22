"""
Mars Survival Gymnasium Environment

A high-fidelity grid-world simulation for training autonomous rover agents
in extreme Martian conditions.
"""

from enum import IntEnum
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TerrainType(IntEnum):
    """Terrain types on the Mars grid."""
    FLAT = 0
    SANDY = 1
    ROCKY = 2
    CRATER = 3
    ICE = 4
    BASE = 5


class Action(IntEnum):
    """Available agent actions."""
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    MINE_ICE = 4
    USE_OXYGEN = 5
    CHARGE_BATTERY = 6


# Constants matching the TypeScript implementation
GRID_SIZE = 15
MAX_STATS = 100
TICK_HOURS = 0.5

# Movement costs per terrain
TERRAIN_COSTS = {
    TerrainType.FLAT: 1.0,
    TerrainType.SANDY: 2.0,
    TerrainType.ROCKY: 3.0,
    TerrainType.CRATER: 2.0,
    TerrainType.ICE: 1.0,
    TerrainType.BASE: 1.0,
}

# Reward values
REWARDS = {
    "survival_tick": 0.1,
    "mine_ice": 50.0,
    "use_consumable": 10.0,
    "move_flat": -1.0,
    "move_sandy": -2.0,
    "move_rocky": -3.0,
    "crater_entry": -10.0,
    "death": -100.0,
    "boundary_hit": -1.0,
    "invalid_action": -2.0,
}


class MarsSurvivalEnv(gym.Env):
    """
    Mars Survival RL Environment.
    
    A 15x15 grid world simulating Martian survival with resource management,
    environmental hazards, and autonomous rover navigation.
    
    Observation Space (12 dimensions):
        - Position (x, y): normalized [0, 1]
        - Health: normalized [0, 1]
        - Energy: normalized [0, 1]
        - Oxygen: normalized [0, 1]
        - Temperature: normalized [-100°C, 30°C] -> [0, 1]
        - Radiation: normalized [0, 10] -> [0, 1]
        - Dust storm intensity: [0, 1]
        - Time of day: normalized [0, 24] -> [0, 1]
        - Oxygen tanks in inventory: normalized [0, 5] -> [0, 1]
        - Recharge packs in inventory: normalized [0, 5] -> [0, 1]
        - Current terrain type: normalized [0, 5] -> [0, 1]
        - Ice collected: normalized [0, 20] -> [0, 1]
    
    Action Space (Discrete 7):
        0: Move North
        1: Move South
        2: Move East
        3: Move West
        4: Mine Ice
        5: Use Oxygen Tank
        6: Charge Battery (at base)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)
        
        # Initialize state (will be set in reset)
        self._grid: np.ndarray = None
        self._agent_pos: np.ndarray = None
        self._health: float = MAX_STATS
        self._energy: float = MAX_STATS
        self._oxygen: float = MAX_STATS
        self._body_temp: float = 37.0
        
        # Inventory
        self._ice: int = 0
        self._oxygen_tanks: int = 3
        self._recharge_packs: int = 2
        
        # Environment state
        self._sol: int = 1
        self._time_of_day: float = 8.0
        self._temperature: float = -30.0
        self._radiation: float = 1.2
        self._dust_storm: float = 0.0
        
        # Episode tracking
        self._step_count: int = 0
        self._total_reward: float = 0.0
        
        # Random number generator
        self._np_random: np.random.Generator = None

    def _generate_grid(self) -> np.ndarray:
        """Generate a random Mars terrain grid."""
        grid = np.full((GRID_SIZE, GRID_SIZE), TerrainType.FLAT, dtype=np.int8)
        
        # Place base at center
        center = GRID_SIZE // 2
        grid[center, center] = TerrainType.BASE
        
        # Randomly place other terrain types
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if y == center and x == center:
                    continue  # Skip base
                    
                rand = self._np_random.random()
                if rand > 0.92:
                    grid[y, x] = TerrainType.ICE
                elif rand > 0.80:
                    grid[y, x] = TerrainType.ROCKY
                elif rand > 0.70:
                    grid[y, x] = TerrainType.CRATER
                elif rand > 0.50:
                    grid[y, x] = TerrainType.SANDY
                # else: remains FLAT
        
        return grid

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector."""
        current_terrain = self._grid[self._agent_pos[1], self._agent_pos[0]]
        
        obs = np.array([
            self._agent_pos[0] / (GRID_SIZE - 1),  # x position
            self._agent_pos[1] / (GRID_SIZE - 1),  # y position
            self._health / MAX_STATS,               # health
            self._energy / MAX_STATS,               # energy
            self._oxygen / MAX_STATS,               # oxygen
            (self._temperature + 100) / 130,        # temp: [-100, 30] -> [0, 1]
            self._radiation / 10.0,                 # radiation: [0, 10] -> [0, 1]
            self._dust_storm,                       # dust: already [0, 1]
            self._time_of_day / 24.0,               # time: [0, 24] -> [0, 1]
            min(self._oxygen_tanks, 5) / 5.0,       # o2 tanks
            min(self._recharge_packs, 5) / 5.0,     # recharge packs
            current_terrain / 5.0,                  # terrain type
        ], dtype=np.float32)
        
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary information about the environment state."""
        return {
            "position": self._agent_pos.tolist(),
            "health": self._health,
            "energy": self._energy,
            "oxygen": self._oxygen,
            "temperature": self._temperature,
            "radiation": self._radiation,
            "dust_storm": self._dust_storm,
            "sol": self._sol,
            "time_of_day": self._time_of_day,
            "ice_collected": self._ice,
            "oxygen_tanks": self._oxygen_tanks,
            "recharge_packs": self._recharge_packs,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "grid": self._grid.tolist(),
            "current_terrain": int(self._grid[self._agent_pos[1], self._agent_pos[0]]),
        }

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)
        
        # Generate new grid
        self._grid = self._generate_grid()
        
        # Reset agent to base (center)
        center = GRID_SIZE // 2
        self._agent_pos = np.array([center, center], dtype=np.int32)
        
        # Reset vitals
        self._health = MAX_STATS
        self._energy = MAX_STATS
        self._oxygen = MAX_STATS
        self._body_temp = 37.0
        
        # Reset inventory
        self._ice = 0
        self._oxygen_tanks = 3
        self._recharge_packs = 2
        
        # Reset environment
        self._sol = 1
        self._time_of_day = 8.0
        self._temperature = -30.0
        self._radiation = 1.2
        self._dust_storm = 0.0
        
        # Reset tracking
        self._step_count = 0
        self._total_reward = 0.0
        
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self._step_count += 1
        reward = REWARDS["survival_tick"]  # Base survival reward
        terminated = False
        truncated = False
        
        # Execute action
        action_reward = self._execute_action(action)
        reward += action_reward
        
        # Update environment (time, temperature, radiation, dust)
        self._update_environment()
        
        # Apply environmental effects to agent
        env_damage = self._apply_environmental_effects()
        
        # Apply depletion (oxygen/energy decay)
        self._apply_depletion()
        
        # Check death conditions
        if self._health <= 0:
            self._health = 0
            reward += REWARDS["death"]
            terminated = True
        
        self._total_reward += reward
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> float:
        """Execute the given action and return the reward."""
        reward = 0.0
        
        if action == Action.MOVE_NORTH:
            reward = self._move(0, -1)
        elif action == Action.MOVE_SOUTH:
            reward = self._move(0, 1)
        elif action == Action.MOVE_EAST:
            reward = self._move(1, 0)
        elif action == Action.MOVE_WEST:
            reward = self._move(-1, 0)
        elif action == Action.MINE_ICE:
            reward = self._mine_ice()
        elif action == Action.USE_OXYGEN:
            reward = self._use_oxygen_tank()
        elif action == Action.CHARGE_BATTERY:
            reward = self._charge_battery()
        
        return reward

    def _move(self, dx: int, dy: int) -> float:
        """Move agent in the given direction."""
        new_x = self._agent_pos[0] + dx
        new_y = self._agent_pos[1] + dy
        
        # Check boundaries
        if new_x < 0 or new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE:
            return REWARDS["boundary_hit"]
        
        # Get terrain at new position
        terrain = TerrainType(self._grid[new_y, new_x])
        energy_cost = TERRAIN_COSTS[terrain]
        
        # Move agent
        self._agent_pos[0] = new_x
        self._agent_pos[1] = new_y
        self._energy = max(0, self._energy - energy_cost)
        
        # Calculate reward based on terrain
        if terrain == TerrainType.CRATER:
            # 50% chance of damage
            if self._np_random.random() > 0.5:
                self._health = max(0, self._health - 5)
            return REWARDS["crater_entry"]
        elif terrain == TerrainType.ROCKY:
            return REWARDS["move_rocky"]
        elif terrain == TerrainType.SANDY:
            return REWARDS["move_sandy"]
        else:
            return REWARDS["move_flat"]

    def _mine_ice(self) -> float:
        """Attempt to mine ice at current position."""
        x, y = self._agent_pos
        
        if self._grid[y, x] == TerrainType.ICE:
            # Mine the ice
            self._ice += 1
            self._energy = max(0, self._energy - 10)
            # Deplete the ice deposit
            self._grid[y, x] = TerrainType.FLAT
            return REWARDS["mine_ice"]
        else:
            return REWARDS["invalid_action"]

    def _use_oxygen_tank(self) -> float:
        """Use an oxygen tank from inventory."""
        if self._oxygen_tanks > 0:
            self._oxygen_tanks -= 1
            self._oxygen = min(MAX_STATS, self._oxygen + 50)
            self._health = MAX_STATS  # Critical: restores health
            return REWARDS["use_consumable"]
        else:
            return REWARDS["invalid_action"]

    def _charge_battery(self) -> float:
        """Charge battery using recharge pack (only at base)."""
        x, y = self._agent_pos
        
        # Must be at base
        if self._grid[y, x] != TerrainType.BASE:
            return REWARDS["invalid_action"]
        
        if self._recharge_packs > 0:
            self._recharge_packs -= 1
            self._energy = min(MAX_STATS, self._energy + 50)
            self._health = MAX_STATS  # Critical: restores health
            return REWARDS["use_consumable"]
        else:
            return REWARDS["invalid_action"]

    def _update_environment(self) -> None:
        """Update environmental conditions (sol cycle)."""
        # Advance time
        self._time_of_day += TICK_HOURS
        if self._time_of_day >= 24:
            self._time_of_day = 0
            self._sol += 1
        
        # Update temperature based on time
        is_night = self._time_of_day < 6 or self._time_of_day > 20
        if is_night:
            self._temperature = -80 + self._np_random.random() * 5
        else:
            self._temperature = -20 + self._np_random.random() * 5
        
        # Fluctuating radiation
        self._radiation += self._np_random.random() * 0.4 - 0.2
        if self._np_random.random() > 0.98:
            self._radiation += 4.0  # Radiation burst
        self._radiation = np.clip(self._radiation, 0.1, 10.0)
        
        # Dust storm dynamics
        if self._np_random.random() > 0.95:
            self._dust_storm = min(1.0, self._dust_storm + 0.2)
        else:
            self._dust_storm = max(0.0, self._dust_storm - 0.05)

    def _apply_environmental_effects(self) -> float:
        """Apply environmental damage to agent."""
        damage = 0.0
        
        # Temperature effects
        if self._temperature < -60:
            damage += 0.5
        if self._temperature < -75:
            damage += 1.5
        
        # Radiation effects
        if self._radiation > 5.0:
            damage += 0.5
        if self._radiation > 8.0:
            damage += 2.0
        
        # Dust storm effects
        if self._dust_storm > 0.5:
            damage += 1.0
        if self._dust_storm > 0.9:
            damage += 2.0
        
        self._health = max(0, self._health - damage)
        return damage

    def _apply_depletion(self) -> None:
        """Apply oxygen and energy decay, with penalties for depletion."""
        # Decay
        self._oxygen = max(0, self._oxygen - 0.5)
        self._energy = max(0, self._energy - 0.1)
        
        # Depletion penalties
        if self._oxygen <= 0:
            self._health = max(0, self._health - 5)
        if self._energy <= 0:
            self._health = max(0, self._health - 5)

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """Render as ASCII art."""
        terrain_chars = {
            TerrainType.FLAT: ".",
            TerrainType.SANDY: "~",
            TerrainType.ROCKY: "#",
            TerrainType.CRATER: "O",
            TerrainType.ICE: "*",
            TerrainType.BASE: "B",
        }
        
        lines = []
        lines.append(f"Sol {self._sol} | Time: {self._time_of_day:.1f}h | Reward: {self._total_reward:.1f}")
        lines.append(f"HP: {self._health:.0f} | E: {self._energy:.0f} | O2: {self._oxygen:.0f}")
        lines.append(f"Temp: {self._temperature:.0f}°C | Rad: {self._radiation:.1f}mSv | Storm: {self._dust_storm:.1%}")
        lines.append(f"Inv: Ice={self._ice} O2Tanks={self._oxygen_tanks} Packs={self._recharge_packs}")
        lines.append("-" * (GRID_SIZE + 2))
        
        for y in range(GRID_SIZE):
            row = "|"
            for x in range(GRID_SIZE):
                if self._agent_pos[0] == x and self._agent_pos[1] == y:
                    row += "A"
                else:
                    row += terrain_chars[TerrainType(self._grid[y, x])]
            row += "|"
            lines.append(row)
        
        lines.append("-" * (GRID_SIZE + 2))
        
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
        return output

    def get_state_dict(self) -> dict:
        """Get full state as dictionary for WebSocket serialization."""
        return {
            "agent": {
                "position": {"x": int(self._agent_pos[0]), "y": int(self._agent_pos[1])},
                "health": float(self._health),
                "energy": float(self._energy),
                "oxygen": float(self._oxygen),
                "bodyTemp": float(self._body_temp),
                "inventory": {
                    "ice": int(self._ice),
                    "samples": 0,
                    "oxygenTanks": int(self._oxygen_tanks),
                    "rechargePacks": int(self._recharge_packs),
                },
            },
            "environment": {
                "sol": int(self._sol),
                "timeOfDay": float(self._time_of_day),
                "temperature": float(self._temperature),
                "radiationLevel": float(self._radiation),
                "dustStormIntensity": float(self._dust_storm),
                "grid": [[int(cell) for cell in row] for row in self._grid],
                "solarPanels": [],
            },
            "step": int(self._step_count),
            "totalReward": float(self._total_reward),
        }
