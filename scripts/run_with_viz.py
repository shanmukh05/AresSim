#!/usr/bin/env python3
"""
Run a trained agent with WebSocket visualization.

Loads a trained model and runs it in the environment while
broadcasting state to the React UI via WebSocket.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError("websockets package required. Install with: pip install websockets")

try:
    from stable_baselines3 import PPO
except ImportError:
    raise ImportError(
        "stable-baselines3 required. "
        "Install with: pip install 'gym-mars[train]'"
    )

import gymnasium as gym
import numpy as np

# Ensure gym_mars is registered
import gym_mars
from gym_mars.envs import MarsSurvivalEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs a trained agent with WebSocket visualization."""
    
    def __init__(
        self,
        model_path: str,
        host: str = "localhost",
        port: int = 8765,
        step_delay: float = 0.5,
        deterministic: bool = True,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.step_delay = step_delay
        self.deterministic = deterministic
        
        self.model: Optional[PPO] = None
        self.env: Optional[MarsSurvivalEnv] = None
        self.clients: set[WebSocketServerProtocol] = set()
        self._running = False
        self._paused = True
        
    async def start(self):
        """Start the agent runner with WebSocket server."""
        # Load model
        logger.info(f"Loading model from: {self.model_path}")
        self.model = PPO.load(self.model_path)
        logger.info("Model loaded successfully")
        
        # Create environment
        self.env = MarsSurvivalEnv()
        obs, info = self.env.reset()
        logger.info("Environment initialized")
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        
        async with websockets.serve(self._handle_client, self.host, self.port):
            self._running = True
            logger.info("Server started. Waiting for connections...")
            logger.info("Agent is PAUSED. Send {'type': 'start'} to begin.")
            
            # Run agent loop
            await self._agent_loop()
    
    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle WebSocket client connections."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total: {len(self.clients)}")
        
        try:
            # Send initial state
            await self._send_state(websocket)
            
            async for message in websocket:
                await self._process_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected. Remaining: {len(self.clients)}")
    
    async def _process_message(self, websocket: WebSocketServerProtocol, message: str):
        """Process incoming messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "get_state":
                await self._send_state(websocket)
            elif msg_type == "start":
                self._paused = False
                logger.info("Agent STARTED")
            elif msg_type == "pause":
                self._paused = True
                logger.info("Agent PAUSED")
            elif msg_type == "reset":
                obs, info = self.env.reset(seed=data.get("seed"))
                self._paused = True
                await self._broadcast_state()
                logger.info("Environment RESET")
            elif msg_type == "set_delay":
                self.step_delay = float(data.get("delay", 0.5))
                logger.info(f"Step delay set to: {self.step_delay}s")
                
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
    
    async def _agent_loop(self):
        """Main agent loop that runs the trained policy."""
        obs, info = self.env.reset()
        episode = 0
        step = 0
        episode_reward = 0.0
        
        while self._running:
            if self._paused or not self.clients:
                await asyncio.sleep(0.1)
                continue
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            episode_reward += reward
            step += 1
            
            # Broadcast state
            await self._broadcast_state(action=int(action), reward=float(reward))
            
            # Check episode end
            if terminated or truncated:
                episode += 1
                logger.info(f"Episode {episode} finished. Steps: {step}, Reward: {episode_reward:.1f}")
                
                # Reset
                obs, info = self.env.reset()
                step = 0
                episode_reward = 0.0
                
                # Auto-pause between episodes
                self._paused = True
                await self._broadcast_state()
            
            # Delay for visualization
            await asyncio.sleep(self.step_delay)
    
    async def _send_state(self, websocket: WebSocketServerProtocol):
        """Send current state to a client."""
        state = self.env.get_state_dict()
        state["type"] = "state"
        state["mode"] = "agent"
        state["paused"] = self._paused
        state["modelPath"] = self.model_path
        await websocket.send(json.dumps(state))
    
    async def _broadcast_state(self, action: Optional[int] = None, reward: Optional[float] = None):
        """Broadcast state to all clients."""
        if not self.clients:
            return
            
        state = self.env.get_state_dict()
        state["type"] = "state"
        state["mode"] = "agent"
        state["paused"] = self._paused
        if action is not None:
            state["lastAction"] = action
        if reward is not None:
            state["lastReward"] = reward
        
        # Debug: log position being sent
        pos = state["agent"]["position"]
        logger.debug(f"Broadcasting position: ({pos['x']}, {pos['y']})")
            
        message = json.dumps(state)
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run trained agent with WebSocket visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="WebSocket server host"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="WebSocket server port"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between steps (seconds)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions (default: deterministic)"
    )
    
    args = parser.parse_args()
    
    runner = AgentRunner(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        step_delay=args.delay,
        deterministic=not args.stochastic,
    )
    
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
