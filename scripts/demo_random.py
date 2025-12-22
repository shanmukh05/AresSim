#!/usr/bin/env python3
"""
Demo script that runs random actions to demonstrate UI visualization.

Unlike trained agents that may not move, this uses random movement
actions to verify the UI position updates work correctly.
"""

import asyncio
import json
import logging
import random
from typing import Optional

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError("websockets package required")

import gym_mars
from gym_mars.envs import MarsSurvivalEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomAgentDemo:
    """Runs random movement actions for demo purposes."""
    
    def __init__(self, host: str = "localhost", port: int = 8765, step_delay: float = 0.3):
        self.host = host
        self.port = port
        self.step_delay = step_delay
        self.env: Optional[MarsSurvivalEnv] = None
        self.clients: set = set()
        self._running = False
        self._paused = True
        
    async def start(self):
        self.env = MarsSurvivalEnv()
        self.env.reset()
        logger.info("Environment initialized")
        
        logger.info(f"Starting demo server on ws://{self.host}:{self.port}")
        
        async with websockets.serve(self._handle_client, self.host, self.port):
            self._running = True
            logger.info("Server ready. Switch to Remote mode in UI, then click RESUME SIM.")
            await self._agent_loop()
    
    async def _handle_client(self, websocket):
        self.clients.add(websocket)
        logger.info(f"Client connected. Total: {len(self.clients)}")
        
        try:
            await self._send_state(websocket)
            async for message in websocket:
                await self._process_message(message)
        except:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected. Remaining: {len(self.clients)}")
    
    async def _process_message(self, message: str):
        try:
            data = json.loads(message)
            if data.get("type") == "start":
                self._paused = False
                logger.info("Demo STARTED")
            elif data.get("type") == "pause":
                self._paused = True
                logger.info("Demo PAUSED")
            elif data.get("type") == "reset":
                self.env.reset()
                logger.info("Environment RESET")
                await self._broadcast_state()
        except Exception as e:
            logger.exception(f"Error: {e}")
    
    async def _agent_loop(self):
        step = 0
        
        while self._running:
            if self._paused or not self.clients:
                await asyncio.sleep(0.1)
                continue
            
            # Use primarily movement actions (0-3) with occasional other actions
            if random.random() < 0.8:
                action = random.randint(0, 3)  # Movement
            else:
                action = random.randint(4, 6)  # Other actions
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            step += 1
            
            # Log position for debugging
            pos = self.env._agent_pos
            logger.info(f"Step {step}: action={action}, pos=({pos[0]}, {pos[1]})")
            
            await self._broadcast_state(action=action, reward=float(reward))
            
            if terminated or truncated:
                logger.info(f"Episode finished at step {step}")
                self.env.reset()
                step = 0
                self._paused = True
                await self._broadcast_state()
            
            await asyncio.sleep(self.step_delay)
    
    async def _send_state(self, websocket):
        state = self.env.get_state_dict()
        state["type"] = "state"
        state["mode"] = "agent"
        state["paused"] = self._paused
        await websocket.send(json.dumps(state))
    
    async def _broadcast_state(self, action: Optional[int] = None, reward: Optional[float] = None):
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
        
        message = json.dumps(state)
        await asyncio.gather(*[c.send(message) for c in self.clients], return_exceptions=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Random agent demo for UI testing")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between steps")
    args = parser.parse_args()
    
    demo = RandomAgentDemo(step_delay=args.delay)
    try:
        asyncio.run(demo.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
