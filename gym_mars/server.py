"""
WebSocket Server for Mars Survival Environment.

Bridges the Python Gymnasium environment with the React visualization UI,
enabling real-time state synchronization and remote control.
"""

import asyncio
import json
import logging
from enum import Enum
from typing import Optional

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    raise ImportError("websockets package required. Install with: pip install websockets")

from gym_mars.envs import MarsSurvivalEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    GET_STATE = "get_state"
    STEP = "step"
    RESET = "reset"
    SET_MODE = "set_mode"
    ERROR = "error"
    STATE = "state"
    STEP_RESULT = "step_result"


class ControlMode(str, Enum):
    """Environment control modes."""
    MANUAL = "manual"  # UI controls the environment
    AGENT = "agent"    # RL agent controls the environment


class EnvServer:
    """
    WebSocket server that wraps the Mars Survival environment.
    
    Supports two modes:
    - MANUAL: Actions come from UI button clicks
    - AGENT: Actions come from a connected RL agent
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.env: Optional[MarsSurvivalEnv] = None
        self.mode = ControlMode.MANUAL
        self.clients: set[WebSocketServerProtocol] = set()
        self._running = False
        
    async def start(self):
        """Start the WebSocket server."""
        self.env = MarsSurvivalEnv()
        self.env.reset()
        
        logger.info(f"Starting Mars Survival Environment Server on ws://{self.host}:{self.port}")
        
        async with websockets.serve(self._handle_client, self.host, self.port):
            self._running = True
            logger.info("Server started. Waiting for connections...")
            await asyncio.Future()  # Run forever
    
    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new WebSocket connection."""
        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
        
        try:
            # Send initial state
            await self._send_state(websocket)
            
            async for message in websocket:
                await self._process_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client removed. Remaining clients: {len(self.clients)}")
    
    async def _process_message(self, websocket: WebSocketServerProtocol, message: str):
        """Process an incoming message from a client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == MessageType.GET_STATE:
                await self._send_state(websocket)
                
            elif msg_type == MessageType.RESET:
                seed = data.get("seed")
                obs, info = self.env.reset(seed=seed)
                await self._broadcast_state()
                logger.info(f"Environment reset (seed={seed})")
                
            elif msg_type == MessageType.STEP:
                action = data.get("action")
                if action is not None:
                    result = await self._step(int(action))
                    await websocket.send(json.dumps({
                        "type": MessageType.STEP_RESULT,
                        "observation": result["observation"],
                        "reward": result["reward"],
                        "terminated": result["terminated"],
                        "truncated": result["truncated"],
                        "info": result["info"],
                    }))
                    # Broadcast state to all clients
                    await self._broadcast_state()
                else:
                    await self._send_error(websocket, "Missing 'action' field")
                    
            elif msg_type == MessageType.SET_MODE:
                mode = data.get("mode")
                if mode in [m.value for m in ControlMode]:
                    self.mode = ControlMode(mode)
                    logger.info(f"Control mode set to: {self.mode.value}")
                    await self._broadcast_state()
                else:
                    await self._send_error(websocket, f"Invalid mode: {mode}")
                    
            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await self._send_error(websocket, str(e))
    
    async def _step(self, action: int) -> dict:
        """Execute a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        return {
            "observation": obs.tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
    
    async def _send_state(self, websocket: WebSocketServerProtocol):
        """Send current environment state to a client."""
        state = self.env.get_state_dict()
        state["type"] = MessageType.STATE
        state["mode"] = self.mode.value
        await websocket.send(json.dumps(state))
    
    async def _broadcast_state(self):
        """Broadcast current state to all connected clients."""
        if not self.clients:
            return
            
        state = self.env.get_state_dict()
        state["type"] = MessageType.STATE
        state["mode"] = self.mode.value
        message = json.dumps(state)
        
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error: str):
        """Send an error message to a client."""
        await websocket.send(json.dumps({
            "type": MessageType.ERROR,
            "error": error,
        }))
        logger.warning(f"Sent error to client: {error}")


def run_server(host: str = "localhost", port: int = 8765):
    """Run the WebSocket server (blocking)."""
    server = EnvServer(host=host, port=port)
    asyncio.run(server.start())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mars Survival Environment WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)
