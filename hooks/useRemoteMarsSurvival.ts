import { useState, useEffect, useCallback, useRef } from 'react';
import {
    AgentState,
    EnvironmentState,
    TerrainType,
    AgentAction,
    LogEntry,
    RemoteStateMessage,
    TERRAIN_INT_MAP
} from '../types';
import { useWebSocket, ConnectionStatus } from './useWebSocket';
import { GRID_SIZE } from '../constants';

const WS_URL = 'ws://localhost:8765';

// Action mapping (matches Python Action enum)
const ACTION_MAP: Record<AgentAction, number> = {
    [AgentAction.MOVE_NORTH]: 0,
    [AgentAction.MOVE_SOUTH]: 1,
    [AgentAction.MOVE_EAST]: 2,
    [AgentAction.MOVE_WEST]: 3,
    [AgentAction.MINE_ICE]: 4,
    [AgentAction.USE_OXYGEN_TANK]: 5,
    [AgentAction.CHARGE_BATTERY]: 6,
    [AgentAction.REPAIR_SUIT]: 5, // Map to USE_OXYGEN for now
    [AgentAction.BUILD_SOLAR]: 6, // Map to CHARGE for now
    [AgentAction.IDLE]: 0,       // Map to MOVE_NORTH (no-op alternative)
};

const ACTION_NAMES: Record<number, string> = {
    0: 'Move North',
    1: 'Move South',
    2: 'Move East',
    3: 'Move West',
    4: 'Mine Ice',
    5: 'Use O2 Tank',
    6: 'Charge Battery',
};

/**
 * Hook for connecting to a remote Mars Survival environment via WebSocket.
 */
export const useRemoteMarsSurvival = () => {
    const [agent, setAgent] = useState<AgentState>({
        position: { x: Math.floor(GRID_SIZE / 2), y: Math.floor(GRID_SIZE / 2) },
        health: 100,
        energy: 100,
        oxygen: 100,
        bodyTemp: 37,
        inventory: { ice: 0, samples: 0, oxygenTanks: 3, rechargePacks: 2 },
    });

    const [environment, setEnvironment] = useState<EnvironmentState>({
        sol: 1,
        timeOfDay: 8.0,
        temperature: -30,
        radiationLevel: 1.2,
        dustStormIntensity: 0,
        grid: Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(TerrainType.FLAT)),
        solarPanels: [],
    });

    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [totalReward, setTotalReward] = useState(0);
    const [lastReward, setLastReward] = useState<{ value: number; source: string } | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [remoteMode, setRemoteMode] = useState<'manual' | 'agent'>('manual');
    const [isPaused, setIsPaused] = useState(true);
    const [stepCount, setStepCount] = useState(0);

    const addLog = useCallback((message: string, type: LogEntry['type'] = 'info', reward: number = 0) => {
        const newLog: LogEntry = {
            id: Math.random().toString(36).substring(7),
            timestamp: new Date().toLocaleTimeString(),
            message,
            type,
            reward
        };
        setLogs(prev => [newLog, ...prev].slice(0, 50));
    }, []);

    const handleMessage = useCallback((data: RemoteStateMessage) => {
        if (data.type === 'state') {
            // Debug: log position updates
            console.log('Received state update, position:', data.agent.position);

            // Convert remote state to local format - ensure new object references
            setAgent(prev => ({
                ...prev,
                position: { x: data.agent.position.x, y: data.agent.position.y },
                health: data.agent.health,
                energy: data.agent.energy,
                oxygen: data.agent.oxygen,
                bodyTemp: data.agent.bodyTemp,
                inventory: { ...data.agent.inventory },
            }));

            // Convert grid from integers to TerrainType enums
            const convertedGrid = data.environment.grid.map(row =>
                row.map(cell => TERRAIN_INT_MAP[cell] || TerrainType.FLAT)
            );

            setEnvironment(prev => ({
                ...prev,
                sol: data.environment.sol,
                timeOfDay: data.environment.timeOfDay,
                temperature: data.environment.temperature,
                radiationLevel: data.environment.radiationLevel,
                dustStormIntensity: data.environment.dustStormIntensity,
                grid: convertedGrid,
                solarPanels: data.environment.solarPanels || [],
            }));

            setTotalReward(data.totalReward);
            setStepCount(data.step);
            setRemoteMode(data.mode);
            setIsPaused(data.paused ?? true);

            // Log last action if present
            if (data.lastAction !== undefined && data.lastReward !== undefined) {
                const actionName = ACTION_NAMES[data.lastAction] || `Action ${data.lastAction}`;
                addLog(`Agent: ${actionName}`, data.lastReward >= 0 ? 'success' : 'warning', data.lastReward);
                setLastReward({ value: data.lastReward, source: actionName });
            }
        }
    }, [addLog]);

    const { status, connect, disconnect, send, isConnected } = useWebSocket({
        url: WS_URL,
        autoConnect: false,
        onMessage: handleMessage as (data: unknown) => void,
        onConnect: () => addLog('Connected to Python environment', 'success'),
        onDisconnect: () => addLog('Disconnected from Python environment', 'warning'),
        onError: () => addLog('WebSocket error', 'danger'),
    });

    const performAction = useCallback((action: AgentAction) => {
        if (!isConnected) return;

        const actionInt = ACTION_MAP[action];
        send({ type: 'step', action: actionInt });
    }, [isConnected, send]);

    const reset = useCallback(() => {
        if (!isConnected) return;
        send({ type: 'reset' });
        setLogs([]);
        addLog('Environment reset', 'info');
    }, [isConnected, send, addLog]);

    const startAgent = useCallback(() => {
        if (!isConnected) return;
        send({ type: 'start' });
        setIsRunning(true);
        addLog('Agent started', 'success');
    }, [isConnected, send, addLog]);

    const pauseAgent = useCallback(() => {
        if (!isConnected) return;
        send({ type: 'pause' });
        setIsRunning(false);
        addLog('Agent paused', 'info');
    }, [isConnected, send, addLog]);

    const toggleRun = useCallback(() => {
        if (isPaused) {
            startAgent();
        } else {
            pauseAgent();
        }
    }, [isPaused, startAgent, pauseAgent]);

    return {
        agent,
        environment,
        performAction,
        logs,
        totalReward,
        lastReward,
        isRunning: !isPaused,
        setIsRunning: (running: boolean) => running ? startAgent() : pauseAgent(),
        // Remote-specific
        connectionStatus: status,
        isConnected,
        connect,
        disconnect,
        reset,
        remoteMode,
        stepCount,
    };
};
