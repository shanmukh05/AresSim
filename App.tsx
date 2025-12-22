import React, { useState } from 'react';
import { useMarsSurvival } from './hooks/useMarsSurvival';
import { useRemoteMarsSurvival } from './hooks/useRemoteMarsSurvival';
import { MarsMap } from './components/MarsMap';
import { StatusMonitor } from './components/StatusMonitor';
import { ControlDeck } from './components/ControlDeck';
import { EventLog } from './components/EventLog';
import { EnvironmentStats } from './components/EnvironmentStats';
import { RewardPanel } from './components/RewardPanel';
import { Activity, Terminal, Wifi, WifiOff, Cpu, User, RefreshCw } from 'lucide-react';

type SimMode = 'local' | 'remote';

const App: React.FC = () => {
    const [simMode, setSimMode] = useState<SimMode>('local');

    // Local simulation hook
    const localState = useMarsSurvival();

    // Remote simulation hook
    const remoteState = useRemoteMarsSurvival();

    // Use the appropriate state based on mode
    const state = simMode === 'local' ? localState : remoteState;
    const { agent, environment, performAction, logs, totalReward, lastReward, isRunning, setIsRunning } = state;

    // Remote-specific properties
    const connectionStatus = simMode === 'remote' ? remoteState.connectionStatus : 'disconnected';
    const isConnected = simMode === 'remote' ? remoteState.isConnected : false;
    const remoteMode = simMode === 'remote' ? remoteState.remoteMode : 'manual';
    const stepCount = simMode === 'remote' ? remoteState.stepCount : 0;

    const handleModeSwitch = (newMode: SimMode) => {
        if (newMode === 'remote' && !remoteState.isConnected) {
            remoteState.connect();
        } else if (newMode === 'local' && remoteState.isConnected) {
            remoteState.disconnect();
        }
        setSimMode(newMode);
    };

    const getConnectionColor = () => {
        switch (connectionStatus) {
            case 'connected': return 'text-green-400';
            case 'connecting': return 'text-yellow-400 animate-pulse';
            case 'error': return 'text-red-400';
            default: return 'text-slate-500';
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-slate-200 p-4 md:p-8 font-sans selection:bg-orange-500/30">

            {/* Header */}
            <header className="mb-6 flex flex-col md:flex-row justify-between items-start md:items-end border-b border-slate-800 pb-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
                        <Terminal className="text-orange-500" />
                        MARS RL <span className="text-slate-600 font-light">ENVIRONMENT VISUALIZER</span>
                    </h1>
                    <p className="text-slate-400 text-sm mt-1 max-w-xl">
                        Real-time dashboard for autonomous rover operations. Monitor agent rewards, state vectors, and environmental hazards.
                    </p>
                </div>
                <div className="mt-4 md:mt-0 flex gap-6 items-end">
                    {/* Mode Toggle */}
                    <div className="flex flex-col items-end gap-1">
                        <div className="text-xs text-slate-500 uppercase tracking-widest">Simulation Mode</div>
                        <div className="flex gap-1 bg-slate-900 rounded-lg p-1 border border-slate-700">
                            <button
                                onClick={() => handleModeSwitch('local')}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm font-medium transition-all ${simMode === 'local'
                                    ? 'bg-orange-500 text-white'
                                    : 'text-slate-400 hover:text-slate-200'
                                    }`}
                            >
                                <User size={14} />
                                Local
                            </button>
                            <button
                                onClick={() => handleModeSwitch('remote')}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm font-medium transition-all ${simMode === 'remote'
                                    ? 'bg-cyan-500 text-white'
                                    : 'text-slate-400 hover:text-slate-200'
                                    }`}
                            >
                                <Cpu size={14} />
                                Remote
                            </button>
                        </div>
                    </div>

                    {/* Connection Status (Remote Mode) */}
                    {simMode === 'remote' && (
                        <div className="flex flex-col items-end gap-1">
                            <div className="text-xs text-slate-500 uppercase tracking-widest">Python Env</div>
                            <div className={`flex items-center gap-2 font-mono text-sm ${getConnectionColor()}`}>
                                {isConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
                                {connectionStatus.toUpperCase()}
                                {remoteMode === 'agent' && (
                                    <span className="ml-2 px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-xs">
                                        RL AGENT
                                    </span>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Simulation Status */}
                    <div className="text-right">
                        <div className="text-xs text-slate-500 uppercase tracking-widest">Simulation Status</div>
                        <div className={`font-mono text-lg ${isRunning ? 'text-green-400 animate-pulse' : 'text-yellow-500'}`}>
                            {isRunning ? 'RUNNING' : 'PAUSED'}
                            {simMode === 'remote' && stepCount > 0 && (
                                <span className="text-xs text-slate-500 ml-2">Step {stepCount}</span>
                            )}
                        </div>
                    </div>
                </div>
            </header>

            {/* Remote Mode Banner */}
            {simMode === 'remote' && !isConnected && (
                <div className="mb-4 p-4 bg-cyan-900/20 border border-cyan-800 rounded-lg flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Cpu className="text-cyan-400" size={24} />
                        <div>
                            <div className="font-medium text-cyan-200">Remote Mode Active</div>
                            <div className="text-sm text-cyan-400/80">
                                Start the Python server: <code className="bg-slate-800 px-2 py-0.5 rounded">python -m gym_mars.server</code>
                            </div>
                        </div>
                    </div>
                    <button
                        onClick={() => remoteState.connect()}
                        className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors"
                    >
                        <RefreshCw size={16} />
                        Connect
                    </button>
                </div>
            )}

            {/* Top Environment Bar */}
            <EnvironmentStats environment={environment} />

            {/* Main Grid Layout - Fixed Height on Large Screens */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:h-[600px] h-auto">

                {/* Left Column: Stats & Controls */}
                <div className="lg:col-span-3 flex flex-col gap-6 lg:h-full lg:overflow-y-auto pr-1">
                    <StatusMonitor agent={agent} />
                    <ControlDeck
                        onAction={performAction}
                        isRunning={isRunning}
                        toggleRun={() => setIsRunning(!isRunning)}
                        disabled={simMode === 'remote' && !isConnected}
                    />
                    {/* Observation Debug View */}
                    <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded p-2 overflow-auto font-mono text-[10px] text-slate-500 min-h-[100px]">
                        <div className="flex items-center justify-between mb-1">
                            <span>OBSERVATION VECTOR (PARTIAL):</span>
                            {simMode === 'remote' && (
                                <span className={`px-1.5 py-0.5 rounded text-[8px] ${isConnected ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-700 text-slate-400'}`}>
                                    {isConnected ? 'LIVE' : 'OFFLINE'}
                                </span>
                            )}
                        </div>
                        <pre>{JSON.stringify({
                            pos: [agent.position.x, agent.position.y],
                            stats: [agent.health, agent.energy, agent.oxygen],
                            env: [environment.temperature, environment.radiationLevel]
                        }, null, 1)}</pre>
                    </div>
                </div>

                {/* Center Column: Map */}
                <div className="lg:col-span-6 flex flex-col lg:h-full min-h-[400px]">
                    <div className="flex-1 bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col items-center justify-center relative overflow-hidden">
                        <MarsMap grid={environment.grid} agent={agent} />

                        {/* Decorative HUD Elements */}
                        <div className="absolute top-4 right-4 text-orange-500/20 pointer-events-none">
                            <Activity size={48} />
                        </div>

                        {/* Remote mode indicator */}
                        {/* {simMode === 'remote' && isConnected && (
                            <div className="absolute top-0 left-1/2 -translate-x-1/2 flex items-center gap-2 px-2 py-1 bg-cyan-900/50 border border-cyan-700/50 rounded text-xs text-cyan-400">
                                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                                Python Environment
                            </div>
                        )} */}
                    </div>
                </div>

                {/* Right Column: Rewards & Logs */}
                <div className="lg:col-span-3 flex flex-col gap-4 lg:h-full h-[500px]">
                    <RewardPanel totalReward={totalReward} lastReward={lastReward} />

                    {/* EventLog Container: Strictly constrained to take remaining space using relative/absolute trick */}
                    <div className="flex-1 min-h-0 relative rounded-lg border border-slate-700 bg-slate-950 overflow-hidden">
                        <div className="absolute inset-0 flex flex-col">
                            <EventLog logs={logs} />
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default App;