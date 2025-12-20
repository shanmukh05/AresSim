import React from 'react';
import { useMarsSurvival } from './hooks/useMarsSurvival';
import { MarsMap } from './components/MarsMap';
import { StatusMonitor } from './components/StatusMonitor';
import { ControlDeck } from './components/ControlDeck';
import { EventLog } from './components/EventLog';
import { EnvironmentStats } from './components/EnvironmentStats';
import { RewardPanel } from './components/RewardPanel';
import { Activity, Terminal } from 'lucide-react';

const App: React.FC = () => {
  const { 
    agent, 
    environment, 
    performAction, 
    logs, 
    totalReward,
    lastReward,
    isRunning, 
    setIsRunning 
  } = useMarsSurvival();

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
        <div className="mt-4 md:mt-0 flex gap-4">
            <div className="text-right">
                <div className="text-xs text-slate-500 uppercase tracking-widest">Simulation Status</div>
                <div className={`font-mono text-lg ${isRunning ? 'text-green-400 animate-pulse' : 'text-yellow-500'}`}>
                    {isRunning ? 'RUNNING' : 'PAUSED'}
                </div>
            </div>
        </div>
      </header>

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
            />
            {/* Observation Debug View */}
            <div className="flex-1 bg-slate-900/50 border border-slate-800 rounded p-2 overflow-auto font-mono text-[10px] text-slate-500 min-h-[100px]">
                <div>OBSERVATION VECTOR (PARTIAL):</div>
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