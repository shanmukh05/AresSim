import React from 'react';
import { AgentState } from '../types';
import { Heart, Zap, Wind, Shield, Thermometer, Box } from 'lucide-react';

interface StatusMonitorProps {
  agent: AgentState;
}

const ProgressBar = ({ value, color, label, icon: Icon }: { value: number; color: string; label: string; icon: any }) => (
  <div className="mb-4">
    <div className="flex justify-between items-center mb-1 text-xs uppercase tracking-wider text-slate-400">
      <div className="flex items-center gap-2">
        <Icon size={14} />
        {label}
      </div>
      <span className="font-mono">{value.toFixed(1)}%</span>
    </div>
    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
      <div 
        className={`h-full ${color} transition-all duration-500 ease-out`}
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  </div>
);

export const StatusMonitor: React.FC<StatusMonitorProps> = ({ agent }) => {
  return (
    <div className="bg-slate-900/80 border border-slate-700 p-4 rounded-lg backdrop-blur-sm">
      <h3 className="text-sm font-bold text-slate-200 mb-4 border-b border-slate-700 pb-2 flex items-center gap-2">
        <span className="w-2 h-2 bg-green-500 rounded-full animate-ping"></span>
        AGENT VITALS
      </h3>
      
      <ProgressBar 
        label="Health" 
        value={agent.health} 
        color={agent.health < 30 ? "bg-red-500" : "bg-green-500"} 
        icon={Heart}
      />
      <ProgressBar 
        label="Energy" 
        value={agent.energy} 
        color={agent.energy < 20 ? "bg-red-400" : "bg-yellow-400"} 
        icon={Zap}
      />
      <ProgressBar 
        label="Oxygen" 
        value={agent.oxygen} 
        color={agent.oxygen < 15 ? "bg-red-500" : "bg-cyan-400"} 
        icon={Wind}
      />

      {/* Body Temp moved to Vitals section */}
      <div className="flex justify-between items-center py-2 px-1 bg-slate-800/30 rounded border border-slate-800/50 mt-2">
        <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-400">
          <Thermometer size={14} className="text-orange-400" />
          Body Temp
        </div>
        <span className="font-mono text-sm text-slate-200">{agent.bodyTemp.toFixed(1)}°C</span>
      </div>

      <div className="mt-6 border-t border-slate-800 pt-4">
        <div className="text-[10px] font-bold text-slate-500 uppercase mb-2 flex items-center gap-1">
          <Box size={10} />
          Inventory Management
        </div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="bg-slate-800 p-2 rounded flex flex-col items-center justify-center">
              <div className="font-mono text-base text-cyan-300">{agent.inventory.ice}</div>
              <span className="text-slate-400 text-[9px] uppercase">Ice (kg)</span>
          </div>
          <div className="bg-slate-800 p-2 rounded flex flex-col items-center justify-center">
              <div className="font-mono text-base text-cyan-500">{agent.inventory.oxygenTanks}</div>
              <span className="text-slate-400 text-[9px] uppercase">O2 Tanks</span>
          </div>
          <div className="bg-slate-800 p-2 rounded flex flex-col items-center justify-center">
              <div className="font-mono text-base text-yellow-500">{agent.inventory.rechargePacks}</div>
              <span className="text-slate-400 text-[9px] uppercase">Packs</span>
          </div>
        </div>
      </div>
    </div>
  );
};