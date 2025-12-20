import React from 'react';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface RewardPanelProps {
  totalReward: number;
  lastReward: { value: number; source: string } | null;
}

export const RewardPanel: React.FC<RewardPanelProps> = ({ totalReward, lastReward }) => {
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 shrink-0">
      <h3 className="text-xs font-bold text-slate-400 uppercase mb-4 flex items-center gap-2">
        <Activity size={14} />
        Reward Function Metrics
      </h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Total Accumulation */}
        <div className="bg-slate-950 p-3 rounded border border-slate-800">
          <div className="text-xs text-slate-500 mb-1">TOTAL REWARD</div>
          <div className={`text-2xl font-mono font-bold ${totalReward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {totalReward.toFixed(1)}
          </div>
        </div>

        {/* Last Action Impact */}
        <div className="bg-slate-950 p-3 rounded border border-slate-800">
          <div className="text-xs text-slate-500 mb-1">LAST DELTA</div>
          {lastReward ? (
            <div className="animate-fadeIn">
              <div className={`text-2xl font-mono font-bold flex items-center gap-1 ${lastReward.value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {lastReward.value > 0 ? '+' : ''}{lastReward.value.toFixed(1)}
                {lastReward.value >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              </div>
              <div className="text-[10px] text-slate-500 truncate mt-1">
                {lastReward.source}
              </div>
            </div>
          ) : (
             <div className="text-sm text-slate-600 italic mt-1">Waiting for action...</div>
          )}
        </div>
      </div>

      {/* Reward/Penalty Table */}
      <div className="space-y-2">
        <div className="text-[10px] text-slate-500 font-bold uppercase border-b border-slate-800 pb-1">
          Attrition Policies (HP Loss)
        </div>
        
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] pb-2">
          <div className="flex justify-between text-slate-400">
            <span>Cold (&lt; -60°C)</span>
            <span className="text-red-400">-0.5/t</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Radiation (&gt; 5.0)</span>
            <span className="text-red-400">-0.5/t</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Lethal (&lt; -75°C)</span>
            <span className="text-red-500 font-bold">-2.0/t</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Storm (&gt; 0.5)</span>
            <span className="text-red-400">-1.0/t</span>
          </div>
        </div>

        <div className="text-[10px] text-slate-500 font-bold uppercase border-b border-slate-800 pb-1">
          Active Policies
        </div>
        
        <div className="grid grid-cols-3 gap-1">
            <div className="bg-slate-800/50 p-1 rounded border border-slate-800 text-center">
                <div className="text-[9px] text-slate-500">FLAT</div>
                <div className="text-red-400 font-mono text-xs">-1.0</div>
            </div>
            <div className="bg-slate-800/50 p-1 rounded border border-slate-800 text-center">
                <div className="text-[9px] text-slate-500">SANDY</div>
                <div className="text-red-400 font-mono text-xs">-2.0</div>
            </div>
            <div className="bg-slate-800/50 p-1 rounded border border-slate-800 text-center">
                <div className="text-[9px] text-slate-500">ROCKY</div>
                <div className="text-red-400 font-mono text-xs">-3.0</div>
            </div>
        </div>

        <div className="flex justify-between items-center text-xs pt-1">
           <span className="text-slate-300">Hazard (Crater)</span>
           <span className="text-red-500 font-mono">-10.0</span>
        </div>
        <div className="flex justify-between items-center text-xs">
           <span className="text-slate-300">Ice Deposit Mined</span>
           <span className="text-cyan-400 font-mono">+50.0</span>
        </div>
        <div className="flex justify-between items-center text-xs">
           <span className="text-slate-300">Medical Support</span>
           <span className="text-green-400 font-mono">Resets HP</span>
        </div>
        <div className="flex justify-between items-center text-xs">
           <span className="text-slate-300">Survival Bonus</span>
           <span className="text-green-400 font-mono">+0.1 / tick</span>
        </div>
      </div>
    </div>
  );
};