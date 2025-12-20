import React from 'react';
import { AgentAction } from '../types';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Pickaxe, Zap, Play, Pause, Wind } from 'lucide-react';

interface ControlDeckProps {
  onAction: (action: AgentAction) => void;
  isRunning: boolean;
  toggleRun: () => void;
}

export const ControlDeck: React.FC<ControlDeckProps> = ({ onAction, isRunning, toggleRun }) => {
  const btnClass = "p-3 bg-slate-800 hover:bg-slate-700 active:bg-slate-600 border border-slate-600 rounded transition-colors flex items-center justify-center";
  
  return (
    <div className="bg-slate-900/80 border border-slate-700 p-4 rounded-lg">
       <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-2">
         <h3 className="text-sm font-bold text-slate-200">MANUAL OVERRIDE</h3>
         <button 
           onClick={toggleRun}
           className={`px-3 py-1 text-xs font-bold rounded flex items-center gap-1 ${isRunning ? 'bg-red-500/20 text-red-400 border border-red-500' : 'bg-green-500/20 text-green-400 border border-green-500'}`}
         >
            {isRunning ? <><Pause size={12}/> PAUSE SIM</> : <><Play size={12}/> RESUME SIM</>}
         </button>
       </div>

      <div className="grid grid-cols-3 gap-2 mb-4 max-w-[200px] mx-auto">
        <div /> 
        <button className={btnClass} onClick={() => onAction(AgentAction.MOVE_NORTH)}><ArrowUp size={20} /></button>
        <div />
        
        <button className={btnClass} onClick={() => onAction(AgentAction.MOVE_WEST)}><ArrowLeft size={20} /></button>
        <div className="flex items-center justify-center text-slate-500 text-xs">NAV</div>
        <button className={btnClass} onClick={() => onAction(AgentAction.MOVE_EAST)}><ArrowRight size={20} /></button>
        
        <div />
        <button className={btnClass} onClick={() => onAction(AgentAction.MOVE_SOUTH)}><ArrowDown size={20} /></button>
        <div />
      </div>

      <div className="grid grid-cols-3 gap-2">
        <button className={`${btnClass} text-cyan-300`} onClick={() => onAction(AgentAction.MINE_ICE)} title="Mine Ice">
            <Pickaxe size={18} />
        </button>
        <button className={`${btnClass} text-cyan-500`} onClick={() => onAction(AgentAction.USE_OXYGEN_TANK)} title="Use Oxygen Tank">
            <Wind size={18} />
        </button>
        <button className={`${btnClass} text-blue-400`} onClick={() => onAction(AgentAction.CHARGE_BATTERY)} title="Recharge at Base">
            <div className="relative">
              <Zap size={18} />
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-blue-500 rounded-full border border-white"></div>
            </div>
        </button>
      </div>
    </div>
  );
};