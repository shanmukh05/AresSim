import React from 'react';
import { TerrainType, Coordinates, AgentState } from '../types';
import { TERRAIN_COLORS } from '../constants';
import { MapPin, Zap, Triangle, Hexagon, Droplet, Skull, Square } from 'lucide-react';

interface MarsMapProps {
  grid: TerrainType[][];
  agent: AgentState;
}

export const MarsMap: React.FC<MarsMapProps> = ({ grid, agent }) => {
  return (
    <div className="relative flex flex-col h-full w-full">
      {/* Grid Container - Centered and Constrained */}
      <div className="flex-grow flex items-center justify-center overflow-hidden">
        <div 
          className="grid gap-0.5 shadow-2xl bg-black/50 p-1 rounded border border-slate-700/50"
          style={{ 
            gridTemplateColumns: `repeat(${grid.length}, minmax(0, 1fr))`,
            width: '100%',
            maxWidth: '400px', // Constrained width to fix visualization issues
            aspectRatio: '1/1'
          }}
        >
          {grid.map((row, y) => (
            row.map((cellType, x) => {
              const isAgentHere = agent.position.x === x && agent.position.y === y;
              const isDiscovered = true; // Fog of war could go here

              return (
                <div
                  key={`${x}-${y}`}
                  className={`
                    w-full h-full rounded-[1px] flex items-center justify-center relative
                    transition-colors duration-300
                    ${isDiscovered ? TERRAIN_COLORS[cellType] : 'bg-black'}
                    ${isAgentHere ? 'ring-1 ring-white z-10' : ''}
                  `}
                >
                  {/* Terrain Icons - Scaled down */}
                  {cellType === TerrainType.BASE && <Hexagon size={12} className="text-blue-400 opacity-80" />}
                  {cellType === TerrainType.ICE && <Droplet size={10} className="text-cyan-200 opacity-70" />}
                  {cellType === TerrainType.CRATER && <Skull size={10} className="text-red-900 opacity-50" />}
                  {cellType === TerrainType.ROCKY && <Triangle size={8} className="text-stone-400 opacity-50" />}

                  {/* Agent Icon */}
                  {isAgentHere && (
                    <div className="absolute inset-0 flex items-center justify-center animate-pulse">
                      <div className="w-2 h-2 bg-white rounded-full shadow-[0_0_8px_rgba(255,255,255,0.8)]"></div>
                    </div>
                  )}
                </div>
              );
            })
          ))}
        </div>
      </div>

      {/* Map Overlay Info */}
      <div className="absolute top-0 left-0 text-[10px] font-mono text-slate-400 bg-black/60 px-2 py-1 rounded pointer-events-none border border-slate-800">
        POS: {agent.position.x}:{agent.position.y}
      </div>

      {/* Legend with Rewards/Costs */}
      <div className="mt-2 pt-2 border-t border-slate-800/50 grid grid-cols-4 sm:grid-cols-7 gap-y-3 gap-x-2 text-[9px] text-slate-500 uppercase tracking-tighter">
        <div className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse"></div> Agent</div>
        
        <div className="flex flex-col">
            <div className="flex items-center gap-1"><Hexagon size={10} className="text-blue-400"/> Base</div>
            <span className="text-[8px] text-blue-500/70 ml-3.5">Dock</span>
        </div>
        
        <div className="flex flex-col">
            <div className="flex items-center gap-1"><Droplet size={10} className="text-cyan-200"/> Ice</div>
            <span className="text-[8px] text-cyan-500/70 ml-3.5">+50.0</span>
        </div>

        <div className="flex flex-col">
            <div className="flex items-center gap-1"><Square size={10} className="text-orange-900/40 fill-orange-900/20"/> Flat</div>
            <span className="text-[8px] text-orange-900/70 ml-3.5">-1.0</span>
        </div>

        <div className="flex flex-col">
            <div className="flex items-center gap-1"><div className="w-2 h-2 bg-amber-700/30 border border-amber-800"></div> Sandy</div>
            <span className="text-[8px] text-orange-400/70 ml-3.5">-2.0</span>
        </div>

        <div className="flex flex-col">
            <div className="flex items-center gap-1"><Triangle size={10} className="text-stone-400"/> Rocky</div>
            <span className="text-[8px] text-orange-600/70 ml-3.5">-3.0</span>
        </div>

        <div className="flex flex-col">
            <div className="flex items-center gap-1"><Skull size={10} className="text-red-900"/> Crater</div>
            <span className="text-[8px] text-red-600/70 ml-3.5">-10.0</span>
        </div>
      </div>
    </div>
  );
};