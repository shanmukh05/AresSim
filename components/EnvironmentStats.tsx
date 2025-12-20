import React from 'react';
import { EnvironmentState } from '../types';
import { Sun, CloudFog, Radio } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface EnvironmentStatsProps {
  environment: EnvironmentState;
}

// Mock data generator for the sparkline
const generateMockHistory = (current: number) => {
    return Array.from({length: 10}, (_, i) => ({
        time: i,
        val: current + (Math.random() * 4 - 2)
    }));
};

export const EnvironmentStats: React.FC<EnvironmentStatsProps> = ({ environment }) => {
  const data = generateMockHistory(environment.temperature);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        {/* Time Panel */}
        <div className="bg-slate-900/60 border border-slate-700 p-3 rounded flex items-center gap-3">
            <div className="p-2 bg-yellow-900/30 rounded-full text-yellow-500">
                <Sun size={20} />
            </div>
            <div>
                <div className="text-xs text-slate-400">SOL {environment.sol}</div>
                <div className="text-xl font-mono text-white">
                    {Math.floor(environment.timeOfDay).toString().padStart(2, '0')}:
                    {Math.floor((environment.timeOfDay % 1) * 60).toString().padStart(2, '0')}
                </div>
            </div>
        </div>

        {/* Temperature Panel */}
        <div className="bg-slate-900/60 border border-slate-700 p-3 rounded flex flex-col justify-between">
            <div className="flex justify-between items-start">
                <div>
                    <div className="text-xs text-slate-400">EXTERNAL TEMP</div>
                    <div className="text-lg font-mono text-white">{environment.temperature.toFixed(1)}°C</div>
                </div>
                {/* Mini Sparkline */}
                <div className="w-16 h-10">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <Line type="monotone" dataKey="val" stroke="#f97316" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>

        {/* Radiation Panel */}
        <div className="bg-slate-900/60 border border-slate-700 p-3 rounded flex items-center gap-3">
             <div className={`p-2 rounded-full ${environment.radiationLevel > 0.7 ? 'bg-red-900/30 text-red-500 animate-pulse' : 'bg-green-900/30 text-green-500'}`}>
                <Radio size={20} />
            </div>
            <div>
                <div className="text-xs text-slate-400">RADIATION</div>
                <div className="text-lg font-mono text-white">{environment.radiationLevel.toFixed(2)} mSv</div>
            </div>
        </div>

        {/* Dust Storm Warning */}
         <div className={`
            border p-3 rounded flex items-center gap-3 transition-colors duration-500
            ${environment.dustStormIntensity > 0.3 ? 'bg-orange-950/50 border-orange-500' : 'bg-slate-900/60 border-slate-700'}
         `}>
             <div className={`${environment.dustStormIntensity > 0.3 ? 'text-orange-500' : 'text-slate-600'}`}>
                <CloudFog size={20} />
            </div>
            <div>
                <div className="text-xs text-slate-400">WEATHER</div>
                <div className={`text-sm font-bold uppercase ${environment.dustStormIntensity > 0.3 ? 'text-orange-400' : 'text-slate-300'}`}>
                    {environment.dustStormIntensity > 0.6 ? 'SEVERE STORM' : environment.dustStormIntensity > 0.2 ? 'HIGH WINDS' : 'CLEAR'}
                </div>
            </div>
        </div>
    </div>
  );
};
