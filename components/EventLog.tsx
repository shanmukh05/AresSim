import React, { useEffect, useRef } from 'react';
import { LogEntry } from '../types';

interface EventLogProps {
  logs: LogEntry[];
}

export const EventLog: React.FC<EventLogProps> = ({ logs }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to top when new logs arrive (since logs are ordered newest-first)
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }
  }, [logs]);

  return (
    <div className="flex flex-col h-full w-full">
      <div className="p-2 border-b border-slate-800 bg-slate-900 flex justify-between items-center shrink-0">
        <span className="text-xs font-bold text-slate-400">SYSTEM LOG</span>
        <span className="text-[10px] text-slate-600 font-mono">LIVE FEED</span>
      </div>
      <div 
        className="flex-1 overflow-y-auto p-2 font-mono text-xs space-y-1 scroll-smooth" 
        ref={scrollRef}
      >
        {logs.map((log) => (
          <div key={log.id} className="flex gap-2 animate-fadeIn border-l-2 border-transparent hover:border-slate-700 pl-1 transition-colors">
            <span className="text-slate-600 shrink-0">[{log.timestamp}]</span>
            <span className={`
              break-words
              ${log.type === 'danger' ? 'text-red-500 font-bold' : ''}
              ${log.type === 'warning' ? 'text-orange-400' : ''}
              ${log.type === 'success' ? 'text-green-400' : ''}
              ${log.type === 'info' ? 'text-slate-300' : ''}
            `}>
              {log.message}
            </span>
            {log.reward !== 0 && (
                <span className={`ml-auto shrink-0 ${log.reward && log.reward > 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {log.reward && log.reward > 0 ? '+' : ''}{log.reward}
                </span>
            )}
          </div>
        ))}
        {logs.length === 0 && <div className="text-slate-600 italic">No events recorded.</div>}
      </div>
    </div>
  );
};