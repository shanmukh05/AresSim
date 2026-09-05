/** Analytics charts for weather and terrain visit patterns. */

import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, Tooltip, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { ChartCard, AXIS_PROPS, CHART_COLORS, TOOLTIP_STYLE } from "./ChartCard";
import type { AnalyticsData } from "./useAnalyticsData";
import type { WeatherState } from "../../types/sim";

const WEATHER_STATES: WeatherState[] = ["Clear", "Dusty", "Dust Front", "Cold Night", "Severe Storm"];
const TERRAIN_VISITED_KEYS = ["move", "scan", "extract", "unload", "build", "service", "wait", "invalid"] as const;

export function EnvironmentTab({ data }: { data: AnalyticsData }) {
  const weather = useMemo(() => WEATHER_STATES.map((state) => ({ state, count: data.weatherCounts[state] ?? 0 })).filter((entry) => entry.count > 0), [data.weatherCounts]);
  const terrainActions = useMemo(() => data.terrainVisitedByAction.filter((entry) => entry.count > 0), [data.terrainVisitedByAction]);
  const solBySol = useMemo(() => {
    const bySol = new Map<number, { sol: number; reward: number }>();
    for (const p of data.series) {
      const entry = bySol.get(p.sol) ?? { sol: p.sol, reward: 0 };
      entry.reward = Number((entry.reward + p.reward).toFixed(3));
      bySol.set(p.sol, entry);
    }
    return [...bySol.values()].sort((a, b) => a.sol - b.sol);
  }, [data.series]);

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <ChartCard title="Weather distribution" description="Count of each weather state across steps">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={weather} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="state" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Bar dataKey="count" radius={[2, 2, 0, 0]}>
              {weather.map((_, index) => (
                <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Types of agent actions" description="Pie of all action types">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={terrainActions} dataKey="count" nameKey="action" cx="50%" cy="50%" outerRadius={80} label>
              {terrainActions.map((_, index) => (
                <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Sol-by-sol reward" description="Total reward grouped by Sol">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={solBySol} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="sol" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Bar dataKey="reward" radius={[2, 2, 0, 0]}>
              {solBySol.map((entry, index) => (
                <Cell key={index} fill={entry.reward >= 0 ? CHART_COLORS[3] : CHART_COLORS[4]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="rounded-md border border-stone-800 bg-stone-950/40 p-3 text-xs text-stone-400">
        <div className="mb-1 font-semibold text-stone-200">Environment summary</div>
        Weather states observed: <span className="text-stone-100">{weather.length}</span>
        <br />
        Distinct action types: <span className="text-stone-100">{terrainActions.length}</span>
        <br />
        Sols traversed: <span className="text-stone-100">{solBySol.length}</span>
      </div>
    </div>
  );
}

void TERRAIN_VISITED_KEYS;
