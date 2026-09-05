/** Analytics charts for action mixes and spatial behavior. */

import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, ResponsiveContainer, ReferenceLine, ComposedChart, Line } from "recharts";
import { ChartCard, AXIS_PROPS, CHART_COLORS, TOOLTIP_STYLE } from "./ChartCard";
import type { AnalyticsData } from "./useAnalyticsData";

const ACTIONS = ["move", "scan", "extract", "unload", "build", "service", "wait", "invalid"] as const;

export function BehaviorTab({ data }: { data: AnalyticsData }) {
  const histogram = useMemo(() => ACTIONS.map((action) => ({ action, count: data.actionCounts[action] ?? 0 })), [data.actionCounts]);
  const frequencyStacked = useMemo(() => {
    const buckets = new Map<number, Record<string, number>>();
    for (const p of data.series) {
      const b = buckets.get(p.step) ?? { step: p.step };
      b[p.action] = (b[p.action] ?? 0) + 1;
      buckets.set(p.step, b);
    }
    return [...buckets.values()].sort((a, b) => a.step - b.step).slice(1).map((b) => ({
      ...b,
      step: b.step,
    }));
  }, [data.series]);
  const validInvalid = useMemo(() => data.series.map((p) => ({ step: p.step, valid: p.valid ? 1 : 0, invalid: p.valid ? 0 : 1 })), [data.series]);
  const path = useMemo(() => data.series.map((p) => ({ x: p.roverX, y: p.roverY, z: Math.max(0.5, p.reward), battery: p.battery, step: p.step })), [data.series]);
  const rewardByAction = useMemo(() => {
    const groups = new Map<string, { action: string; min: number; max: number; sum: number; count: number }>();
    for (const p of data.series) {
      const g = groups.get(p.action) ?? { action: p.action, min: Infinity, max: -Infinity, sum: 0, count: 0 };
      g.min = Math.min(g.min, p.reward);
      g.max = Math.max(g.max, p.reward);
      g.sum += p.reward;
      g.count += 1;
      groups.set(p.action, g);
    }
    return [...groups.values()].map((g) => ({ action: g.action, min: Number(g.min.toFixed(3)), max: Number(g.max.toFixed(3)), avg: Number((g.sum / g.count).toFixed(3)) }));
  }, [data.series]);

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <ChartCard title="Action distribution" description="Count of each action">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={histogram} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="action" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Bar dataKey="count" radius={[2, 2, 0, 0]} fill={CHART_COLORS[0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Action frequency over time" description="Per-step stacked counts">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={frequencyStacked} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {ACTIONS.map((action, i) => (
              <Bar key={action} dataKey={action} stackId="1" fill={CHART_COLORS[i % CHART_COLORS.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Valid vs invalid over time" description="Per-step ratio">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={validInvalid} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="valid" stackId="1" fill={CHART_COLORS[3]} />
            <Bar dataKey="invalid" stackId="1" fill={CHART_COLORS[4]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Rover path" description="Grid coordinates — size = reward, color = battery">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="x" domain={[0, 32]} {...AXIS_PROPS} />
            <YAxis type="number" dataKey="y" name="y" domain={[0, 32]} {...AXIS_PROPS} />
            <ZAxis type="number" dataKey="z" range={[20, 200]} name="reward" />
            <Tooltip {...TOOLTIP_STYLE} cursor={{ strokeDasharray: "3 3" }} />
            <Scatter name="path" data={path} fill={CHART_COLORS[0]} />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Reward by action" description="Min/max/avg across the run">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rewardByAction} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="action" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine y={0} stroke="#52525b" />
            <Bar dataKey="min" fill={CHART_COLORS[4]} fillOpacity={0.3} />
            <Bar dataKey="max" fill={CHART_COLORS[3]} fillOpacity={0.3} />
            <Line type="monotone" dataKey="avg" stroke={CHART_COLORS[0]} strokeWidth={2} dot={{ r: 3 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="rounded-md border border-stone-800 bg-stone-950/40 p-3 text-xs text-stone-400">
        <div className="mb-1 font-semibold text-stone-200">Behavior summary</div>
        Total steps: <span className="text-stone-100">{data.totalSteps}</span>
        <br />
        Valid ratio: <span className={data.validRatio >= 0.7 ? "text-emerald-200" : "text-rose-200"}>{(data.validRatio * 100).toFixed(1)}%</span>
        <br />
        Average reward per step: <span className="text-stone-100">{data.averageRewardPerStep.toFixed(3)}</span>
      </div>
    </div>
  );
}
