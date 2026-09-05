/** Analytics charts for per-step and cumulative rewards. */

import { useMemo } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartCard, AXIS_PROPS, CHART_COLORS, TOOLTIP_STYLE } from "./ChartCard";
import type { AnalyticsData } from "./useAnalyticsData";

export function RewardsTab({ data }: { data: AnalyticsData }) {
  const perStep = useMemo(() => data.series.map((p) => ({ step: p.step, reward: Number(p.reward.toFixed(3)) })), [data.series]);
  const cumulative = useMemo(() => data.series.map((p) => ({ step: p.step, total: Number(p.cumulativeReward.toFixed(3)) })), [data.series]);
  const perCategory = useMemo(() => data.series.map((p) => {
    const out: Record<string, number | string> = { step: p.step };
    for (const key of data.rewardCategories) out[key] = Number((p.rewardTerms[key] ?? 0).toFixed(3));
    return out;
  }), [data.series, data.rewardCategories]);
  const cumulativeStacked = useMemo(() => {
    let cum: Record<string, number> = {};
    return data.series.map((p) => {
      cum = { ...cum };
      for (const key of data.rewardCategories) cum[key] = Number((cum[key] ?? 0) + (p.rewardTerms[key] ?? 0)).toFixed(3) as unknown as number;
      return { step: p.step, ...cum };
    });
  }, [data.series, data.rewardCategories]);

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <ChartCard title="Total reward per step" description="Signed reward each timestep">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={perStep} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <ReferenceLine y={0} stroke="#52525b" />
            <Line type="monotone" dataKey="reward" stroke={CHART_COLORS[0]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Cumulative total reward" description="Sum over time">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={cumulative} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="cumReward" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS[0]} stopOpacity={0.6} />
                <stop offset="100%" stopColor={CHART_COLORS[0]} stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Area type="monotone" dataKey="total" stroke={CHART_COLORS[0]} strokeWidth={2} fill="url(#cumReward)" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Reward per category over time" description="Click legend to toggle">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={perCategory} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine y={0} stroke="#52525b" />
            {data.rewardCategories.map((key, index) => (
              <Line key={key} type="monotone" dataKey={key} stroke={CHART_COLORS[index % CHART_COLORS.length]} dot={false} strokeWidth={1.6} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Cumulative reward by category" description="Stacked area">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={cumulativeStacked} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {data.rewardCategories.map((key, index) => (
              <Area key={key} type="monotone" dataKey={key} stackId="1" stroke={CHART_COLORS[index % CHART_COLORS.length]} fill={CHART_COLORS[index % CHART_COLORS.length]} fillOpacity={0.35} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Reward per step (bars)" description="Signed per-step bars">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={perStep} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <ReferenceLine y={0} stroke="#52525b" />
            <Bar dataKey="reward" radius={[2, 2, 0, 0]}>
              {perStep.map((entry, index) => (
                <Cell key={index} fill={entry.reward >= 0 ? CHART_COLORS[3] : CHART_COLORS[4]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Stacked reward categories per step" description="Bars stacked by category">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={perCategory} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine y={0} stroke="#52525b" />
            {data.rewardCategories.map((key, index) => (
              <Bar key={key} dataKey={key} stackId="1" fill={CHART_COLORS[index % CHART_COLORS.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}