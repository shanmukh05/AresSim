/** Analytics charts for habitat build and collection progress. */

import { useMemo } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, ComposedChart } from "recharts";
import { ChartCard, AXIS_PROPS, CHART_COLORS, TOOLTIP_STYLE } from "./ChartCard";
import type { AnalyticsData } from "./useAnalyticsData";

export function ProgressTab({ data }: { data: AnalyticsData }) {
  const habitat = useMemo(() => data.series.map((p) => ({ step: p.step, progress: Number(p.buildProgress.toFixed(2)) })), [data.series]);
  const iceScan = useMemo(() => {
    let ice = 0;
    let scan = 0;
    return data.series.map((p) => {
      ice = Math.max(ice, p.cumulativeByCategory.iceCollected ?? 0);
      scan = Math.max(scan, p.cumulativeByCategory.terrainScanned ?? 0);
      return { step: p.step, ice: Number(ice.toFixed(2)), scan: Number(scan.toFixed(2)) };
    });
  }, [data.series]);
  const serviceEvents = useMemo(() => data.series.filter((p) => p.action === "service" && p.valid).map((p) => ({ step: p.step, events: 1 })), [data.series]);
  const objectiveStacked = useMemo(() => data.series.map((p) => ({
    step: p.step,
    iceCollected: Number((p.cumulativeByCategory.iceCollected ?? 0).toFixed(3)),
    terrainScanned: Number((p.cumulativeByCategory.terrainScanned ?? 0).toFixed(3)),
    habitatBuilt: Number((p.cumulativeByCategory.habitatBuilt ?? 0).toFixed(3)),
    serviced: Number((p.cumulativeByCategory.serviced ?? 0).toFixed(3)),
    delivered: Number((p.cumulativeByCategory.delivered ?? 0).toFixed(3)),
    traversal: Number((p.cumulativeByCategory.traversal ?? 0).toFixed(3)),
    blocked: Number((p.cumulativeByCategory.blockedPenalty ?? 0).toFixed(3)),
  })), [data.series]);
  const efficiency = useMemo(() => {
    const out: { step: number; efficiency: number }[] = [];
    let running = 0;
    for (const p of data.series) {
      running += p.reward;
      const eff = p.step > 0 ? running / p.step : 0;
      out.push({ step: p.step, efficiency: Number(eff.toFixed(4)) });
    }
    return out;
  }, [data.series]);

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <ChartCard title="Habitat build progress" description="100% target">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={habitat} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="habitat" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS[2]} stopOpacity={0.6} />
                <stop offset="100%" stopColor={CHART_COLORS[2]} stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} domain={[0, 100]} />
            <Tooltip {...TOOLTIP_STYLE} />
            <ReferenceLine y={100} stroke="#34d399" strokeDasharray="4 2" label={{ value: "complete", fontSize: 10, fill: "#34d399" }} />
            <Area type="monotone" dataKey="progress" stroke={CHART_COLORS[2]} strokeWidth={2} fill="url(#habitat)" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Cumulative ice collected & sites scanned" description="Two objective tracks">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={iceScan} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Line type="monotone" dataKey="ice" stroke={CHART_COLORS[1]} strokeWidth={2} dot={false} name="ice reward" />
            <Line type="monotone" dataKey="scan" stroke={CHART_COLORS[0]} strokeWidth={2} dot={false} name="scan reward" />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Service events" description="Per service step">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={serviceEvents} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Bar dataKey="events" fill={CHART_COLORS[3]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Cumulative reward by objective category" description="Stacked area">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={objectiveStacked} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            {(["iceCollected", "terrainScanned", "delivered", "habitatBuilt", "serviced", "traversal", "blocked"] as const).map((key, i) => (
              <Area key={key} type="monotone" dataKey={key} stackId="1" stroke={CHART_COLORS[i % CHART_COLORS.length]} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.32} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Step count vs reward efficiency" description="Running reward / step">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={efficiency} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <ReferenceLine y={0} stroke="#52525b" />
            <Line type="monotone" dataKey="efficiency" stroke={CHART_COLORS[5]} strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="rounded-md border border-stone-800 bg-stone-950/40 p-3 text-xs text-stone-400">
        <div className="mb-1 font-semibold text-stone-200">Progress summary</div>
        Final battery: <span className="text-stone-100">{data.finalBattery.toFixed(2)}%</span>
        <br />
        Final livability: <span className="text-stone-100">{data.finalLivability.toFixed(2)}%</span>
        <br />
        Total reward: <span className="text-stone-100">{data.totalReward.toFixed(3)}</span>
      </div>
    </div>
  );
}
