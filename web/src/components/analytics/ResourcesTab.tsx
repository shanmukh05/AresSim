/** Analytics charts for battery, life support, power, dust, and cargo. */

import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartCard, AXIS_PROPS, CHART_COLORS, TOOLTIP_STYLE } from "./ChartCard";
import type { AnalyticsData } from "./useAnalyticsData";

export function ResourcesTab({ data }: { data: AnalyticsData }) {
  const battery = useMemo(() => data.series.map((p) => ({ step: p.step, battery: Number(p.battery.toFixed(2)) })), [data.series]);
  const life = useMemo(() => data.series.map((p) => ({ step: p.step, water: Number(p.water.toFixed(2)), oxygen: Number(p.oxygen.toFixed(2)) })), [data.series]);
  const livability = useMemo(() => data.series.map((p) => ({ step: p.step, livability: Number(p.livability.toFixed(2)) })), [data.series]);
  const power = useMemo(() => data.series.map((p) => ({ step: p.step, generated: Number(p.powerGenerated.toFixed(2)), consumed: Number(p.powerConsumed.toFixed(2)), margin: Number((p.powerGenerated - p.powerConsumed).toFixed(2)) })), [data.series]);
  const dust = useMemo(() => data.series.map((p) => ({ step: p.step, dust: Number(p.dustIntensity.toFixed(3)) })), [data.series]);
  const health = useMemo(() => data.series.map((p) => ({ step: p.step, health: Number(p.health.toFixed(2)) })), [data.series]);
  const payload = useMemo(() => data.series.map((p) => ({ step: p.step, used: p.payloadUsedKg, ice: p.cargoIce, samples: p.cargoSamples, capacity: p.payloadCapacityKg })), [data.series]);

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <ChartCard title="Rover battery" description="Thresholds at 35% warn and 18% danger">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={battery} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="battery" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS[4]} stopOpacity={0.55} />
                <stop offset="100%" stopColor={CHART_COLORS[4]} stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} domain={[0, 100]} />
            <Tooltip {...TOOLTIP_STYLE} />
            <ReferenceLine y={35} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: "warn", fontSize: 10, fill: "#f59e0b" }} />
            <ReferenceLine y={18} stroke="#fb7185" strokeDasharray="4 2" label={{ value: "danger", fontSize: 10, fill: "#fb7185" }} />
            <Area type="monotone" dataKey="battery" stroke={CHART_COLORS[4]} strokeWidth={2} fill="url(#battery)" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Colony resources" description="Water and oxygen reserves">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={life} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Line type="monotone" dataKey="water" stroke={CHART_COLORS[1]} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="oxygen" stroke={CHART_COLORS[6]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Habitat livability" description="Threshold at 16%">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={livability} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} domain={[0, 100]} />
            <Tooltip {...TOOLTIP_STYLE} />
            <ReferenceLine y={16} stroke="#fb7185" strokeDasharray="4 2" />
            <Line type="monotone" dataKey="livability" stroke={CHART_COLORS[3]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Power balance" description="Generated vs consumed">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={power} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Area type="monotone" dataKey="generated" stackId="g" stroke={CHART_COLORS[0]} fill={CHART_COLORS[0]} fillOpacity={0.3} />
            <Area type="monotone" dataKey="consumed" stackId="c" stroke={CHART_COLORS[4]} fill={CHART_COLORS[4]} fillOpacity={0.3} />
            <Line type="monotone" dataKey="margin" stroke="#f5f5f4" strokeWidth={1} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Dust intensity" description="Atmospheric dust">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={dust} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="dust" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS[9]} stopOpacity={0.55} />
                <stop offset="100%" stopColor={CHART_COLORS[9]} stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} domain={[0, 1]} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Area type="monotone" dataKey="dust" stroke={CHART_COLORS[9]} strokeWidth={2} fill="url(#dust)" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Rover health" description="Threshold at 35%">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={health} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} domain={[0, 100]} />
            <Tooltip {...TOOLTIP_STYLE} />
            <ReferenceLine y={35} stroke="#f59e0b" strokeDasharray="4 2" />
            <Line type="monotone" dataKey="health" stroke={CHART_COLORS[10]} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Rover payload" description="Ice and samples against payload capacity">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={payload} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#292524" strokeDasharray="3 3" />
            <XAxis dataKey="step" {...AXIS_PROPS} />
            <YAxis {...AXIS_PROPS} domain={[0, "dataMax"]} />
            <Tooltip {...TOOLTIP_STYLE} />
            <Area type="stepAfter" dataKey="ice" stackId="payload" stroke={CHART_COLORS[1]} fill={CHART_COLORS[1]} fillOpacity={0.35} />
            <Area type="stepAfter" dataKey="samples" stackId="payload" stroke={CHART_COLORS[0]} fill={CHART_COLORS[0]} fillOpacity={0.35} />
            <Line type="stepAfter" dataKey="capacity" stroke={CHART_COLORS[4]} strokeDasharray="4 2" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
