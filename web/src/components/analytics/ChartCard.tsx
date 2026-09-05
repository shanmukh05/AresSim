/** Shared chart card chrome and styling constants for analytics tabs. */

import type { ReactNode } from "react";

export function ChartCard({ title, description, children, height = 240 }: { title: string; description?: string; children: ReactNode; height?: number }) {
  return (
    <section className="grid gap-2 rounded-md border border-stone-800 bg-stone-950/40 p-3" data-testid="chart-card">
      <header className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-stone-100">{title}</h3>
        {description ? <p className="text-[11px] text-stone-500">{description}</p> : null}
      </header>
      <div className="grid" style={{ height }}>
        {children}
      </div>
    </section>
  );
}

export const CHART_COLORS = [
  "#22d3ee",
  "#f59e0b",
  "#a78bfa",
  "#34d399",
  "#fb7185",
  "#60a5fa",
  "#facc15",
  "#f472b6",
  "#4ade80",
  "#f97316",
  "#94a3b8",
];

export const AXIS_PROPS = {
  stroke: "#78716c",
  fontSize: 11,
  tick: { fill: "#a8a29e" },
  axisLine: { stroke: "#44403c" },
  tickLine: { stroke: "#44403c" },
} as const;

export const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#0c0a09",
    border: "1px solid #44403c",
    borderRadius: 8,
    fontSize: 12,
    color: "#f5f5f4",
  },
  labelStyle: { color: "#a8a29e", marginBottom: 4 },
  itemStyle: { color: "#f5f5f4" },
} as const;