/** Shared small UI primitives (icon buttons, tooltips) for the shell. */

import type { ReactNode } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";

export function IconButton({
  label,
  children,
  active = false,
  disabled = false,
  onClick,
}: {
  label: string;
  children: ReactNode;
  active?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button
          aria-label={label}
          disabled={disabled}
          className={`grid h-10 w-10 place-items-center rounded-md border text-sm shadow-[inset_0_1px_0_rgba(255,255,255,0.18),inset_0_-6px_12px_rgba(0,0,0,0.28),0_8px_18px_rgba(0,0,0,0.34)] ring-1 ring-white/5 transition hover:-translate-y-0.5 active:translate-y-px active:shadow-[inset_0_4px_10px_rgba(0,0,0,0.45)] disabled:pointer-events-none disabled:opacity-40 disabled:shadow-none ${
            active
              ? "border-cyan-100 bg-gradient-to-b from-cyan-300/35 via-cyan-800/55 to-cyan-950 text-cyan-50 ring-cyan-200/35"
              : "border-stone-500 bg-gradient-to-b from-stone-700 via-stone-900 to-black text-stone-100 hover:border-cyan-300/80 hover:text-white"
          }`}
          onClick={onClick}
          type="button"
        >
          {children}
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="z-50 rounded border border-stone-700 bg-stone-950 px-2 py-1 text-xs text-stone-100 shadow-xl" sideOffset={8}>
          {label}
          <Tooltip.Arrow className="fill-stone-950" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function Metric({ label, value, tone = "default" }: { label: string; value: string; tone?: "default" | "good" | "warn" | "bad" }) {
  const color = {
    default: "text-stone-100",
    good: "text-emerald-200",
    warn: "text-amber-200",
    bad: "text-rose-200",
  }[tone];

  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wide text-stone-500">{label}</div>
      <div className={`truncate text-sm font-semibold ${color}`}>{value}</div>
    </div>
  );
}
