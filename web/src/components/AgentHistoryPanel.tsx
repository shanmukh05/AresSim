/** Panel listing recent agent/player actions and outcomes. */

import { ArrowDownUp, LocateFixed } from "lucide-react";
import * as Select from "@radix-ui/react-select";
import { useMemo } from "react";
import { useAresStore } from "../state/useAresStore";
import type { ActionType } from "../types/sim";

const filters: (ActionType | "all")[] = ["all", "move", "scan", "extract", "unload", "build", "service", "wait", "invalid", "event"];

export function AgentHistoryPanel() {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const newestFirst = useAresStore((state) => state.historyNewestFirst);
  const activeFilter = useAresStore((state) => state.historyFilter);
  const toggleOrder = useAresStore((state) => state.toggleHistoryOrder);
  const setHistoryFilter = useAresStore((state) => state.setHistoryFilter);
  const selectTarget = useAresStore((state) => state.selectTarget);
  const highlightCell = useAresStore((state) => state.highlightCell);

  const rows = useMemo(() => {
    const filtered = activeFilter === "all" ? snapshot.history : snapshot.history.filter((entry) => entry.action === activeFilter);
    return newestFirst ? filtered : [...filtered].reverse();
  }, [activeFilter, newestFirst, snapshot.history]);

  return (
    <section className="grid h-full min-h-0 grid-rows-[auto_auto_minmax(0,1fr)] overflow-hidden border-t border-stone-800/80 bg-transparent p-3" data-testid="agent-history">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-white">Agent Step History</h2>
          <p className="text-xs text-stone-500">Path, actions, rewards, and events</p>
          <p className="text-xs text-stone-500">
            Elapsed simulation steps: <span className="font-semibold text-cyan-100">{snapshot.step}</span>
          </p>
        </div>
        <button className="flex h-8 items-center gap-2 rounded border border-stone-700 px-2 text-xs text-stone-300 hover:border-stone-500" onClick={toggleOrder} type="button">
          <ArrowDownUp size={14} />
          {newestFirst ? "Newest" : "Oldest"}
        </button>
      </div>

      <div className="mb-3">
        <label className="mb-1 block text-[10px] uppercase tracking-wide text-stone-500" htmlFor="history-filter-trigger">
          Filter
        </label>
        <Select.Root value={activeFilter} onValueChange={(value) => setHistoryFilter(value as ActionType | "all")}>
          <Select.Trigger
            aria-label="Filter agent history"
            className="flex h-9 w-full items-center justify-between rounded border border-stone-700 bg-stone-950/70 px-3 text-sm capitalize text-stone-200 hover:border-stone-500"
            id="history-filter-trigger"
          >
            <Select.Value />
            <Select.Icon className="text-stone-500">▾</Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Content className="z-50 overflow-hidden rounded border border-stone-700 bg-stone-950 text-sm text-stone-100 shadow-2xl">
              <Select.Viewport className="p-1">
                {filters.map((filter) => (
                  <Select.Item key={filter} className="cursor-pointer rounded px-3 py-2 capitalize outline-none data-[highlighted]:bg-cyan-300/15 data-[highlighted]:text-cyan-100" value={filter}>
                    <Select.ItemText>{filter.replace("_", " ")}</Select.ItemText>
                  </Select.Item>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>
      </div>

      <div className="grid min-h-0 content-start gap-2 overflow-y-auto pr-1">
        {rows.map((entry) => (
          <button
            key={entry.id}
            className="grid grid-cols-[52px_76px_1fr_52px_20px] items-center gap-2 rounded border border-stone-800 bg-stone-950/45 px-2 py-2 text-left text-xs hover:border-cyan-300/50"
            data-testid={`history-row-${entry.step}`}
            onClick={() => {
              selectTarget({ kind: "history", id: entry.id });
              highlightCell(entry.target ?? null);
            }}
            type="button"
          >
            <span className="font-mono text-stone-500">#{entry.step.toString().padStart(3, "0")}</span>
            <span className="capitalize text-cyan-100">{entry.action.replace("_", " ")}</span>
            <span className="truncate text-stone-300">{entry.result}</span>
            <span className={entry.reward >= 0 ? "text-emerald-200" : "text-rose-200"}>{entry.reward.toFixed(2)}</span>
            <LocateFixed size={14} className="text-stone-500" />
            <span className="col-span-5 truncate text-[11px] text-stone-500">
              {Object.entries(entry.rewardTerms)
                .map(([key, value]) => `${key}: ${value.toFixed(2)}`)
                .join(" | ")}
              {entry.events.length ? ` | ${entry.events.join(", ")}` : ""}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
