/** Modal host for live and replay analytics tabs. */

import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { Activity, BarChart3, LineChart as LineChartIcon, Map as MapIcon, Sun } from "lucide-react";
import { useAresStore } from "../../state/useAresStore";
import { useAnalyticsData } from "./useAnalyticsData";
import { RewardsTab } from "./RewardsTab";
import { ResourcesTab } from "./ResourcesTab";
import { BehaviorTab } from "./BehaviorTab";
import { ProgressTab } from "./ProgressTab";
import { EnvironmentTab } from "./EnvironmentTab";

const TABS = [
  { value: "rewards", label: "Rewards", icon: LineChartIcon },
  { value: "resources", label: "Resources", icon: Activity },
  { value: "behavior", label: "Behavior", icon: BarChart3 },
  { value: "progress", label: "Progress", icon: MapIcon },
  { value: "environment", label: "Environment", icon: Sun },
] as const;

export function AnalyticsModal({ open, onOpenChange, initialTab = "rewards" }: { open: boolean; onOpenChange: (open: boolean) => void; initialTab?: string }) {
  const snapshot = useAresStore((state) => state.snapshot)!;
  const data = useAnalyticsData();

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 grid max-h-[88vh] w-[min(1120px,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 grid-rows-[auto_auto_minmax(0,1fr)] overflow-hidden rounded border border-stone-700 bg-[#121015] text-stone-100 shadow-2xl">
          <div className="flex items-start justify-between gap-4 border-b border-stone-800 px-5 py-4">
            <div>
              <Dialog.Title className="text-xl font-semibold text-white">Run Analytics</Dialog.Title>
              <Dialog.Description className="mt-1 text-sm text-stone-400">
                Reward, resource, behavior, progress, and environment visualizations for the current {data.mode} run.
              </Dialog.Description>
            </div>
            <div className="flex items-center gap-3 text-right text-xs text-stone-400">
              <div>
                <div className="text-[10px] uppercase tracking-wide">Source</div>
                <div className={`font-semibold ${data.mode === "load" ? "text-cyan-200" : "text-emerald-200"}`}>{data.mode === "load" ? "Loaded trajectory" : "Live session"}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wide">Steps</div>
                <div className="font-semibold text-stone-100">{data.totalSteps}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wide">Seed</div>
                <div className="font-semibold text-stone-100">{snapshot.seed}</div>
              </div>
            </div>
          </div>

          {data.limited ? (
            <div className="border-b border-amber-300/30 bg-amber-950/40 px-5 py-2 text-xs text-amber-100">
              Limited replay data — load a complete <code>aresim.trajectory.episode.v1</code> file or run a longer session for full charts.
            </div>
          ) : null}

          <Tabs.Root className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)]" defaultValue={initialTab}>
            <Tabs.List className="grid grid-cols-5 border-b border-stone-800 bg-stone-950/45 p-2 text-xs max-sm:grid-cols-2">
              {TABS.map((tab) => {
                const Icon = tab.icon;
                return (
                  <Tabs.Trigger key={tab.value} className="flex items-center justify-center gap-1.5 rounded px-3 py-2 capitalize text-stone-400 data-[state=active]:bg-stone-800 data-[state=active]:text-white" value={tab.value}>
                    <Icon size={14} />
                    {tab.label}
                  </Tabs.Trigger>
                );
              })}
            </Tabs.List>

            <div className="min-h-0 overflow-y-auto p-5">
              <Tabs.Content value="rewards" className="min-h-0 outline-none">
                <RewardsTab data={data} />
              </Tabs.Content>
              <Tabs.Content value="resources" className="min-h-0 outline-none">
                <ResourcesTab data={data} />
              </Tabs.Content>
              <Tabs.Content value="behavior" className="min-h-0 outline-none">
                <BehaviorTab data={data} />
              </Tabs.Content>
              <Tabs.Content value="progress" className="min-h-0 outline-none">
                <ProgressTab data={data} />
              </Tabs.Content>
              <Tabs.Content value="environment" className="min-h-0 outline-none">
                <EnvironmentTab data={data} />
              </Tabs.Content>
            </div>
          </Tabs.Root>

          <Dialog.Close className="grid absolute right-3 top-3 h-9 w-9 place-items-center rounded border border-stone-700 bg-stone-950 text-stone-300 hover:border-stone-500" aria-label="Close analytics">
            <span aria-hidden="true" className="text-lg leading-none">×</span>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
