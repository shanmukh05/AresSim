/**
 * Flatten store history into chart series for the analytics modal.
 * Aggregates only; it does not recompute rewards.
 */

import { useMemo } from "react";
import { useAresStore } from "../../state/useAresStore";
import type { ActionType, AnalyticsSeriesPoint, WeatherState } from "../../types/sim";

export interface AnalyticsData {
  series: AnalyticsSeriesPoint[];
  mode: "live" | "load";
  limited: boolean;
  totalSteps: number;
  rewardCategories: string[];
  actionCounts: Record<string, number>;
  validCount: number;
  invalidCount: number;
  validRatio: number;
  totalReward: number;
  finalBattery: number;
  finalLivability: number;
  averageRewardPerStep: number;
  weatherCounts: Record<WeatherState, number>;
  terrainVisitedByAction: { action: string; count: number }[];
}

const ACTION_CATEGORIES: ActionType[] = ["move", "scan", "extract", "unload", "build", "service", "wait", "invalid"];

export function useAnalyticsData(): AnalyticsData {
  const series = useAresStore((state) => state.analyticsSeries);
  const runMode = useAresStore((state) => state.runMode);
  const loadedGameplay = useAresStore((state) => state.loadedGameplay);

  return useMemo(() => {
    const mode: "live" | "load" = runMode === "load" ? "load" : "live";
    const limited = mode === "load" && loadedGameplay !== null ? loadedGameplay.steps.length === 0 : series.length <= 1;
    const totalSteps = series.length > 0 ? series[series.length - 1].step : 0;
    const rewardCategories = collectCategories(series);
    const actionCounts = countBy(series, (point) => point.action);
    const validCount = series.filter((point) => point.valid).length;
    const invalidCount = series.filter((point) => !point.valid).length;
    const totalReward = series.length > 0 ? series[series.length - 1].cumulativeReward : 0;
    const finalBattery = series.length > 0 ? series[series.length - 1].battery : 0;
    const finalLivability = series.length > 0 ? series[series.length - 1].livability : 0;
    const averageRewardPerStep = totalSteps > 0 ? totalReward / totalSteps : 0;
    const weatherCounts = countBy(series, (point) => point.weather) as Record<WeatherState, number>;
    const terrainVisitedByAction = ACTION_CATEGORIES.map((action) => ({
      action,
      count: actionCounts[action] ?? 0,
    }));

    return {
      series,
      mode,
      limited,
      totalSteps,
      rewardCategories,
      actionCounts,
      validCount,
      invalidCount,
      validRatio: validCount + invalidCount > 0 ? validCount / (validCount + invalidCount) : 1,
      totalReward,
      finalBattery,
      finalLivability,
      averageRewardPerStep,
      weatherCounts,
      terrainVisitedByAction,
    };
  }, [series, runMode, loadedGameplay]);
}

function countBy<T extends string>(series: AnalyticsSeriesPoint[], key: (point: AnalyticsSeriesPoint) => T): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const point of series) {
    const k = key(point);
    counts[k] = (counts[k] ?? 0) + 1;
  }
  return counts;
}

function collectCategories(series: AnalyticsSeriesPoint[]): string[] {
  const set = new Set<string>();
  for (const point of series) {
    for (const key of Object.keys(point.rewardTerms)) set.add(key);
  }
  return [...set].sort();
}
