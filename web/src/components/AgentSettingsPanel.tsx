/**
 * Redesigned Agent Settings panel (dev mode), streamlined Agent Picker (deployed mode),
 * and LLM + Decision Model provider/key configuration dialog.
 */

import { useEffect, useMemo, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import {
  Bot,
  Brain,
  Check,
  ChevronDown,
  ChevronUp,
  Cpu,
  Eye,
  EyeOff,
  FolderCheck,
  KeyRound,
  Layers,
  Lock,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import { useAresStore } from "../state/useAresStore";
import {
  ALGORITHMS,
  ALGORITHM_CATALOG,
  CAPABILITY_REQUIREMENTS,
  supportsActionSampling,
  usesCheckpoint,
  type AlgorithmGroup,
  type AlgorithmId,
} from "../lib/algorithms";
import type { LlmDmCatalog, LlmHistoryMode } from "../types/sim";

const GROUP_META: Array<{ id: AlgorithmGroup; label: string; icon: React.ReactNode }> = [
  { id: "baseline", label: "Baselines", icon: <Bot size={12} /> },
  { id: "rl", label: "RL Policies", icon: <Layers size={12} /> },
  { id: "llm", label: "LLM & Decision Models", icon: <Sparkles size={12} /> },
];

/** Full dev-mode Agent Settings panel with visual categorized selector and policy config. */
export function AgentSettingsPanel({ onClose }: { onClose: () => void }) {
  const algorithmId = useAresStore((state) => state.algorithmId);
  const checkpointPath = useAresStore((state) => state.checkpointPath);
  const defaultCheckpoints = useAresStore((state) => state.defaultCheckpoints);
  const policyActionMode = useAresStore((state) => state.policyActionMode);
  const policyAttachKey = useAresStore((state) => state.policyAttachKey);
  const backendBusy = useAresStore((state) => state.backendBusy);
  const rllibAvailable = useAresStore((state) => state.rllibAvailable);
  const muzeroAvailable = useAresStore((state) => state.muzeroAvailable);
  const worldModelsAvailable = useAresStore((state) => state.worldModelsAvailable);
  const llmDmAvailable = useAresStore((state) => state.llmDmAvailable);
  const policyCapabilitiesLoaded = useAresStore((state) => state.policyCapabilitiesLoaded);
  const setAlgorithmId = useAresStore((state) => state.setAlgorithmId);
  const setCheckpointPath = useAresStore((state) => state.setCheckpointPath);
  const setPolicyActionMode = useAresStore((state) => state.setPolicyActionMode);
  const attachCurrentPolicy = useAresStore((state) => state.attachCurrentPolicy);
  const refreshPolicyCapabilities = useAresStore((state) => state.refreshPolicyCapabilities);

  useEffect(() => {
    void refreshPolicyCapabilities();
  }, [refreshPolicyCapabilities]);

  const capabilities = {
    rllib: rllibAvailable,
    muzero: muzeroAvailable,
    world_models: worldModelsAvailable,
    llm_dm: llmDmAvailable,
  };
  const selectedDef = ALGORITHM_CATALOG[algorithmId];
  const bestDefault = typeof defaultCheckpoints[algorithmId] === "string" ? String(defaultCheckpoints[algorithmId]) : null;
  const isAttached = policyAttachKey !== null && policyAttachKey.startsWith(`${algorithmId}:`);

  return (
    <section
      id="agent-settings-panel"
      aria-label="Agent settings"
      data-testid="agent-settings-panel"
      className="pointer-events-auto w-[min(740px,calc(100vw-1.5rem))] overflow-hidden rounded-xl border border-violet-400/25 bg-[#0b0e15]/96 shadow-[0_20px_60px_rgba(0,0,0,0.65)] backdrop-blur-2xl"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onClose();
        }
      }}
    >
      <div className="flex items-center justify-between border-b border-white/8 bg-gradient-to-r from-violet-500/10 via-transparent to-cyan-500/5 px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-lg border border-violet-400/30 bg-violet-500/15 text-violet-200">
            <SlidersHorizontal size={14} />
          </span>
          <div>
            <h2 className="text-xs font-bold tracking-wide text-white">Agent Settings</h2>
            <p className="text-[10px] text-stone-400">Choose a controller and configure inference parameters</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${isAttached
                ? "border-emerald-400/35 bg-emerald-500/12 text-emerald-200"
                : "border-white/10 bg-white/[0.04] text-stone-400"
              }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${isAttached ? "bg-emerald-400" : "bg-stone-500"}`} />
            {isAttached ? "Attached" : "Standby"}
          </span>
          <button
            type="button"
            aria-label="Close agent settings"
            className="grid h-7 w-7 place-items-center rounded-lg text-stone-400 transition hover:bg-white/10 hover:text-white"
            onClick={onClose}
          >
            <X size={15} />
          </button>
        </div>
      </div>

      <div className="max-h-[min(64vh,520px)] space-y-3.5 overflow-y-auto p-4">
        <div className="space-y-3" role="radiogroup" aria-label="Algorithm policy">
          {GROUP_META.map((group) => {
            const items = ALGORITHMS.filter((item) => item.group === group.id);
            return (
              <div key={group.id}>
                <div className="mb-1.5 flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-[0.14em] text-stone-500">
                  {group.icon}
                  {group.label}
                </div>
                <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
                  {items.map((option) => {
                    const blocked =
                      option.capability !== null &&
                      policyCapabilitiesLoaded &&
                      !capabilities[option.capability];
                    const active = algorithmId === option.value;
                    const hasBest = Boolean(defaultCheckpoints[option.value]);
                    return (
                      <button
                        key={option.value}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        disabled={blocked}
                        onClick={() => setAlgorithmId(option.value)}
                        className={`group flex flex-col items-start rounded-lg border px-2.5 py-2 text-left transition ${active
                            ? "border-violet-400/60 bg-violet-500/16 text-white shadow-[0_0_16px_rgba(167,139,250,0.16)]"
                            : blocked
                              ? "cursor-not-allowed border-white/[0.04] bg-white/[0.015] text-stone-600 opacity-55"
                              : "border-white/[0.07] bg-white/[0.025] text-stone-300 hover:border-violet-300/30 hover:bg-white/[0.05] hover:text-white"
                          }`}
                      >
                        <div className="flex w-full items-center justify-between gap-1">
                          <span className="truncate text-[11px] font-semibold">{option.label}</span>
                          {blocked && option.capability ? (
                            <span className="shrink-0 rounded bg-white/5 px-1 py-0.5 text-[8px] text-stone-500">
                              <Lock size={9} />
                            </span>
                          ) : hasBest ? (
                            <span className="shrink-0 rounded bg-emerald-500/15 px-1.5 py-0.5 text-[8px] font-semibold text-emerald-300">
                              Best
                            </span>
                          ) : option.value === "llm_dm" ? (
                            <span className="shrink-0 rounded bg-cyan-500/15 px-1.5 py-0.5 text-[8px] font-semibold text-cyan-200">
                              Live
                            </span>
                          ) : null}
                        </div>
                        <span className="mt-0.5 line-clamp-1 text-[9px] text-stone-500 group-hover:text-stone-400">
                          {blocked && option.capability
                            ? CAPABILITY_REQUIREMENTS[option.capability]
                            : option.description}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {usesCheckpoint(algorithmId) ? (
          <div className="space-y-3 rounded-lg border border-white/[0.08] bg-black/30 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label htmlFor="policy-checkpoint-path" className="text-[10px] font-bold uppercase tracking-[0.12em] text-stone-400">
                Checkpoint path
              </label>
              {bestDefault ? (
                <button
                  type="button"
                  onClick={() => {
                    setCheckpointPath(bestDefault);
                    void attachCurrentPolicy();
                  }}
                  className="inline-flex items-center gap-1 rounded-md border border-emerald-400/25 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-200 transition hover:bg-emerald-500/20"
                >
                  <FolderCheck size={11} />
                  Use best checkpoint
                </button>
              ) : null}
            </div>
            <div className="flex items-center gap-1.5">
              <input
                id="policy-checkpoint-path"
                aria-label="Checkpoint path"
                className="h-9 min-w-0 flex-1 rounded-lg border border-white/[0.09] bg-[#080a0f] px-3 font-mono text-[11px] text-white outline-none transition placeholder:text-stone-600 focus:border-violet-400/50"
                placeholder="/absolute/path/to/checkpoint.json"
                value={checkpointPath}
                onChange={(event) => setCheckpointPath(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void attachCurrentPolicy();
                  }
                }}
              />
              <button
                type="button"
                aria-label="Apply checkpoint path"
                disabled={backendBusy}
                onClick={() => void attachCurrentPolicy()}
                className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-violet-400/40 bg-violet-500/18 px-3 text-[11px] font-semibold text-violet-100 transition hover:bg-violet-500/28 disabled:opacity-40"
              >
                <Check size={14} />
                Load
              </button>
            </div>

            {supportsActionSampling(algorithmId) ? (
              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] pt-2.5">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-stone-400">Action selection</div>
                  <div className="text-[9px] text-stone-500">
                    {selectedDef.actionSelection === "q"
                      ? "Choose ε-greedy exploration or highest Q-value."
                      : "Sample from policy distribution or pick highest probability."}
                  </div>
                </div>
                <div className="inline-flex rounded-lg border border-white/10 bg-[#080a0f] p-0.5" role="group" aria-label="Policy action selection">
                  {(["stochastic", "deterministic"] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      disabled={backendBusy}
                      aria-pressed={policyActionMode === mode}
                      onClick={() => {
                        setPolicyActionMode(mode);
                        void attachCurrentPolicy();
                      }}
                      className={`rounded-md px-2.5 py-1 text-[10px] font-semibold transition ${policyActionMode === mode
                          ? "bg-violet-500/25 text-violet-100 shadow-sm"
                          : "text-stone-400 hover:text-stone-200"
                        }`}
                    >
                      {mode === "stochastic"
                        ? selectedDef.actionSelection === "q"
                          ? "Sampling (ε-greedy)"
                          : "Sampling"
                        : selectedDef.actionSelection === "q"
                          ? "Greedy (Max Q)"
                          : "Deterministic"}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {algorithmId === "llm_dm" ? (
          <LlmDmConfigForm onConnected={onClose} />
        ) : null}
      </div>
    </section>
  );
}

/**
 * Lightweight deployment-mode Agent Picker: no dev settings or checkpoint inputs.
 * Every RL model runs its curated best checkpoint automatically; LLM + DM opens API key setup.
 */
export function DeployedAgentPicker({ onClose, onOpenLlmSetup }: { onClose: () => void; onOpenLlmSetup: () => void }) {
  const algorithmId = useAresStore((state) => state.algorithmId);
  const defaultCheckpoints = useAresStore((state) => state.defaultCheckpoints);
  const rllibAvailable = useAresStore((state) => state.rllibAvailable);
  const muzeroAvailable = useAresStore((state) => state.muzeroAvailable);
  const worldModelsAvailable = useAresStore((state) => state.worldModelsAvailable);
  const llmDmAvailable = useAresStore((state) => state.llmDmAvailable);
  const policyCapabilitiesLoaded = useAresStore((state) => state.policyCapabilitiesLoaded);
  const setAlgorithmId = useAresStore((state) => state.setAlgorithmId);
  const attachCurrentPolicy = useAresStore((state) => state.attachCurrentPolicy);

  const capabilities = {
    rllib: rllibAvailable,
    muzero: muzeroAvailable,
    world_models: worldModelsAvailable,
    llm_dm: llmDmAvailable,
  };

  const handleSelect = (id: AlgorithmId) => {
    setAlgorithmId(id);
    if (id === "llm_dm") {
      onClose();
      onOpenLlmSetup();
      return;
    }
    void attachCurrentPolicy();
    onClose();
  };

  return (
    <section
      aria-label="Choose agent"
      data-testid="deployed-agent-picker"
      className="pointer-events-auto w-[min(560px,calc(100vw-1.5rem))] overflow-hidden rounded-xl border border-violet-400/25 bg-[#0b0e15]/96 shadow-[0_20px_60px_rgba(0,0,0,0.65)] backdrop-blur-2xl"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onClose();
        }
      }}
    >
      <div className="flex items-center justify-between border-b border-white/8 px-3.5 py-2.5">
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-violet-300" />
          <span className="text-xs font-bold text-white">Select Agent</span>
          <span className="rounded-full bg-emerald-500/12 px-2 py-0.5 text-[9px] font-semibold text-emerald-300">
            Best checkpoints preloaded
          </span>
        </div>
        <button
          type="button"
          aria-label="Close agent picker"
          onClick={onClose}
          className="grid h-6 w-6 place-items-center rounded text-stone-400 hover:bg-white/10 hover:text-white"
        >
          <X size={14} />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-1.5 p-3 sm:grid-cols-3">
        {ALGORITHMS.map((option) => {
          const missingCapability =
            option.capability !== null && policyCapabilitiesLoaded && !capabilities[option.capability];
          const missingBestCkpt =
            option.kind === "checkpoint" && policyCapabilitiesLoaded && !defaultCheckpoints[option.value];
          const blocked = missingCapability || missingBestCkpt;
          const active = algorithmId === option.value;
          return (
            <button
              key={option.value}
              type="button"
              disabled={blocked}
              onClick={() => handleSelect(option.value)}
              className={`flex flex-col items-start rounded-lg border px-2.5 py-2 text-left transition ${active
                  ? "border-violet-400/60 bg-violet-500/18 text-white"
                  : blocked
                    ? "cursor-not-allowed border-white/[0.04] bg-white/[0.015] text-stone-600 opacity-50"
                    : "border-white/[0.08] bg-white/[0.025] text-stone-200 hover:border-violet-300/35 hover:bg-white/[0.06]"
                }`}
            >
              <div className="flex w-full items-center justify-between gap-1">
                <span className="truncate text-[11px] font-semibold">{option.label}</span>
                {option.value === "llm_dm" ? (
                  <KeyRound size={11} className="shrink-0 text-cyan-300" />
                ) : null}
              </div>
              <span className="mt-0.5 line-clamp-1 text-[9px] text-stone-500">
                {missingBestCkpt ? "Unavailable on host" : option.description}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

/** Modal dialog wrapper for LLM + DM model & API key configuration (used in deployed mode and from chat). */
export function LlmDmConfigModal({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[min(620px,calc(100vw-2rem))] max-h-[86vh] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-xl border border-violet-400/25 bg-[#0b0e15] p-4 text-stone-100 shadow-2xl">
          <div className="mb-3 flex items-center justify-between border-b border-white/8 pb-3">
            <div className="flex items-center gap-2.5">
              <span className="grid h-8 w-8 place-items-center rounded-lg border border-cyan-400/25 bg-cyan-500/10 text-cyan-200">
                <Brain size={16} />
              </span>
              <div>
                <Dialog.Title className="text-sm font-bold text-white">LLM + Decision Model Setup</Dialog.Title>
                <Dialog.Description className="text-[10px] text-stone-400">
                  Select an Actor (Decision Model or LLM), an optional LLM Learner, and supply API keys.
                </Dialog.Description>
              </div>
            </div>
            <Dialog.Close aria-label="Close LLM-DM setup" className="grid h-7 w-7 place-items-center rounded-lg text-stone-400 hover:bg-white/10 hover:text-white">
              <X size={15} />
            </Dialog.Close>
          </div>
          <LlmDmConfigForm onConnected={() => onOpenChange(false)} />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/** Shared form for configuring LLM/DM Actor, Learner, API keys, and context settings. */
export function LlmDmConfigForm({ onConnected }: { onConnected?: () => void }) {
  const catalog = useAresStore((state) => state.llmDmCatalog);
  const config = useAresStore((state) => state.llmDmConfig);
  const deployedMode = useAresStore((state) => state.deployedMode);
  const backendBusy = useAresStore((state) => state.backendBusy);
  const updateLlmDmConfig = useAresStore((state) => state.updateLlmDmConfig);
  const attachCurrentPolicy = useAresStore((state) => state.attachCurrentPolicy);
  const setLlmFeedOpen = useAresStore((state) => state.setLlmFeedOpen);

  const [actorKind, setActorKind] = useState<"dm" | "llm">(() =>
    config.actorProvider === "decision_models" || config.actorProvider === "openrouter_decisions"
      ? "dm"
      : "llm",
  );
  const [showActorKey, setShowActorKey] = useState(false);
  const [showLearnerKey, setShowLearnerKey] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const fallbackCatalog: LlmDmCatalog = useMemo(
    () =>
      catalog ?? {
        dmProviders: [
          { id: "decision_models", label: "Jev", credentialEnv: "JEV_API_KEY", defaultModel: "jev-1.14.0", placeholder: "jev-1.14.0" },
          { id: "openrouter_decisions", label: "OpenRouter", credentialEnv: "OPENROUTER_API_KEY", defaultModel: "liquid/d1.2-ultra", placeholder: "liquid/d1.2-ultra or openai/gpt-6-luna-decisions" },
          { id: "openai", label: "OpenAI", credentialEnv: "OPENAI_API_KEY", defaultModel: "gpt-6-luna", placeholder: "gpt-6-luna" },
        ],
        llmProviders: [
          { id: "anthropic", label: "Anthropic", credentialEnv: "ANTHROPIC_API_KEY", defaultModel: "claude-haiku-5-5", placeholder: "claude-haiku-5-5" },
          { id: "openai", label: "OpenAI", credentialEnv: "OPENAI_API_KEY", defaultModel: "gpt-6-luna", placeholder: "gpt-6-luna" },
          { id: "gemini", label: "Gemini", credentialEnv: "GEMINI_API_KEY", defaultModel: "gemini-4-neon", placeholder: "gemini-4-neon" },
          { id: "deepseek", label: "DeepSeek", credentialEnv: "DEEPSEEK_API_KEY", defaultModel: "deepseek-flash", placeholder: "deepseek-flash" },
          { id: "openrouter", label: "OpenRouter", credentialEnv: "OPENROUTER_API_KEY", defaultModel: "deepseek/deepseek-v4.1-flash-1017", defaultRoute: "wafer/us", placeholder: "deepseek/deepseek-v4.1-flash-1017" },
        ],
        envKeysConfigured: {},
      },
    [catalog],
  );

  const actorProviderList = actorKind === "dm" ? fallbackCatalog.dmProviders : fallbackCatalog.llmProviders;
  const actorProviderSpec =
    actorProviderList.find((p) => p.id === config.actorProvider) ?? actorProviderList[0];
  const learnerProviderSpec =
    fallbackCatalog.llmProviders.find((p) => p.id === config.learnerProvider) ?? fallbackCatalog.llmProviders[0];

  const actorCredEnv = actorProviderSpec?.credentialEnv ?? "JEV_API_KEY";
  const learnerCredEnv = learnerProviderSpec?.credentialEnv ?? "GEMINI_API_KEY";
  const sameCredEnv = config.learnerEnabled && actorCredEnv === learnerCredEnv;

  const actorEnvReady = !deployedMode && Boolean(fallbackCatalog.envKeysConfigured[actorCredEnv]);
  const learnerEnvReady = !deployedMode && Boolean(fallbackCatalog.envKeysConfigured[learnerCredEnv]);

  const switchActorKind = (nextKind: "dm" | "llm") => {
    setActorKind(nextKind);
    const list = nextKind === "dm" ? fallbackCatalog.dmProviders : fallbackCatalog.llmProviders;
    const match = list.find((item) => item.id === config.actorProvider) ?? list[0];
    if (match && match.id !== config.actorProvider) {
      updateLlmDmConfig({
        actorProvider: match.id,
        actorModel: match.defaultModel,
        actorRoute: match.defaultRoute ?? null,
      });
    }
  };

  const handleConnect = async () => {
    await attachCurrentPolicy();
    setLlmFeedOpen(true);
    if (useAresStore.getState().policyAttachKey) {
      onConnected?.();
    }
  };

  const parseNum = (raw: string, fallback: number) => {
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  };

  return (
    <div className="space-y-3 rounded-lg border border-cyan-400/15 bg-black/30 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.12em] text-cyan-200">
          <Sparkles size={12} />
          Actor & Learner Pipeline
        </div>
        <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[9px] text-stone-400">
          <ShieldCheck size={10} className="text-emerald-300" />
          {deployedMode ? "Keys stay in session memory only" : "Dev mode: uses .env.local keys automatically"}
        </span>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {/* Actor Column (3 rows: Provider, Model ID, API Key) */}
        <div className="space-y-2.5 rounded-lg border border-white/[0.07] bg-[#080a0f]/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-200">1. Actor</span>
            <div className="flex items-center gap-1.5">
              {actorEnvReady ? (
                <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[8px] font-semibold text-emerald-300">
                  .env
                </span>
              ) : null}
              <div className="inline-flex rounded-md border border-white/10 bg-[#0d1119] p-0.5" role="group" aria-label="Actor type">
                {(["dm", "llm"] as const).map((kind) => (
                  <button
                    key={kind}
                    type="button"
                    aria-pressed={actorKind === kind}
                    onClick={() => switchActorKind(kind)}
                    className={`rounded px-2 py-0.5 text-[9px] font-bold uppercase transition ${
                      actorKind === kind
                        ? "bg-cyan-500/25 text-cyan-100"
                        : "text-stone-500 hover:text-stone-200"
                    }`}
                  >
                    {kind}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Row 1: Provider */}
          <div className="space-y-1">
            <label htmlFor="actor-provider-select" className="block text-[9px] font-semibold uppercase tracking-wide text-stone-400">
              Provider
            </label>
            <select
              id="actor-provider-select"
              aria-label="Actor provider"
              className="h-8 w-full rounded-md border border-white/10 bg-[#0d1119] px-2 text-[11px] font-medium text-white outline-none focus:border-cyan-400/50"
              value={actorProviderSpec?.id ?? config.actorProvider}
              onChange={(event) => {
                const next = actorProviderList.find((p) => p.id === event.target.value);
                updateLlmDmConfig({
                  actorProvider: event.target.value,
                  actorModel: next?.defaultModel ?? "",
                  actorRoute: next?.defaultRoute ?? null,
                });
              }}
            >
              {actorProviderList.map((item) => (
                <option key={item.id} value={item.id} className="bg-[#0b0e15]">
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          {/* Row 2: Model ID (+ optional Route for OpenRouter LLM) */}
          <div className="space-y-1">
            <label htmlFor="actor-model-input" className="block text-[9px] font-semibold uppercase tracking-wide text-stone-400">
              Model ID
            </label>
            <div className="flex gap-1.5">
              <input
                id="actor-model-input"
                aria-label="Actor model ID"
                type="text"
                autoComplete="off"
                placeholder={actorProviderSpec?.placeholder ?? "Enter model ID"}
                value={config.actorModel}
                onChange={(event) => updateLlmDmConfig({ actorModel: event.target.value })}
                className="h-8 min-w-0 flex-1 rounded-md border border-white/10 bg-[#0d1119] px-2.5 font-mono text-[10.5px] text-white outline-none placeholder:text-stone-600 focus:border-cyan-400/50"
              />
              {config.actorProvider === "openrouter" ? (
                <input
                  aria-label="Actor OpenRouter route"
                  type="text"
                  autoComplete="off"
                  placeholder="route (wafer/us)"
                  value={config.actorRoute ?? ""}
                  onChange={(event) => updateLlmDmConfig({ actorRoute: event.target.value || null })}
                  className="h-8 w-24 shrink-0 rounded-md border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none placeholder:text-stone-600 focus:border-cyan-400/50"
                />
              ) : null}
            </div>
          </div>

          {/* Row 3: API Key */}
          <div className="space-y-1">
            <label htmlFor="actor-key-input" className="flex items-center justify-between text-[9px] font-semibold uppercase tracking-wide text-stone-400">
              <span>API Key ({actorCredEnv})</span>
              <span className="normal-case text-[8.5px] text-stone-500">
                {actorEnvReady ? "Optional override" : "Required"}
              </span>
            </label>
            <div className="flex items-center rounded-md border border-white/10 bg-[#0d1119] px-2 focus-within:border-cyan-400/50">
              <KeyRound size={11} className="mr-1.5 shrink-0 text-stone-500" />
              <input
                id="actor-key-input"
                aria-label={`Actor API key (${actorCredEnv})`}
                type={showActorKey ? "text" : "password"}
                autoComplete="off"
                placeholder={actorEnvReady ? "Using server .env key" : `Paste ${actorCredEnv}`}
                value={config.actorApiKey ?? ""}
                onChange={(event) => updateLlmDmConfig({ actorApiKey: event.target.value })}
                className="h-8 min-w-0 flex-1 bg-transparent font-mono text-[10px] text-white outline-none placeholder:text-stone-600"
              />
              <button
                type="button"
                aria-label={showActorKey ? "Hide actor key" : "Show actor key"}
                onClick={() => setShowActorKey((v) => !v)}
                className="p-1 text-stone-500 hover:text-stone-200"
              >
                {showActorKey ? <EyeOff size={12} /> : <Eye size={12} />}
              </button>
            </div>
          </div>
        </div>

        {/* Learner Column (3 rows: Provider, Model ID, API Key) */}
        <div className="space-y-2.5 rounded-lg border border-white/[0.07] bg-[#080a0f]/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <label className="flex cursor-pointer items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-violet-200">
              <input
                type="checkbox"
                checked={config.learnerEnabled}
                onChange={(event) => updateLlmDmConfig({ learnerEnabled: event.target.checked })}
                className="accent-violet-400"
              />
              2. Learner (LLM)
            </label>
            {config.learnerEnabled && learnerEnvReady ? (
              <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[8px] font-semibold text-emerald-300">
                .env
              </span>
            ) : null}
          </div>

          {/* Row 1: Provider */}
          <div className="space-y-1">
            <label htmlFor="learner-provider-select" className="block text-[9px] font-semibold uppercase tracking-wide text-stone-400">
              Provider
            </label>
            <select
              id="learner-provider-select"
              aria-label="Learner provider"
              disabled={!config.learnerEnabled}
              className="h-8 w-full rounded-md border border-white/10 bg-[#0d1119] px-2 text-[11px] font-medium text-white outline-none focus:border-violet-400/50 disabled:opacity-40"
              value={learnerProviderSpec?.id ?? config.learnerProvider}
              onChange={(event) => {
                const next = fallbackCatalog.llmProviders.find((p) => p.id === event.target.value);
                updateLlmDmConfig({
                  learnerProvider: event.target.value,
                  learnerModel: next?.defaultModel ?? "",
                  learnerRoute: next?.defaultRoute ?? null,
                });
              }}
            >
              {fallbackCatalog.llmProviders.map((item) => (
                <option key={item.id} value={item.id} className="bg-[#0b0e15]">
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          {/* Row 2: Model ID (+ optional Route for OpenRouter) */}
          <div className="space-y-1">
            <label htmlFor="learner-model-input" className="block text-[9px] font-semibold uppercase tracking-wide text-stone-400">
              Model ID
            </label>
            <div className="flex gap-1.5">
              <input
                id="learner-model-input"
                aria-label="Learner model ID"
                type="text"
                autoComplete="off"
                disabled={!config.learnerEnabled}
                placeholder={learnerProviderSpec?.placeholder ?? "Enter model ID"}
                value={config.learnerModel}
                onChange={(event) => updateLlmDmConfig({ learnerModel: event.target.value })}
                className="h-8 min-w-0 flex-1 rounded-md border border-white/10 bg-[#0d1119] px-2.5 font-mono text-[10.5px] text-white outline-none placeholder:text-stone-600 focus:border-violet-400/50 disabled:opacity-40"
              />
              {config.learnerProvider === "openrouter" ? (
                <input
                  aria-label="Learner OpenRouter route"
                  type="text"
                  autoComplete="off"
                  disabled={!config.learnerEnabled}
                  placeholder="route (wafer/us)"
                  value={config.learnerRoute ?? ""}
                  onChange={(event) => updateLlmDmConfig({ learnerRoute: event.target.value || null })}
                  className="h-8 w-24 shrink-0 rounded-md border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none placeholder:text-stone-600 focus:border-violet-400/50 disabled:opacity-40"
                />
              ) : null}
            </div>
          </div>

          {/* Row 3: API Key */}
          <div className="space-y-1">
            <label htmlFor="learner-key-input" className="flex items-center justify-between text-[9px] font-semibold uppercase tracking-wide text-stone-400">
              <span>API Key ({learnerCredEnv})</span>
              <span className="normal-case text-[8.5px] text-stone-500">
                {!config.learnerEnabled
                  ? "Disabled"
                  : sameCredEnv
                    ? "Reuses Actor key"
                    : learnerEnvReady
                      ? "Optional override"
                      : "For live learnings"}
              </span>
            </label>
            <div className="flex items-center rounded-md border border-white/10 bg-[#0d1119] px-2 focus-within:border-violet-400/50">
              <KeyRound size={11} className="mr-1.5 shrink-0 text-stone-500" />
              <input
                id="learner-key-input"
                aria-label={`Learner API key (${learnerCredEnv})`}
                type={showLearnerKey ? "text" : "password"}
                autoComplete="off"
                disabled={!config.learnerEnabled || sameCredEnv}
                placeholder={
                  sameCredEnv
                    ? "Shared with Actor key"
                    : learnerEnvReady
                      ? "Using server .env key"
                      : `Paste ${learnerCredEnv}`
                }
                value={sameCredEnv ? "" : config.learnerApiKey ?? ""}
                onChange={(event) => updateLlmDmConfig({ learnerApiKey: event.target.value })}
                className="h-8 min-w-0 flex-1 bg-transparent font-mono text-[10px] text-white outline-none placeholder:text-stone-600 disabled:opacity-40"
              />
              <button
                type="button"
                disabled={!config.learnerEnabled || sameCredEnv}
                aria-label={showLearnerKey ? "Hide learner key" : "Show learner key"}
                onClick={() => setShowLearnerKey((v) => !v)}
                className="p-1 text-stone-500 hover:text-stone-200 disabled:opacity-30"
              >
                {showLearnerKey ? <EyeOff size={12} /> : <Eye size={12} />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Collapsible Advanced Settings (context length, output tokens, history, episode steps) */}
      <div className="rounded-lg border border-white/[0.07] bg-[#080a0f]/60">
        <button
          type="button"
          aria-expanded={advancedOpen}
          onClick={() => setAdvancedOpen((open) => !open)}
          className="flex w-full items-center justify-between px-3 py-2 text-[10px] font-semibold text-stone-300 transition hover:text-white"
        >
          <span className="flex items-center gap-1.5">
            <Settings2 size={12} className="text-stone-400" />
            Context & Runtime Settings
            <span className="font-mono text-[9px] font-normal text-stone-500">
              ({Math.round(config.actorContextWindow / 1024)}k ctx · {config.historyMode} · {config.maxEpisodeSteps} steps)
            </span>
          </span>
          {advancedOpen ? <ChevronUp size={13} className="text-stone-400" /> : <ChevronDown size={13} className="text-stone-400" />}
        </button>

        {advancedOpen ? (
          <div className="grid grid-cols-2 gap-2.5 border-t border-white/[0.06] p-3 sm:grid-cols-4">
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">Actor context</span>
              <input
                type="number"
                min={8192}
                max={2000000}
                step={8192}
                value={config.actorContextWindow}
                onChange={(e) => updateLlmDmConfig({ actorContextWindow: parseNum(e.target.value, 131072) })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none focus:border-cyan-400/50"
              />
            </label>
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">Actor max output</span>
              <input
                type="number"
                min={256}
                max={65536}
                step={256}
                value={config.actorMaxOutputTokens}
                onChange={(e) => updateLlmDmConfig({ actorMaxOutputTokens: parseNum(e.target.value, 2048) })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none focus:border-cyan-400/50"
              />
            </label>
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">Learner context</span>
              <input
                type="number"
                min={16384}
                max={2000000}
                step={8192}
                disabled={!config.learnerEnabled}
                value={config.learnerContextWindow}
                onChange={(e) => updateLlmDmConfig({ learnerContextWindow: parseNum(e.target.value, 262144) })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none focus:border-violet-400/50 disabled:opacity-40"
              />
            </label>
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">Learner max output</span>
              <input
                type="number"
                min={1024}
                max={131072}
                step={1024}
                disabled={!config.learnerEnabled}
                value={config.learnerMaxOutputTokens}
                onChange={(e) => updateLlmDmConfig({ learnerMaxOutputTokens: parseNum(e.target.value, 16384) })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none focus:border-violet-400/50 disabled:opacity-40"
              />
            </label>
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">History mode</span>
              <select
                value={config.historyMode}
                onChange={(e) => updateLlmDmConfig({ historyMode: e.target.value as LlmHistoryMode })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-1.5 text-[10px] text-white outline-none focus:border-cyan-400/50"
              >
                <option value="focused" className="bg-[#0b0e15]">Focused</option>
                <option value="facts_recent" className="bg-[#0b0e15]">Facts + Recent</option>
                <option value="recent" className="bg-[#0b0e15]">Recent</option>
                <option value="none" className="bg-[#0b0e15]">None</option>
              </select>
            </label>
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">Advice / step</span>
              <input
                type="number"
                min={1}
                max={10}
                value={config.adviceLimit}
                onChange={(e) => updateLlmDmConfig({ adviceLimit: parseNum(e.target.value, 3) })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none focus:border-cyan-400/50"
              />
            </label>
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">Timeout (sec)</span>
              <input
                type="number"
                min={10}
                max={300}
                step={5}
                value={config.attemptTimeoutSeconds}
                onChange={(e) => updateLlmDmConfig({ attemptTimeoutSeconds: parseNum(e.target.value, 60) })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none focus:border-cyan-400/50"
              />
            </label>
            <label className="space-y-1 text-[9px] font-semibold text-stone-400">
              <span className="block">Max episode steps</span>
              <input
                type="number"
                min={16}
                max={2000}
                step={16}
                value={config.maxEpisodeSteps}
                onChange={(e) => updateLlmDmConfig({ maxEpisodeSteps: parseNum(e.target.value, 256) })}
                className="h-7 w-full rounded border border-white/10 bg-[#0d1119] px-2 font-mono text-[10px] text-white outline-none focus:border-cyan-400/50"
              />
            </label>
          </div>
        ) : null}
      </div>

      <div className="flex items-center justify-between gap-2 pt-0.5">
        <span className="text-[10px] text-stone-500">Learnings accumulate live across episodes.</span>
        <button
          type="button"
          disabled={backendBusy || !config.actorModel.trim()}
          onClick={() => void handleConnect()}
          className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-cyan-400/45 bg-cyan-500/18 px-3.5 text-[11px] font-bold text-cyan-100 transition hover:bg-cyan-500/28 disabled:opacity-40"
        >
          <Check size={14} />
          Connect & Open Live Feed
        </button>
      </div>
    </div>
  );
}
