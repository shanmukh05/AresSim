/**
 * Twitch-chat-inspired live feed and learnings panel for LLM + Decision Model runs.
 * Displays real-time Actor choices, DM confidence, Goal updates, applied advice,
 * and live Learner-synthesized rules.
 */

import { useEffect, useRef, useState } from "react";
import {
  BookMarked,
  Brain,
  ChevronDown,
  ChevronUp,
  KeyRound,
  Loader2,
  MessageSquareText,
  Sparkles,
  Target,
  Trash2,
  X,
} from "lucide-react";
import { useAresStore } from "../state/useAresStore";
import type { LlmFeedMessage, LlmLearningStatement } from "../types/sim";

export function LlmChatPanel({ onClose, onOpenConfig }: { onClose: () => void; onOpenConfig: () => void }) {
  const feed = useAresStore((state) => state.llmFeed);
  const learnings = useAresStore((state) => state.llmLearnings);
  const currentGoal = useAresStore((state) => state.llmCurrentGoal);
  const lastAppliedIds = useAresStore((state) => state.llmLastAppliedIds);
  const llmDmConfig = useAresStore((state) => state.llmDmConfig);
  const policyAttachKey = useAresStore((state) => state.policyAttachKey);
  const learnerConfigured = useAresStore((state) => state.llmLearnerConfigured);
  const llmLearningBusy = useAresStore((state) => state.llmLearningBusy);
  const llmAutoLearn = useAresStore((state) => state.llmAutoLearn);
  const backendBusy = useAresStore((state) => state.backendBusy);
  const deployedMode = useAresStore((state) => state.deployedMode);
  const setLlmAutoLearn = useAresStore((state) => state.setLlmAutoLearn);
  const runLlmLearn = useAresStore((state) => state.runLlmLearn);
  const clearLlmFeed = useAresStore((state) => state.clearLlmFeed);
  const stepCount = useAresStore((state) => state.snapshot?.step ?? 0);

  const [tab, setTab] = useState<"feed" | "learnings">("feed");
  const [highlightedCode, setHighlightedCode] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const isAttached = Boolean(policyAttachKey?.startsWith("llm_dm:"));

  useEffect(() => {
    if (tab !== "feed" || !scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [feed.length, tab]);

  const jumpToLearning = (code: string) => {
    setHighlightedCode(code);
    setTab("learnings");
  };

  return (
    <section
      aria-label="LLM live stream and learnings"
      data-testid="llm-chat-panel"
      className="flex h-[min(420px,calc(100vh-230px))] w-[min(370px,calc(100vw-2rem))] flex-col overflow-hidden rounded-xl border border-violet-400/25 bg-[#0b0c12]/92 shadow-[0_20px_54px_rgba(0,0,0,0.62)] backdrop-blur-2xl"
    >
      {/* Twitch-style Stream Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-white/8 bg-gradient-to-r from-violet-600/18 via-[#12101d]/90 to-cyan-500/10 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="relative flex h-2.5 w-2.5 shrink-0">
            {isAttached ? (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-400 opacity-75" />
            ) : null}
            <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${isAttached ? "bg-rose-500" : "bg-stone-600"}`} />
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-black uppercase tracking-[0.14em] text-white">LLM-DM Stream</span>
              {backendBusy && isAttached ? (
                <Loader2 size={11} className="animate-spin text-violet-300" />
              ) : null}
            </div>
            <div className="truncate text-[9px] text-stone-400">
              {llmDmConfig.actorModel}
              {llmDmConfig.learnerEnabled ? ` · ${llmDmConfig.learnerModel}` : ""}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            aria-label="Configure LLM-DM models and API keys"
            title="Models & API keys"
            onClick={onOpenConfig}
            className="inline-flex h-6 items-center gap-1 rounded-md border border-cyan-400/25 bg-cyan-500/10 px-1.5 text-[9px] font-semibold text-cyan-200 transition hover:bg-cyan-500/20"
          >
            <KeyRound size={10} />
            Keys
          </button>
          <button
            type="button"
            aria-label="Clear live feed"
            title="Clear feed"
            onClick={clearLlmFeed}
            className="grid h-6 w-6 place-items-center rounded-md text-stone-500 transition hover:bg-white/8 hover:text-stone-200"
          >
            <Trash2 size={11} />
          </button>
          <button
            type="button"
            aria-label="Close LLM stream"
            onClick={onClose}
            className="grid h-6 w-6 place-items-center rounded-md text-stone-400 transition hover:bg-white/10 hover:text-white"
          >
            <X size={13} />
          </button>
        </div>
      </div>

      {/* Tab Strip + Goal Pill */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-white/[0.06] bg-black/30 px-2.5 py-1.5">
        <div className="flex gap-1">
          <button
            type="button"
            onClick={() => setTab("feed")}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-bold transition ${
              tab === "feed"
                ? "bg-violet-500/25 text-violet-100"
                : "text-stone-400 hover:bg-white/5 hover:text-stone-200"
            }`}
          >
            <MessageSquareText size={11} />
            Live Chat
          </button>
          <button
            type="button"
            onClick={() => setTab("learnings")}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-bold transition ${
              tab === "learnings"
                ? "bg-emerald-500/22 text-emerald-100"
                : "text-stone-400 hover:bg-white/5 hover:text-stone-200"
            }`}
          >
            <BookMarked size={11} />
            Learnings
            <span className="rounded-full bg-white/10 px-1.5 text-[9px] tabular-nums">{learnings.length}</span>
          </button>
        </div>

        {currentGoal ? (
          <span className="inline-flex max-w-[140px] items-center gap-1 truncate rounded-full border border-amber-400/25 bg-amber-500/10 px-2 py-0.5 text-[9px] font-semibold text-amber-200">
            <Target size={9} className="shrink-0" />
            <span className="truncate">{currentGoal.kind.replace(/_/g, " ")}</span>
          </span>
        ) : null}
      </div>

      {/* Body */}
      {tab === "feed" ? (
        <div ref={scrollRef} className="min-h-0 flex-1 space-y-1.5 overflow-y-auto px-2.5 py-2 font-sans text-[11px]">
          {!isAttached && feed.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-2.5 px-4 text-center">
              <div className="grid h-10 w-10 place-items-center rounded-full border border-violet-400/25 bg-violet-500/12 text-violet-200">
                <Brain size={18} />
              </div>
              <div className="space-y-1">
                <div className="text-xs font-semibold text-white">Ready to stream LLM-DM</div>
                <p className="text-[10px] leading-relaxed text-stone-400">
                  {deployedMode
                    ? "Enter your Decision Model / LLM API key to watch the agent reason, choose actions, and synthesize learnings live."
                    : "Press Play or Step to start with your .env.local keys, or customize Actor & Learner models."}
                </p>
              </div>
              <button
                type="button"
                onClick={onOpenConfig}
                className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-400/40 bg-cyan-500/15 px-3 py-1.5 text-[10px] font-bold text-cyan-100 transition hover:bg-cyan-500/25"
              >
                <KeyRound size={12} />
                {deployedMode ? "Enter API Keys & Models" : "Configure Models & Keys"}
              </button>
            </div>
          ) : (
            feed.map((msg) => (
              <ChatRow key={msg.id} message={msg} onSelectLearning={jumpToLearning} />
            ))
          )}
        </div>
      ) : (
        <div className="min-h-0 flex-1 space-y-2 overflow-y-auto px-2.5 py-2">
          {learnings.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-[10px] text-stone-400">
              <BookMarked size={18} className="text-stone-600" />
              <p>No learnings yet. Run several steps and click <strong className="text-stone-200">Reflect & Learn</strong> to synthesize a rule from live gameplay.</p>
            </div>
          ) : (
            learnings.map((item) => (
              <LearningCard
                key={item.id}
                learning={item}
                isActiveNow={lastAppliedIds.includes(item.code)}
                isHighlighted={highlightedCode === item.code}
              />
            ))
          )}
        </div>
      )}

      {/* Learner Action Bar */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-t border-white/8 bg-black/40 px-2.5 py-2">
        <label className="flex cursor-pointer items-center gap-1.5 text-[9px] text-stone-400">
          <input
            type="checkbox"
            checked={llmAutoLearn}
            onChange={(event) => setLlmAutoLearn(event.target.checked)}
            className="accent-violet-400"
          />
          Auto-learn on end
        </label>
        <button
          type="button"
          disabled={!isAttached || !learnerConfigured || llmLearningBusy || backendBusy || stepCount === 0}
          onClick={() => void runLlmLearn()}
          title={
            !learnerConfigured
              ? "Configure a Learner LLM & API key to generate learnings"
              : "Ask the Learner LLM to review this episode and propose a new learning"
          }
          className="inline-flex h-7 items-center gap-1.5 rounded-md border border-emerald-400/35 bg-emerald-500/15 px-2.5 text-[10px] font-bold text-emerald-100 transition hover:bg-emerald-500/25 disabled:pointer-events-none disabled:opacity-35"
        >
          {llmLearningBusy ? <Loader2 size={11} className="animate-spin" /> : <Sparkles size={11} />}
          {llmLearningBusy ? "Learning..." : "Reflect & Learn"}
        </button>
      </div>
    </section>
  );
}

function ChatRow({ message, onSelectLearning }: { message: LlmFeedMessage; onSelectLearning: (code: string) => void }) {
  if (message.kind === "learner") {
    return (
      <div className="rounded-lg border border-emerald-400/30 bg-emerald-950/35 p-2 shadow-sm">
        <div className="flex items-center justify-between gap-1.5">
          <div className="flex items-center gap-1.5">
            <span className="rounded bg-emerald-500/25 px-1.5 py-0.5 text-[8px] font-black uppercase tracking-wider text-emerald-200">
              {message.badge ?? "LEARNER"}
            </span>
            <span className="truncate text-[10px] font-bold text-emerald-100">{message.author}</span>
          </div>
          <span className="font-mono text-[8px] text-stone-500">#{message.step}</span>
        </div>
        <p className="mt-1 text-[10.5px] leading-snug text-stone-100">{message.text}</p>
        {message.reason && message.learning ? (
          <p className="mt-1 border-t border-emerald-300/15 pt-1 text-[9.5px] leading-snug text-emerald-200/75">
            Why: {message.reason}
          </p>
        ) : null}
      </div>
    );
  }

  if (message.kind === "goal") {
    return (
      <div className="flex items-start gap-1.5 rounded-md border border-amber-400/15 bg-amber-500/[0.06] px-2 py-1 leading-snug">
        <span className="mt-0.5 shrink-0 font-mono text-[9px] text-stone-500">#{message.step}</span>
        <span className="shrink-0 rounded bg-amber-400/20 px-1 py-0.5 text-[8px] font-bold uppercase text-amber-200">
          GOAL
        </span>
        <span className="min-w-0 text-[10px] text-amber-100/90">{message.text}</span>
      </div>
    );
  }

  if (message.kind === "system") {
    return (
      <div className="rounded-md bg-white/[0.03] px-2 py-1 text-[9.5px] text-stone-400">
        <span className="mr-1.5 font-mono text-[8px] text-stone-500">{message.timestamp}</span>
        <span className="font-semibold text-cyan-200/80">{message.text}</span>
      </div>
    );
  }

  // Actor message (Twitch chat style line)
  const topProbs = message.probabilities
    ? Object.entries(message.probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
    : [];

  return (
    <div className="rounded-md px-1.5 py-1 transition hover:bg-white/[0.04]">
      <div className="flex flex-wrap items-baseline gap-1.5 leading-snug">
        <span className="font-mono text-[9px] text-stone-500">#{message.step}</span>
        <span
          className={`rounded px-1 py-px text-[8px] font-bold uppercase ${
            message.fallback ? "bg-rose-500/25 text-rose-200" : "bg-violet-500/25 text-violet-200"
          }`}
        >
          {message.badge ?? "ACTOR"}
        </span>
        <span className="font-bold text-cyan-300">{message.author}:</span>
        <span className="text-stone-100">{message.text}</span>
        {typeof message.confidence === "number" ? (
          <span className="font-mono text-[9px] text-emerald-300/90">
            {Math.round(message.confidence * 100)}%
          </span>
        ) : null}
        {typeof message.latencySec === "number" ? (
          <span className="font-mono text-[8.5px] text-stone-500">{message.latencySec.toFixed(1)}s</span>
        ) : null}
      </div>

      {topProbs.length > 0 ? (
        <div className="mt-1 flex flex-wrap gap-1 pl-6">
          {topProbs.map(([act, prob]) => (
            <span
              key={act}
              className="rounded border border-white/[0.06] bg-black/35 px-1.5 py-px font-mono text-[8.5px] text-stone-400"
            >
              {act.replace(/_/g, " ")} <span className="text-cyan-200">{Math.round(prob * 100)}%</span>
            </span>
          ))}
        </div>
      ) : null}

      {message.appliedAdvice && message.appliedAdvice.length > 0 ? (
        <div className="mt-1 flex flex-wrap items-center gap-1 pl-6">
          <span className="text-[8.5px] text-stone-500">Applied:</span>
          {message.appliedAdvice.map((adv) => (
            <button
              key={adv.id}
              type="button"
              onClick={() => onSelectLearning(adv.id)}
              title={adv.text}
              className="rounded border border-emerald-400/30 bg-emerald-500/12 px-1.5 py-px font-mono text-[8.5px] font-bold text-emerald-200 hover:bg-emerald-500/25"
            >
              {adv.id}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function LearningCard({
  learning,
  isActiveNow,
  isHighlighted,
}: {
  learning: LlmLearningStatement;
  isActiveNow: boolean;
  isHighlighted: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article
      className={`rounded-lg border p-2.5 transition ${
        isActiveNow
          ? "border-emerald-400/45 bg-emerald-950/30 shadow-[0_0_14px_rgba(16,185,129,0.14)]"
          : isHighlighted
            ? "border-violet-400/50 bg-violet-950/30"
            : "border-white/[0.08] bg-white/[0.025]"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 font-mono text-[9px] font-bold text-emerald-200">
            {learning.code}
          </span>
          <span className="text-[9px] font-semibold uppercase tracking-wider text-stone-400">
            {learning.target.replace("_", " ")}
          </span>
          {isActiveNow ? (
            <span className="rounded-full bg-emerald-400/20 px-1.5 py-0.5 text-[8px] font-bold text-emerald-200">
              ACTIVE NOW
            </span>
          ) : null}
        </div>
        {learning.author ? (
          <span className="truncate text-[8.5px] text-stone-500">{learning.author}</span>
        ) : null}
      </div>

      <p className="mt-1.5 text-[10.5px] leading-relaxed text-stone-100">{learning.text}</p>

      {learning.conditions.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1">
          {learning.conditions.map((cond, idx) => (
            <span
              key={`${cond.field}-${idx}`}
              className="rounded border border-white/[0.07] bg-black/40 px-1.5 py-0.5 font-mono text-[8.5px] text-stone-400"
            >
              {cond.field.split(".").slice(-1)[0]} {formatOp(cond.operator)} {String(cond.value)}
            </span>
          ))}
        </div>
      ) : null}

      {learning.reason ? (
        <div className="mt-1.5">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="inline-flex items-center gap-1 text-[9px] font-semibold text-stone-400 hover:text-stone-200"
          >
            {expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
            {expanded ? "Hide reasoning" : "Show reasoning"}
          </button>
          {expanded ? (
            <p className="mt-1 rounded bg-black/35 p-2 text-[9.5px] leading-relaxed text-stone-300">
              {learning.reason}
            </p>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function formatOp(op: string): string {
  return { eq: "=", lt: "<", lte: "≤", gt: ">", gte: "≥" }[op] ?? op;
}
