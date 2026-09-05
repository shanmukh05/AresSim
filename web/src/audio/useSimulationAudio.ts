/**
 * Web-audio cues for actions and ambience. Honors `audioMuted` in the store.
 * Sound is presentation; missing AudioContext must not block gameplay.
 */

import { useEffect, useRef } from "react";
import { useAresStore } from "../state/useAresStore";
import type { ActionType } from "../types/sim";

let sharedContext: AudioContext | null = null;

function audioContext() {
  if (typeof window === "undefined") return null;
  if (!window.AudioContext) return null;
  sharedContext ??= new window.AudioContext();
  return sharedContext;
}

function tone(context: AudioContext, frequency: number, offset: number, duration: number, type: OscillatorType, volume: number) {
  const start = context.currentTime + offset;
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  oscillator.type = type;
  oscillator.frequency.setValueAtTime(frequency, start);
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(volume, start + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  oscillator.connect(gain);
  gain.connect(context.destination);
  oscillator.start(start);
  oscillator.stop(start + duration + 0.02);
}

const ACTION_TONES: Record<ActionType, Array<[number, number, number, OscillatorType, number]>> = {
  move: [[155, 0, 0.08, "square", 0.025], [110, 0.08, 0.07, "square", 0.018]],
  scan: [[520, 0, 0.13, "sine", 0.035], [780, 0.12, 0.18, "sine", 0.028]],
  extract: [[120, 0, 0.22, "sawtooth", 0.025], [175, 0.18, 0.12, "square", 0.02]],
  build: [[220, 0, 0.11, "triangle", 0.03], [330, 0.1, 0.12, "triangle", 0.03], [440, 0.2, 0.18, "triangle", 0.028]],
  service: [[392, 0, 0.15, "sine", 0.035], [523, 0.13, 0.15, "sine", 0.035], [659, 0.26, 0.24, "sine", 0.03]],
  unload: [[260, 0, 0.1, "triangle", 0.03], [390, 0.09, 0.12, "triangle", 0.03], [520, 0.2, 0.2, "sine", 0.025]],
  wait: [[205, 0, 0.2, "sine", 0.018]],
  invalid: [[105, 0, 0.12, "sawtooth", 0.035], [82, 0.12, 0.2, "sawtooth", 0.03]],
  event: [[300, 0, 0.1, "sine", 0.02]],
};

function playAction(action: ActionType) {
  const context = audioContext();
  if (!context) return;
  void context.resume();
  for (const [frequency, offset, duration, type, volume] of ACTION_TONES[action] ?? []) {
    tone(context, frequency, offset, duration, type, volume);
  }
}

function playServiceRequest() {
  const context = audioContext();
  if (!context) return;
  void context.resume();
  tone(context, 740, 0, 0.15, "square", 0.028);
  tone(context, 520, 0.18, 0.2, "square", 0.028);
  tone(context, 740, 0.42, 0.15, "square", 0.025);
}

export function useSimulationAudio() {
  const snapshot = useAresStore((state) => state.snapshot);
  const muted = useAresStore((state) => state.audioMuted);
  const lastHistoryId = useRef<string | null>(null);
  const previousServiceNeeded = useRef<boolean | null>(null);

  useEffect(() => {
    const unlock = () => {
      if (!muted) void audioContext()?.resume();
    };
    window.addEventListener("pointerdown", unlock, { once: true });
    return () => window.removeEventListener("pointerdown", unlock);
  }, [muted]);

  useEffect(() => {
    const latest = snapshot?.history[0];
    if (!latest) return;
    if (lastHistoryId.current === null) {
      lastHistoryId.current = latest.id;
      previousServiceNeeded.current = snapshot.buildPadState.serviceNeeded;
      return;
    }
    if (latest.id !== lastHistoryId.current) {
      lastHistoryId.current = latest.id;
      if (!muted) playAction(latest.action);
    }
    const serviceNeeded = snapshot.buildPadState.serviceNeeded;
    if (previousServiceNeeded.current === false && serviceNeeded && !muted) playServiceRequest();
    previousServiceNeeded.current = serviceNeeded;
  }, [muted, snapshot]);
}
