# 0001 — Engine track (MOSS) is the headline; multi-speaker → phase-2

Status: Accepted — 2026-06-18 (digest of CROSS_PROJECT_ANALYSIS.md)

## Context
- The pre-run brief **locked multi-speaker** (per-character voices) as THE headline.
- The adversarial analysis downgraded multi-speaker to **med** payoff: Kokoro flat affect,
  English-only (booknlp), **chapter-alignment unsolved** (the v1 "whole book = one chapter"
  fallback silently breaks per-chapter `.wav` naming, `.sig` resume, `.failed.jsonl`, and m4b
  markers), and voice quality degrades past ~3 characters.
- The same analysis rated the **engine track** (MOSS-TTS-8B via llama.cpp-Vulkan + zero-shot
  cloning) **high**, and stated that cloning is what lifts multi-speaker from coarse to
  genuinely distinct.

## Decision
- Build the **MOSS engine track first** as the de-facto headline.
- **Demote multi-speaker to phase-2**, riding on MOSS cloning (per-character cloned voices).
  Its enablers — sml voice-switch tags and the per-sentence cache voice-override — defer with
  it (they were only ever enablers for multi-speaker).
- Phase-2's real blocker is **chapter-alignment** (engine-independent); it gets its own spike
  before any commitment. Target shape: MOSS clone-ref-per-character; the cheaper
  Kokoro-voice-id casting from the spec remains a fallback shape.

## Consequences
- The user's locked headline is **reversed** — recorded here so the reversal is legible later.
- The *good* (cloned) version of multi-speaker becomes reachable because the engine lands first.
- See also [0002](0002-moss-resident-coprocess.md), [0003](0003-moss-default-engine.md).
