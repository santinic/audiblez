# More Natural Voices for audiblez: Kokoro-Internal Tuning vs. Alternative Engines (mid-2026)

*Date: 2026-06-02. Method: extends the existing `docs/tts-narration-research.md` landscape doc with a fresh multi-angle deep-research sweep (naturalness leaderboards, expressive-2026 models, leader-validation, Kokoro-internal levers, long-form stability, Apple-Silicon viability, commercial-ceiling calibration), plus per-model adversarial verdicts. 22 research/verify agents, 14 candidate models re-checked against audiblez's five hard constraints by reading model cards, licenses, and issue trackers. Inline URLs cite each quantitative/comparative claim.*

> **Code-verified before publishing:** audiblez's CLI default narrator *was* `af_sky` — Kokoro grades this **C- with only 1–10 minutes of training audio** — while the GUI defaulted to `af_alloy` (C) and only `gen_text()` used the Grade-A `af_heart`. A confirmed inconsistency, now fixed.
>
> **✅ Implementation status (shipped in this repo):** The "DO NOW" and the voice-blending recommendations below are **implemented**. The default narrator is now `af_heart` (A) everywhere; the CLI/GUI surface authoritative Kokoro quality grades and recommend the best voices; and **voice blending is supported on every backend** (CLI: `-v af_warm` or `-v 'af_bella:60,af_heart:40'`; GUI: curated blends in the dropdown + type-your-own). Blends are encoded as *repetition in Kokoro's comma-average voice string*, which was verified end-to-end on the MLX engine — closing the "blend support on MLX is unverified" caveat. See `audiblez/voices.py` for the resolver and the curated `PRESET_BLENDS`.

---

## TL;DR

- **No open model in mid-2026 simultaneously beats Kokoro-82M on naturalness AND satisfies all five audiblez constraints (preset narrator, multi-hour stability, local, Apple-Silicon-friendly, permissive license).** Every clear naturalness-beater breaks at least one hard constraint ([leaderboard synthesis](https://artificialanalysis.ai/text-to-speech/leaderboard), [offlinetts mirror](https://offlinetts.com/blog/tts-arena-leaderboard-2026/)).
- **Highest-leverage action — do this first, it's free:** make sure audiblez's *default* narrator is **`af_heart` (Grade A) or `af_bella` (Grade A-)**, the only two A-tier American voices in Kokoro's official [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md). The CLI default is currently `af_sky` (Grade C-, **1–10 minutes** of training audio) — switching is a zero-cost, zero-dependency naturalness win.
- **The biggest *within-Kokoro* naturalness lever is voice blending** — average two A-grade style tensors into one fixed "house narrator" `.pt`, which stays a deterministic preset with no drift and no reference clip ([custom-voice docs](https://deepwiki.com/zboyles/Kokoro-82M/5.2-custom-voice-creation), [nazdridoy CLI](https://github.com/nazdridoy/kokoro-tts)). This is the single highest-leverage naturalness customization without leaving the engine.
- **Kokoro's real weakness is expressivity, not stability or basic naturalness.** It scored the *highest* MOS (~4.5) in the Trelis "Tricky TTS" set ([MarkTechPost](https://www.marktechpost.com/2026/05/30/best-text-to-speech-tts-models-in-2026-a-benchmark-based-comparison/)) and runs full books stably (a 3-hour audiobook "without a single noticeable artifact" on `af_bella`, [reviewnexa](https://reviewnexa.com/kokoro-tts-review/)). It just "reads rather than performs" — flat on fiction/dialogue. For **non-fiction**, Kokoro-internal tuning is sufficient; the upgrade payoff is smallest there.
- **The leaderboard gap is real but modest.** On the Artificial Analysis Speech Arena, the only open-weight clear-beaters are Fish Audio S2 Pro (~+60–72 Elo), Step Audio EditX (~+49 Elo), and Voxtral (~+5, near-tied) ([AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard), [offlinetts](https://offlinetts.com/blog/tts-arena-leaderboard-2026/)). A ~50 Elo gap ≈ ~57% head-to-head preference — perceptible, not dramatic. And the arena votes on **≥3-second clips**, which does not measure the hours-long stability audiobooks live or die on.
- **Every "more natural" alternative is autoregressive and/or cloning-first** — exactly the architecture class with a documented "stability hallucination" (repetition + word-omission) failure mode that worsens on long inputs ([arXiv 2508.15442](https://arxiv.org/pdf/2508.15442)). Kokoro is non-autoregressive and structurally sidesteps this class.
- **If you want an optional "more expressive" engine, the one defensible pilot is Orpheus-TTS** (Apache-2.0, 8 true preset voices, no reference clip) — but only as a GPU-oriented opt-in, gated on a full-book stability test, never as the default. It has no Mac-native MLX path and a 2048-token (~13s) window that forces chunking with end-of-clip hallucination risk ([issue #276](https://github.com/canopyai/Orpheus-TTS/issues/276), [issue #178](https://github.com/canopyai/Orpheus-TTS/issues/178)).
- **Recommendation: keep Kokoro-82M as default; ship the free internal naturalness wins (best-voice default + optional blended house narrator + chunk/speed tuning); treat Orpheus as a "watch/optional-engine" item, not a swap.**

---

## Q1 — More natural voices WITHOUT changing engine (Kokoro-internal)

The answer to "are there more natural voices within Kokoro?" is **yes, and they're nearly free to adopt.** Four levers, in descending leverage:

### 1. Pick the best voice (zero cost)
Kokoro's official [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) grades voices by quality + quantity of training audio:
- **`af_heart` = Grade A** (top target quality; community lore says it is itself an internal blend, part of why it grades highest).
- **`af_bella` = Grade A-** (🔥, trained on 10–100h of data).
- These are the **only two A-tier American voices**. Next tier (B-) is `af_nicole`, `bf_emma`.
- Most other voices are **C-range**, often from only 10–100 *minutes* of data. **`af_sky` is C- with only 1–10 *minutes* of training audio.**

**Action:** audiblez's CLI default narrator is `af_sky` (`cli.py:17`); switching it to `af_heart`/`af_bella` is the single cheapest naturalness improvement available — no new dependency, no code beyond a default-string change. For English narration, the GUI should de-emphasize the low-grade voices.

`af_bella` is also the best-documented Kokoro voice for **hours-long stability**: a reviewer ran "a 3-hour audiobook without a single noticeable artifact," scored long-form consistency 9.5/10, and used it for 80% of audiobook projects ([reviewnexa](https://reviewnexa.com/kokoro-tts-review/)).

### 2. Voice blending — build a custom "house narrator" (highest customization leverage)
Kokoro voices are **style-vector embeddings**; a weighted average (or slerp, reported smoother) of two creates a new, fixed voice ([custom-voice docs](https://deepwiki.com/zboyles/Kokoro-82M/5.2-custom-voice-creation), [kokorotts-mixer](https://github.com/yvrjsharma/kokorotts-mixer), [ysharma HF Space](https://huggingface.co/spaces/ysharma/Make_Custom_Voices_With_KokoroTTS)). The `nazdridoy/kokoro-tts` epub-to-audiobook CLI exposes this directly as `--voice 'af_sarah:60,am_adam:40'` ([repo](https://github.com/nazdridoy/kokoro-tts)).

Because the blend is a **static style tensor**, it:
- satisfies the *fixed preset, no reference clip* constraint (compute once, save `.pt`, ship as one narrator ID);
- introduces **no per-token drift** — it inherits the base model's long-form stability;
- costs nothing at inference (identical speed to a single voice).

**Integration note (the seam):** audiblez currently passes `voice` as a plain string straight into `KPipeline(..., voice=voice)` in `build_synthesizer` (`core.py:266-270`). `KPipeline` also accepts a **torch tensor** as `voice=`. To ship a blended narrator you'd precompute `torch.load(...,weights_only=True)` on two voicepacks, average them, and pass the tensor instead of the string. The MLX path (`_build_mlx_synth`, `core.py:283-286`) would need the equivalent tensor support verified in `mlx-audio` — currently unverified. This is the **highest-leverage naturalness customization that keeps you Apache-2.0 and on the engine you already ship.**

### 3. Generation parameters (free)
Per [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) and [clore.ai's guide](https://docs.clore.ai/guides/audio-and-voice/kokoro-tts):
- Voices perform best at **100–200 tokens** (max ~500); they're weak on <10–20-token fragments and **rush on >400 tokens** — mitigate via the `speed` parameter.
- Sentence + semantic splitting (on "and", "but", "however") yields more natural output. audiblez already chunks per chapter and uses spaCy sentence splitting; ensuring chunks land in the 100–200-token sweet spot (not tiny fragments, not >400-token blobs) reduces the paragraph-boundary artifacts and rushing that hurt perceived naturalness.
- Expose a `speed` slider; ensure good phonemization (misaki G2P + espeak-ng installed). Sample rate is fixed at 24kHz.

### 4. Checkpoint choice — stay on v1.0 (do NOT migrate)
**v1.1-zh is NOT a recommended default for English.** It adds 100 Chinese speakers + 3 new English voices (Maple/Sol/Vale; Vale only ~1h synthetic British-female training) but is **"not a strict upgrade"** — it *drops* the A-grade `af_heart`/`af_bella` set and changes tokenization ([v1.1-zh card](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh), [aimodels.fyi](https://www.aimodels.fyi/models/huggingFace/kokoro-82m-v1.0-onnx-onnx-community)). Keep v1.0 for the English default; sample Maple/Sol only as optional alternate timbres.

### The hard ceiling on Q1
Kokoro's core limitation is **flat/neutral prosody and weak emotion** — it "reads the script, not performs it"; emotional depth rated 6.5/10, explicitly disqualified for "fiction audiobooks with heavy dialogue / dramatic swings" ([reviewnexa](https://reviewnexa.com/kokoro-tts-review/), [texttolab](https://texttolab.com/blog/kokoro-tts-review)). **No amount of voice/blend/param tuning adds emotional acting.** For neutral non-fiction, internal tuning is excellent and sufficient. For dialogue-heavy fiction where the user wants true expressiveness, a different engine (Q2) is the only path — at the cost of the other constraints.

---

## Q2 — Higher-quality ALTERNATIVE models

### Comparison table

| Model | Naturalness vs Kokoro | Preset narrator (no clip)? | Long-form stability | Apple Silicon | ROCm | Size / speed | License | Verdict for audiblez |
|---|---|---|---|---|---|---|---|---|
| **Kokoro-82M v1.0** *(baseline)* | Mid-pack blind Elo (~1062); **top MOS ~4.5** clean narration | **Yes — 54 fixed IDs** | **Proven** (3–6h books, only paragraph-seam artifacts) | **Yes — native MLX (`mlx-community/Kokoro-82M-bf16`) + MPS** | **Yes — torch ROCm** | **82M / ~0.7GB; 12–22× RT on Mac** | **Apache-2.0** | **Keep as default — uniquely meets all 5** |
| **Orpheus-TTS** (Canopy) | "uncertain" — asserted more expressive, **not benchmarked** vs Kokoro | **Yes — 8 named (tara, leo…)** | **Weak/risky** — 2048-tok (~13s) window, end-of-clip hallucination ([#276](https://github.com/canopyai/Orpheus-TTS/issues/276)), seams, needs `repetition_penalty` tuning | **No official** — only GGUF+LM Studio (Metal) community hack ([#178](https://github.com/canopyai/Orpheus-TTS/issues/178)) | Not documented | 3B (~40× Kokoro); 1.5–4× RT on M3 (5–15× slower) | **Apache-2.0 (code+weights)** | **Optional GPU engine only** — best constraint-fit *expressive* candidate, but never default |
| **Step Audio EditX** (StepFun) | **Yes** — Elo ~1104–1111 (+49) | **No** — zero-shot clone + emotion-EDIT; needs `--prompt-audio` | Unverified; **30s/inference** design cap, AR drift risk | **No** — CUDA + Linux only | No | 3B (~8GB wts, 12–16GB VRAM) | **Apache-2.0** | **Skip** — best license among beaters, wrong tool (edit/clone, no preset, no Mac) |
| **Fish Audio S2 Pro** | **Yes** — Elo ~1123–1128 (highest open) | **No** — clone + tag control | **Fails** — maintainer admits AR "repeating/missing words hard to solve"; voice drift | Third-party MLX **~0.29× RT (slower than real-time)** | No | ~5B; 30+h compute for a 10h book on Mac | **Fish Research License — non-commercial** | **Skip** — best quality, breaks license + size + preset + Mac |
| **Voxtral TTS** (Mistral) | "uncertain" — UTMOS-v2 4.11 vs EL-Flash 4.09; only **~+5 Elo** | **Yes — 20 named voices** | **Weak/unverified** — 2-min native cap; chunking lives in hosted API; authors admit it "hallucinates and skips words" | **Yes** — MLX 4-bit build (`mlx-community/...-mlx-4bit`) | No | ~4.1B / ≥16GB BF16 | **CC BY-NC 4.0 (weights + voices)** | **Skip** — clears preset+Mac, fails license + unproven long-form |
| **Higgs Audio V2** (Boson) | "uncertain" — "incredible" emotion in demos; not on Arena | **No** — clone, or non-deterministic "smart voice" ([#14](https://github.com/boson-ai/higgs-audio/issues/14)) | **Unproven/risky** — ~6B AR, 8K-tok cap, re-feed-chunk to fight drift | **No** — CUDA-only, 24GB VRAM | No | ~6B / 24GB VRAM | **Boson "Community License" (Llama-derived, NOT Apache; 100k-user gate)** | **Skip** — license mislabeled, cloning-first, Mac-hostile |
| **VibeVoice** (Microsoft) | **No** — Elo ~960 (7B), **below Kokoro** | **No** — reference-clip cloning | **Design** for ~90min but **documented mangling/stray-music/heteronym errors** ([#170](https://github.com/microsoft/VibeVoice/issues/170)) | Weak/unofficial (emerging MLX) | No | 1.5B/7B | **MIT** *(but MS pulled the repo over misuse; "not for production")* | **Skip as default** — long-form *design* but no naturalness win; niche multi-voice only |
| **Chatterbox** (Resemble) | **No** — Elo ~1006, **below Kokoro** (vendor "beats EL" is small-N) | **No** — clone from ~5s clip; presets = wrapper WAVs | **Weak** — gibberish/hallucination ([#97](https://github.com/resemble-ai/chatterbox/issues/97)); MPS broken ([#147](https://github.com/resemble-ai/chatterbox/issues/147)) | Partial/CPU-fallback | Yes (via devnen server) | ~0.5B | **MIT** *(forced Perth watermark)* | **Skip** — ideal license, but below Kokoro + cloning + watermark |
| **Dia 1.6B** (Nari) | **No** for narration | **No** — random voice per run; seed unreliable ([#109](https://github.com/nari-labs/dia/issues/109)) | **Fails** — 5–20s usable, speeds up 1.5–3×, dialogue-oriented | No (CUDA-only) | No | 1.6B / ~10GB VRAM | **Apache-2.0** | **Skip** — great demos, no stable preset |
| **Sesame CSM-1B** | **No** — conversational, base | **No** — "not fine-tuned on any specific voice" | **Fails** — 10s utterance scale; "inconsistent… gibberish" ([disc. #14](https://huggingface.co/sesame/csm-1b/discussions/14)) | **Yes — MLX** (csm-mlx) | No | 1B | Apache-2.0 (gated) | **Skip** — only working-MLX non-Kokoro, but wrong model type |
| **IndexTTS2** (Bilibili) | "uncertain" — SOTA WER/emotion in paper | **No** — clone-only | Unproven; AR; practitioners wrap in STT-verify-retry | No (CUDA; hobbyist hack only) | No | ~5.9GB + Qwen-0.6B | **Conditional bilibili license (apache tag conflicts); [#464](https://github.com/index-tts/index-tts/issues/464) unanswered** | **Skip** — non-commercial weights + clone + Mac miss |
| **Kani-TTS-2** | "uncertain" — MOS 4.3 (marketing only) | Thin/clone-roadmap | **Poor** — degrades >40s (v2)/>15s (v1) | v2 has **no** MLX; only v1 370m does | No | 400M | **v2 = LFM1.0 ($10M cap), NOT Apache** | **Skip** — quality version lacks Apache+MLX; versions don't align |
| StyleTTS2 / MeloTTS / Parler-TTS | **No** — at/below Kokoro (Elo ~879 / flat / drift) | mixed | weak/unmaintained | weak | no | small–2.3B | MIT-ish / MIT / Apache | **Skip** — no naturalness win on the relevant axis |

*Do not compare raw Elo across leaderboards — Artificial Analysis uses a ~870–1220 scale while HuggingFace TTS-Arena-V2 uses ~1342–1574 ([HF Arena](https://tts-agi-tts-arena-v2.hf.space/leaderboard)). Use within-board rank and win-rate.*

### Per-contender detail

**Orpheus-TTS (Canopy Labs) — the only defensible optional engine.** Apache-2.0 on both code and weights ([model card](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)), and the *only* 2026 model that combines a permissive license, **8 genuine fixed preset voices** ("tara" listed first for conversational realism — no reference clip), and a claimed long-form-coherence design ([repo](https://github.com/canopyai/Orpheus-TTS), [codersera Mac guide](https://codersera.com/blog/install-and-run-orpheus-3b-tts-on-macos-a-complete-guide/)). 10-second "tara" demos do sound more expressive than Kokoro. **But** it's a ~3B autoregressive Speech-LLM with a hard 2048-token (~13s audio) window, so a whole book becomes thousands of chunks — each carrying end-of-clip hallucination risk (open [issue #276](https://github.com/canopyai/Orpheus-TTS/issues/276), no fix), audible inter-chunk seams patched only by 50ms crossfades, and a fragile `repetition_penalty` operating point that community servers hardcode to 1.1 because nothing else is stable ([Lex-au/Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI)). Apple Silicon is **not officially supported** ([issue #178](https://github.com/canopyai/Orpheus-TTS/issues/178), open) — only an unofficial GGUF+LM Studio path at ~real-time. Verdict: expose as an **optional, GPU-oriented "more expressive" engine** for NVIDIA users who accept chunked output, gated on a full-book stability test. Not a Mac-native or safer narrator; do not displace Kokoro.

**Step Audio EditX (StepFun) — best license among beaters, wrong tool.** Apache-2.0 and a real ~+49 Elo over Kokoro ([offlinetts](https://offlinetts.com/blog/tts-arena-leaderboard-2026/)). But its technical report explicitly says it "circumvents the need for embedding-based priors" — there is **no preset-voice mode at all**; TTS requires `--prompt-audio <reference.wav>` + transcript per call ([repo](https://github.com/stepfun-ai/Step-Audio-EditX), [arXiv 2511.03601](https://arxiv.org/abs/2511.03601)). It's an audio *editing* model with TTS attached, recommends "under 30 seconds per inference," and is **CUDA + Linux only** (no MLX/MPS/ROCm). Skip.

**Fish Audio S2 Pro — best naturalness, worst fit.** Highest open-weight Elo (~1123–1128, [AA leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard)). Disqualified three ways: (1) the **Fish Audio Research License is non-commercial** — Fish's own ["what we mean by open source"](https://fish.audio/blog/what-we-mean-by-open-source-for-s2/) blog says "open weights, not open source (OSI)"; the "MIT self-host" claim is an aggregator conflation with the older OpenAudio S1. (2) **No preset narrator** — 10–30s clone + tags. (3) The maintainer states on HF that "repeating / missing some words is hard to solve giving its an auto-regressive model" — exactly the audiobook-killer, plus ~0.29× real-time on the only Mac port ([mlx-speech docs](https://github.com/appautomaton/mlx-speech/blob/main/docs/fish-s2-pro.md)). Skip.

**Voxtral TTS (Mistral) — clears preset+Mac, dies on license.** The earlier profile understated it: it ships **20 named preset voices** (no clip) and has a working **MLX 4-bit build** ([model card](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603), [lucataco/voxtral-cli](https://github.com/lucataco/voxtral-cli)), with UTMOS-v2 4.11 edging ElevenLabs Flash v2.5's 4.09 ([Mistral](https://mistral.ai/news/voxtral-tts/)). But both weights and the bundled voices are **CC BY-NC 4.0** — commercial audiobook use needs Mistral's paid API. And long-form is genuinely unverified: a 2-minute native cap, chunking that lives in the hosted API (not the open weights), and authors who concede the base model "hallucinates and skips words." Only ~+5 Elo (near-tied) for ~50× the parameters. Skip.

**Higgs Audio V2 (Boson AI) — license is mislabeled.** Reviewers cite "incredible naturalness" for emotion, but the **weights are NOT Apache-2.0** — they ship under a custom Boson/Llama-3-derived "Community License" (HF tag "Other") with a 100k-active-user commercial gate and a no-improve-other-LLMs clause ([LICENSE](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/blob/main/LICENSE)). It's cloning-first; the only no-clip mode ("smart voice") is **non-deterministic** ([issue #14](https://github.com/boson-ai/higgs-audio/issues/14)). ~6B AR, CUDA-only, 24GB VRAM. Skip.

**VibeVoice / Chatterbox — do NOT actually beat Kokoro on naturalness.** Despite the hype, on the neutral Arena VibeVoice-7B sits at Elo ~960 / 38% win-rate and Chatterbox at ~1006 / ~48% — **both below Kokoro's ~1056 / 54.4%** on a far larger sample ([offlinetts](https://offlinetts.com/blog/tts-arena-leaderboard-2026/), [HF Arena](https://tts-agi-tts-arena-v2.hf.space/leaderboard)). Chatterbox's vendor "63.75% beats ElevenLabs" is a small-N vendor test absent from the blind arena. Both are cloning-first; Chatterbox forces a Perth watermark on every output and is MPS-broken ([issue #147](https://github.com/resemble-ai/chatterbox/issues/147)); Microsoft pulled VibeVoice's code over misuse concerns and labels it not-for-production. VibeVoice is interesting *only* as an optional multi-voice/dialogue engine, never a default.

---

## The long-form catch (why short-clip leaderboards mislead for audiobooks)

This is the crux of the whole decision. **The leaderboards that say "X is more natural than Kokoro" are measured on the wrong thing.**

1. **The arenas vote on ≥3-second / ~500-char clips** ([AA methodology](https://artificialanalysis.ai/text-to-speech/methodology)). They measure single-passage *naturalness*, not the property an audiobook lives on: **stability over hours** (no hallucination, repetition, word-drop, or voice drift). The existing landscape doc already flagged this; the new research confirms it is the decisive evidence gap — no strong *open long-form* benchmark exists.

2. **Every clear naturalness-beater is autoregressive and/or cloning-first**, and AR/LM TTS has an architecture-level "stability hallucination" failure class — "mispronunciation, word omission, repetition" that occur "more frequently when generating long or complex sentences" ([arXiv 2508.15442](https://arxiv.org/pdf/2508.15442), naming VALL-E/CosyVoice/NaturalSpeech; also [arXiv 2509.19852](https://arxiv.org/abs/2509.19852)). The existence of a research subfield dedicated to *suppressing* these is itself the evidence: it's unsolved at the architecture level. Concrete instances: F5-TTS's paper-acknowledged "word-skipping" ([arXiv 2410.06885](https://arxiv.org/abs/2410.06885)); XTTS-v2's documented timbre drift across chunks; Higgs's card warning of hallucination needing post-hoc STT; Fish's maintainer admission; Orpheus's open end-of-clip hallucination.

3. **The tell: practitioners bolt failure-recovery scaffolding around these models.** `petermg/Chatterbox-TTS-Extended` and `zeropointnine/tts-audiobook-tool` wrap Chatterbox/IndexTTS2 in **per-chunk Whisper transcription vs. source text + retry-and-keep-best-take** loops. You only build that machinery because the base model drops/repeats/mangles words over a whole book. audiblez does **not** have that pipeline — adopting any of these means importing it.

4. **Kokoro is non-autoregressive (flow/duration-based), so it structurally cannot loop or runaway-hallucinate** the way AR models do. This is why *every shipping epub→audiobook tool* (audiblez, abogen, Kokoro-FastAPI) converged on it ([abogen](https://github.com/denizsafak/abogen), [claudio.uk](https://claudio.uk/posts/epub-to-audiobook.html)). That convergence is the strongest long-form signal available.

The honest read: a ~50-Elo naturalness bump on 3-second clips can come bundled with a *regression* on the axis that actually determines whether a 10-hour book is usable.

---

## Concrete recommendation for audiblez (tiered)

### DO NOW (free, within Kokoro, no new dependency)
1. **Change the CLI default narrator from `af_sky` to `af_heart` or `af_bella`** (`cli.py:17`; note `gen_text()` already defaults to `af_heart` at `core.py:322`, so this also removes an inconsistency). `af_sky` is Grade C- with 1–10 min training data; the swap is the biggest cheap naturalness win. In the GUI (`ui.py:309`, currently defaults to `af_alloy`), default to an A-grade voice and de-emphasize C-grade voices for English narration.
2. **Tune chunking to the 100–200-token sweet spot** and **expose a `speed` slider** to fix rushing on long passages. audiblez already does per-chapter spaCy splitting in `gen_audio_segments` (`core.py:290+`); the change is keeping chunks out of both the <20-token and >400-token zones.

### OPTIONAL — ship a blended "house narrator" (small code, stays Apache-2.0)
3. **Add a precomputed voice-blend preset** (e.g. `af_bella` + `af_heart`, or a touch of a male voice for depth). This is the highest *naturalness customization* without leaving Kokoro, and it stays a deterministic, drift-free, no-reference-clip preset.
   - **What the seam requires:** in `build_synthesizer` (`core.py:266-270`), `voice` is currently a plain string fed to `KPipeline(..., voice=voice)`. `KPipeline` also accepts a **torch tensor**. Implement a blend by `torch.load(...,weights_only=True)` on two voicepacks, weighted-average them, cache one `.pt`, and pass the tensor. Expose it as a single narrator ID (e.g. `house_warm`). **Verify** that the MLX path (`_build_mlx_synth`, `core.py:283-286`) also accepts a tensor through `mlx_audio`'s `model.generate(..., voice=)` — this is currently unverified and may limit blends to the torch backends.

### OPTIONAL ENGINE — Orpheus, gated (only if the user wants expressive fiction)
4. **Offer Orpheus-TTS as an opt-in "more expressive" backend, NVIDIA/GPU-oriented, NOT default.** It's the only Apache-2.0 + true-preset-narrator + claimed-long-form candidate. Gate adoption on: (a) a **full-book stability test** with a fixed voice (e.g. `tara`), watching for end-of-clip hallucination and inter-chunk seams; (b) accepting ~5–15× slower render and no Mac-native MLX path.
   - **What an engine *addition* actually requires (the seam is well-designed for this):** add a new `BackendInfo` to `backends.BACKENDS` in `backends.py` with a new `engine` value (e.g. `'orpheus'`), and add an `elif info.engine == 'orpheus': return _build_orpheus_synth(...)` branch in `build_synthesizer` (mirroring the existing `_build_mlx_synth`). The closure must honor the exact contract: **`synth(text, speed) -> list[np.ndarray]`, float32 @ 24000 Hz** — Orpheus emits 24kHz mono, which matches. Note the `voice[0]` lang-code assumption in `core.py:256/294` — Orpheus's named voices (`tara`, etc.) don't follow Kokoro's `af_`/`am_` prefix convention, so the lang-code derivation and the per-engine voice list in `voices.py` would need an Orpheus-specific path. Everything downstream (chunking, m4b assembly) is engine-agnostic and unchanged.

### WATCH (revisit if the landscape shifts)
- **Voxtral TTS** — has presets + MLX; revisit only **if Mistral relicenses the TTS weights** off CC BY-NC 4.0.
- **Step Audio EditX** — revisit only if it grows a real preset-voice mode AND an MLX/MPS path.
- **A future non-AR, permissive, preset-narrator model with proven hours-long English stability** — none exists as of June 2026.

**Bottom line:** keep Kokoro-82M as the default. The realistic, defensible naturalness gains for audiblez are the *internal* ones (best-voice default, blended house narrator, chunk/speed tuning), which cost almost nothing and preserve all five constraints. An engine swap buys at most ~+50 Elo on short clips at the cost of 3–40× size, unproven long-form stability, and (for everything except Orpheus) a license or preset-narrator violation.

---

## Caveats & what's still unverified

- **Live leaderboards drift.** The Speech Arena re-runs ~4×/day; the 3rd/4th open-weights spot is volatile (one snapshot put Voxtral *below* Kokoro). Treat all Elo deltas as approximate ([AA methodology](https://artificialanalysis.ai/text-to-speech/methodology)).
- **No open long-form benchmark exists.** Every "long-form" claim here is either a single reviewer anecdote (Kokoro's 3h `af_bella` run) or a *design* claim (VibeVoice 90min), not a controlled hours-long WER/drift study. This is the single biggest evidence gap for the whole space.
- **Kokoro's naturalness "lead" is MOS-based, its mid-pack rank is Elo-based** — different protocols; MOS (~4.5, [MarkTechPost](https://www.marktechpost.com/2026/05/30/best-text-to-speech-tts-models-in-2026-a-benchmark-based-comparison/)) and blind Elo should not be directly compared. Kokoro is genuinely strong on clean neutral narration and only loses on *expressivity*.
- **Orpheus's naturalness edge over Kokoro is asserted, not measured** — it has no published Arena Elo or head-to-head MOS vs Kokoro. A full-chapter A/B listening test is required before any adoption.
- **Blend support on the MLX path is unverified.** Tensor `voice=` works in torch `KPipeline`; whether `mlx-audio`'s `model.generate` accepts a blended tensor was not confirmed.
- **License ambiguities flagged, not resolved.** Fish (Research License vs. aggregator "MIT" claims), IndexTTS2 (apache HF tag vs. conditional bilibili license, [#464](https://github.com/index-tts/index-tts/issues/464) unanswered), and Kani-TTS-2 (LFM1.0 vs. marketing "Apache") all have unresolved conflicts; treat the more restrictive reading as authoritative for a shipped OSS tool.
- **The commercial ceiling (ElevenLabs v3, ~4.5–4.6 MOS, performed prosody) is out of scope** (closed/paid) but defines what "more natural" means to the user; the best *open* model is ~96 Elo below the commercial leader ([AA](https://artificialanalysis.ai/text-to-speech/leaderboard), [soloa](https://soloa.ai/blog/tts-models-ranked-realism)). No local swap should be sold as "indistinguishable from human."

---

## Key sources (deduped)

**Leaderboards & benchmarks**
- https://artificialanalysis.ai/text-to-speech/leaderboard
- https://artificialanalysis.ai/text-to-speech/methodology
- https://offlinetts.com/blog/tts-arena-leaderboard-2026/
- https://tts-agi-tts-arena-v2.hf.space/leaderboard
- https://www.marktechpost.com/2026/05/30/best-text-to-speech-tts-models-in-2026-a-benchmark-based-comparison/
- https://soloa.ai/blog/tts-models-ranked-realism
- https://sureprompts.com/blog/voice-generation-models-compared-2026
- https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-part-2

**Kokoro (internal levers)**
- https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- https://huggingface.co/hexgrad/Kokoro-82M
- https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
- https://reviewnexa.com/kokoro-tts-review/
- https://texttolab.com/blog/kokoro-tts-review
- https://deepwiki.com/zboyles/Kokoro-82M/5.2-custom-voice-creation
- https://github.com/nazdridoy/kokoro-tts
- https://github.com/yvrjsharma/kokorotts-mixer
- https://huggingface.co/spaces/ysharma/Make_Custom_Voices_With_KokoroTTS
- https://docs.clore.ai/guides/audio-and-voice/kokoro-tts
- https://github.com/denizsafak/abogen
- https://claudio.uk/posts/epub-to-audiobook.html
- https://github.com/remsky/Kokoro-FastAPI

**Long-form stability (architecture evidence)**
- https://arxiv.org/pdf/2508.15442
- https://arxiv.org/abs/2509.19852
- https://arxiv.org/abs/2410.06885 (F5-TTS word-skipping)
- https://github.com/zeropointnine/tts-audiobook-tool
- https://github.com/petermg/Chatterbox-TTS-Extended

**Alternative models**
- Orpheus: https://github.com/canopyai/Orpheus-TTS · https://huggingface.co/canopylabs/orpheus-3b-0.1-ft · https://github.com/canopyai/Orpheus-TTS/issues/276 · https://github.com/canopyai/Orpheus-TTS/issues/178 · https://github.com/Lex-au/Orpheus-FastAPI · https://codersera.com/blog/install-and-run-orpheus-3b-tts-on-macos-a-complete-guide/
- Step Audio EditX: https://github.com/stepfun-ai/Step-Audio-EditX · https://huggingface.co/stepfun-ai/Step-Audio-EditX · https://arxiv.org/abs/2511.03601
- Fish S2 Pro: https://huggingface.co/fishaudio/s2-pro · https://fish.audio/blog/what-we-mean-by-open-source-for-s2/ · https://github.com/appautomaton/mlx-speech/blob/main/docs/fish-s2-pro.md
- Voxtral TTS: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 · https://mistral.ai/news/voxtral-tts/ · https://github.com/lucataco/voxtral-cli
- Higgs Audio V2: https://github.com/boson-ai/higgs-audio · https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/blob/main/LICENSE · https://github.com/boson-ai/higgs-audio/issues/14
- VibeVoice: https://github.com/microsoft/VibeVoice · https://huggingface.co/microsoft/VibeVoice-1.5B · https://github.com/microsoft/VibeVoice/issues/170
- Chatterbox: https://github.com/resemble-ai/chatterbox · https://github.com/resemble-ai/chatterbox/issues/97 · https://github.com/resemble-ai/chatterbox/issues/147 · https://github.com/devnen/Chatterbox-TTS-Server
- Dia: https://github.com/nari-labs/dia · https://github.com/nari-labs/dia/issues/109
- Sesame CSM: https://huggingface.co/sesame/csm-1b · https://github.com/senstella/csm-mlx
- IndexTTS2: https://github.com/index-tts/index-tts/issues/464 · https://arxiv.org/pdf/2506.21619
- Kani-TTS-2: https://huggingface.co/nineninesix/kani-tts-2-en · https://www.liquid.ai/lfm-license

**Apple Silicon / MLX**
- https://github.com/Blaizzy/mlx-audio
- https://github.com/Blaizzy/mlx-audio-swift
- https://github.com/waybarrios/vllm-mlx/blob/main/docs/benchmarks/audio.md
