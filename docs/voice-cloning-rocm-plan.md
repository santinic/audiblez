# Cloning your podcast narrator for audiblez on an RX 7800 XT (ROCm/RDNA3)

*Date: 2026-06-02. Method: synthesis of verified per-angle findings + adversarial per-model verdicts, re-judged under your fine-tune + ROCm-Linux constraint (Mac/NVIDIA retired). Every quantitative/comparative claim carries an inline source URL; ROCm/gfx1101 specifics are flagged where unverified.*

---

## TL;DR

**Fine-tune a StyleTTS2-family model — specifically Stylish-TTS (or vanilla StyleTTS2 if you want the more battle-tested trainer) — and serve it back through audiblez as a Kokoro-style voicepack on the inference path you already have working.** This is the recommendation because your top two priorities (fidelity to one narrator, then rock-solid stability over a whole book) point at the *same architectural class* — non-autoregressive (NAR) TTS — and that class is *also* the one that survives ROCm/RDNA3 best, because it has **no flash-attention and no bitsandbytes dependency** ([the two libraries that most often break ROCm TTS training on gfx1101](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1608)). StyleTTS2 is the literal architecture behind Kokoro, which you already run and like ([Kokoro "built on the StyleTTS2 architecture", Apache-2.0, 1st in HF TTS Arena](https://huggingface.co/hexgrad/Kokoro-82M)).

**Realistic quality expectation:** after fine-tuning on your tens of hours, the voice will be a *faithful, natural single-speaker clone* — StyleTTS2 reaches "near perfection" on a single speaker with ~4h of clean data ([authors/community guidance](https://github.com/yl4579/StyleTTS2/discussions/128)), and you have far more than that. It will sound clearly more natural than Piper and will not drift, hallucinate, repeat, or drop words over hours the way an autoregressive (AR) model does — because it generates each sentence in one parallel pass with explicit durations ([NAR avoids AR "repetition and word omission" / "cascading generation failures"](https://arxiv.org/pdf/2306.07691)).

**Realistic effort expectation:** the *honest* cost is (1) a one-time multi-day fine-tune on the 7800 XT at a tight-but-workable 16 GB, and (2) you will be an *early adopter on the ROCm training path* — no one has published a StyleTTS2/Stylish-TTS fine-tune on gfx1101, so budget time for a smoke test before committing the full corpus. Inference is the easy, de-risked part.

**If you want the lowest-risk path instead of the highest-fidelity one, train Piper (VITS) first** — it's the most likely to "just work" on your box and never blow up over a book ([pure-PyTorch VITS, no flash-attn/bnb, runs on 8 GB, mature trainer](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)) — at the cost of a flatter, more synthetic voice. A sensible plan does both: ship Piper as the dependable baseline while you get the StyleTTS2 fine-tune converging.

---

## Can Kokoro do it?

**No — not for cloning a *new* narrator.** Kokoro is what audiblez already uses, and it's excellent stock, but it is fundamentally a **preset-voice model**: you pick from its shipped voicepacks. It has **no speaker encoder and no zero-shot cloning path**, and critically **no official training/fine-tuning code** — the model card only points you at the upstream StyleTTS2 architecture repo ([hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)). The only community fine-tune recipes (e.g. [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts), [johal.in fast-LoRA](https://johal.in/kokoro-82m-tts-python-fast-fine-tune/)) are aimed at *adding a new language* or *voice-mixing*, not at faithfully reproducing one specific narrator.

So cloning your narrator needs a **second engine** bolted in alongside Kokoro. The good news is that audiblez's design makes this clean. In `audiblez/core.py`, `build_synthesizer(voice, backend)` returns a closure `synth(text, speed) -> list[np.ndarray]` at 24 kHz float32 mono, and `audiblez/backends.py` selects the engine via `BackendInfo.engine` (currently `'torch'` or `'mlx'`). Everything downstream — sentence chunking in `gen_audio_segments`, `np.concatenate`, the ffmpeg m4b pipeline — is engine-agnostic. **`build_synthesizer` is the single insertion point**: a new engine is one more branch returning the same `synth` contract.

Strategically, the cleanest endgame uses Kokoro's family to your advantage: **fine-tune StyleTTS2/Stylish-TTS on your narrator, then repackage the result as a Kokoro-style voicepack** so it drops into audiblez's *existing* Kokoro inference path unchanged — giving you StyleTTS2's single-speaker trainer for fidelity AND Kokoro's already-working, drift-free long-form inference ([the StyleTTS2↔Kokoro family relationship is direct](https://github.com/yl4579/StyleTTS2)).

---

## Recommended model to fine-tune

### The pick: a StyleTTS2-family single-speaker fine-tune (Stylish-TTS, with vanilla StyleTTS2 as the safer-tooling fallback)

**The evidence, against your priority order:**

**(1) Fidelity — high, and data is a non-issue.** StyleTTS2 is consistently described as very natural and expressive, and the authors' own fine-tune config "finetunes on LJSpeech with 1 hour of speech" with community reports that "30 minutes... 4 hours of data achieving near perfection" ([config_ft.yml + discussion #128](https://github.com/yl4579/StyleTTS2/discussions/128); [dagshub StyleTTS2 writeup](https://dagshub.com/blog/styletts2)). Your tens of hours of one clean narrator is *abundant* — well past the point of diminishing returns, so fidelity is bounded by the model, not your data. **Stylish-TTS** is purpose-built for exactly your goal: "training high quality, single-speaker text-to-speech models (rather than zero-shot voice cloning)" ([Stylish-TTS README](https://github.com/Stylish-TTS/stylish-tts/blob/main/README.md)).

**(2) Long-form stability — structurally the right class.** StyleTTS2 is non-autoregressive: it models duration explicitly and synthesizes per utterance, so there is **no token-by-token loop to accumulate error** — none of the AR drift/hallucination/repetition/word-drop that plagues XTTS, F5, GPT-SoVITS, Orpheus over hours ([NAR "ensures system stability by explicitly modeling the duration of phonetic units"](https://arxiv.org/pdf/2306.07691); [stability-hallucination failure mode formalized for AR LLM-TTS](https://arxiv.org/pdf/2509.19852)). Stylish-TTS goes further with an explicitly-trained duration predictor and the *stated design goal* of "consistent text-to-speech results for long-form text and screen reading" ([README](https://github.com/Stylish-TTS/stylish-tts/blob/main/README.md)). Honest caveat: that long-form consistency is a *design goal with no published multi-hour benchmark*, and the realistic residual failure mode is inter-sentence timbre/prosody jumps, not runaway drift — which audiblez's per-sentence chunking already isolates.

**(3) ROCm trainability — the best-positioned of the high-fidelity options.** The decisive fact: a GitHub code search across Stylish-TTS finds **zero references to flash-attn, bitsandbytes, or xformers** — the classic ROCm training blockers are simply absent; StyleTTS2 likewise needs only PyTorch + a small `monotonic_align` Cython extension that builds against whatever torch is installed (no CUDA-only code) ([StyleTTS2 deps](https://github.com/yl4579/StyleTTS2); [llm-tracker StyleTTS2 guide](https://llm-tracker.info/howto/StyleTTS-2-Setup-Guide)). And gfx1101 (your card) is now **officially ROCm-supported on native Linux since ROCm 6.4.1** ([AMD engineer confirmation, ROCm discussion #2599](https://github.com/ROCm/ROCm/discussions/2599)) and remains in the current compatibility matrix ([rocm.docs.amd.com compatibility-matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)). The one real Stylish-TTS hurdle is `k2` (an FST/alignment lib whose helper script only resolves CUDA/CPU wheels) — but k2's GPU path is *never actually used* (its CTC loss is CPU-pinned and there is a non-k2 `torch` alignment fallback), so you build k2 for CPU once or skip it ([Stylish-TTS losses.py / cli.py](https://github.com/Stylish-TTS/stylish-tts/blob/main/src/stylish_tts/train/cli.py)). If you'd rather avoid the k2 build entirely, **vanilla StyleTTS2 has no k2 at all** and a more documented fine-tune path — it's the lower-friction sibling.

**(4) Fine-tune effort + tooling — moderate and well-trodden.** Both ship clear recipes (StyleTTS2: official `config_ft.yml` from the LibriTTS multispeaker checkpoint, plus the [IIEleven11/StyleTTS2FineTune](https://github.com/IIEleven11/StyleTTS2FineTune) WhisperX→SRT→phonemize pipeline; Stylish-TTS: a 4-stage Acoustic→Textual→Style→Duration recipe). Data prep tooling is mature and reusable across any pick.

**What the fine-tuned voice will sound like:** a competent, natural re-creation of your narrator's *timbre and general speaking style*, synthesized cleanly per sentence at 24 kHz. It is more natural and expressive than Piper, and it is *stable* — you can render a whole book unattended without the "regenerate this line 5 times" dance that AR cloners force. It will not perfectly reproduce a dramatic *performance* (NAR prosody is good, not theatrical), but for faithful long-form narration of your own books it is the best balance of the four priorities.

**Why it beats the alternatives for THIS setup:** F5-TTS and XTTS clone beautifully but are AR/flow models whose makers literally warn about reliability ([F5 author: "I would not recommend this model if you need reliability"](https://github.com/SWivid/F5-TTS/discussions/881); [XTTS documented end-of-sentence hallucinations](https://github.com/coqui-ai/TTS/discussions/4146)) — they fight your #2 priority and need a verify-and-retry harness to be book-safe. Orpheus, Fish-Speech, VibeVoice, and the Llasa/Spark/Zonos family either depend on the broken-on-RDNA3 bitsandbytes/flash-attn stack, lack a working trainer, or have open fine-tune-gibberish bugs (see the table). StyleTTS2-family is the only option that is simultaneously *high-fidelity*, *architecturally stable*, *ROCm-dependency-clean*, and *the same family you already serve*.

---

## Alternatives table

| Model | Fine-tune maturity | Voice fidelity | Long-form stability | ROCm train (gfx1101) | ROCm infer | Data needs | License | Verdict for this user |
|---|---|---|---|---|---|---|---|---|
| **StyleTTS2 / Stylish-TTS** *(PICK)* | Medium — official `config_ft.yml`; IIEleven11 + Stylish 4-stage recipe ([1](https://github.com/yl4579/StyleTTS2/discussions/128),[2](https://github.com/Stylish-TTS/stylish-tts/blob/main/README.md)) | **High** — "~4h near-perfect" single speaker ([3](https://github.com/yl4579/StyleTTS2/discussions/128)) | **High (architectural)** — NAR + explicit duration, no AR drift ([4](https://arxiv.org/pdf/2306.07691)) | **Plausible, dep-clean, UNVERIFIED on gfx1101** — no flash-attn/bnb; only hurdle = k2 (CPU build or skip) ([5](https://github.com/Stylish-TTS/stylish-tts/blob/main/src/stylish_tts/train/cli.py)) | Yes — pure torch (~2GB) or ONNX path ([6](https://dagshub.com/blog/styletts2)) | 0.5–4h plenty; you have far more ([3](https://github.com/yl4579/StyleTTS2/discussions/128)) | MIT code; disclosure clause on weights — fine personal ([7](https://github.com/yl4579/StyleTTS2)) | **TOP PICK.** Best fidelity+stability+ROCm-cleanliness; same family as Kokoro |
| **Kokoro-82M** | **Inference mature; fine-tune unofficial** ([8](https://huggingface.co/hexgrad/Kokoro-82M)) | Excellent stock; new-narrator clone is the gap ([8](https://huggingface.co/hexgrad/Kokoro-82M)) | **Best-in-class** — NAR, 3h book artifact-free ([9](https://github.com/semidark/kikiri-tts)) | No official trainer (use StyleTTS2's) | **Highest-confidence — already wired in audiblez** ([10](https://github.com/ROCm/ROCm/discussions/2599)) | Voice-mix: 0; fine-tune: ~hours | **Apache-2.0** (fully permissive) ([8](https://huggingface.co/hexgrad/Kokoro-82M)) | **Use NOW to prove the ROCm inference path; repackage your StyleTTS2 fine-tune as a Kokoro voicepack** |
| **Piper (VITS)** | **Highest** — mature `piper.train fit` recipe ([11](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)) | **Medium** — ~3.5 MOS, flatter/robotic vs 4.1–4.3 ([12](https://www.promptquorum.com/power-local-llm/local-tts-voice-cloning-piper-coqui-xtts)) | **Highest** — NAR, zero hallucination over hours ([11](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)) | **Most credible** — pure torch, runs on 8 GB / RX 7600 ([11](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)) | **ROCm-independent** — ONNX, even CPU realtime ([11](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)) | 30min works; you're over-provisioned ([13](https://github.com/rhasspy/piper/blob/master/TRAINING.md)) | **GPL-3.0** (active fork; orig MIT archived 2025-10) ([14](https://github.com/OHF-Voice/piper1-gpl)) | **STRONG BASELINE — train first to de-risk; flatter voice is the trade** |
| **F5-TTS** | Medium-high — `finetune_cli`/gradio ([15](https://github.com/SWivid/F5-TTS)) | **High** — "almost spot-on" timbre ([16](https://github.com/SWivid/F5-TTS/discussions/881)) | **Medium** — flow-matching, ~8-10s reliable, drifts long; author warns "not reliable" ([16](https://github.com/SWivid/F5-TTS/discussions/881)) | Plausible, dep-clean (no mandatory FA/bnb), **train on gfx1101 unwitnessed**; use SDPA not flash-attn ([17](https://github.com/SWivid/F5-TTS/discussions/769)) | **Confirmed on RDNA3** (7900 XTX) ([18](https://github.com/SWivid/F5-TTS/discussions/771)) | 10-15h ideal; you have more ([16](https://github.com/SWivid/F5-TTS/discussions/881)) | Code MIT; weights CC-BY-NC — fine personal ([15](https://github.com/SWivid/F5-TTS)) | Strong fidelity, but needs chunk+ASR-verify; AR-ish weakness on #2 |
| **XTTS v2 (Coqui)** | **High** — alltalk wiki, AdamW ([19](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version))) | High — expressive ([20](https://github.com/idiap/coqui-ai-TTS)) | **Low-medium** — AR, documented hallucinations; chunk mandatory ([21](https://github.com/coqui-ai/TTS/discussions/4146)) | **Inference confirmed on 7800 XT**; *fine-tune on RDNA3 UNVERIFIED* (XTTS-WebUI-ROCm ships incomplete ROCm steps) ([22](https://github.com/erew123/alltalk_tts/issues/132),[23](https://github.com/YellowRoseCx/XTTS-WebUI-ROCm)) | Yes ([22](https://github.com/erew123/alltalk_tts/issues/132)) | 3-10min floor; tens of hours ideal ([19](https://github.com/erew123/alltalk_tts/wiki/XTTS-Model-Finetuning-Guide-(Simple-Version))) | Code MPL-2.0; weights CPML (NC) — fine personal ([24](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)) | Best-documented AR fine-tune, but loses on #2; needs verify harness |
| **GPT-SoVITS** | Medium-high — 2-stage GUI trainer ([25](https://github.com/RVC-Boss/GPT-SoVITS)) | High few-shot clone ([26](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7))) | **Low-medium** — AR GPT stage, EOS/truncation bug ([27](https://github.com/RVC-Boss/GPT-SoVITS/issues/1992)) | Workable — `--device ROCM`, gfx1101 community guide; least battle-tested AMD path ([28](https://grzegorz.ie/post/_GPT_SoVITS_ROCM)) | Yes (ROCM path) ([28](https://grzegorz.ie/post/_GPT_SoVITS_ROCM)) | 1min–many hours ([25](https://github.com/RVC-Boss/GPT-SoVITS)) | **MIT code** (cleanest); weights vary ([25](https://github.com/RVC-Boss/GPT-SoVITS)) | Strong fidelity, AR liability; prefer v2Pro + strict chunking |
| **Orpheus-TTS (3B)** | Best tooling (Unsloth) ([29](https://unsloth.ai/blog/tts)) | High ([29](https://unsloth.ai/blog/tts)) | **Poor** — decoder-LLM drift; open EOS hallucination bug ([30](https://github.com/canopyai/Orpheus-TTS/issues/276)) | **Broken easy path** — bnb 4-bit NaN on every AMD GPU; FA backward unsupported on RDNA3; 3B 16-bit tight in 16GB ([31](https://unsloth.ai/docs/get-started/install/amd),[32](https://github.com/Dao-AILab/flash-attention/issues/2452)) | Shaky — vLLM unofficial, segfaults on 7800 XT ([33](https://llm-tracker.info/howto/AMD-GPUs)) | Few hours ([29](https://unsloth.ai/blog/tts)) | Apache-2.0 ([34](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained)) | **AVOID** — its strength (Unsloth 4-bit) is its weakness on AMD |
| **IndexTTS2** | **No official v2 trainer** (issue #518 open) ([35](https://github.com/index-tts/index-tts/issues)) | SOTA zero-shot; fine-tune shows speaker bleed ([36](https://github.com/index-tts/index-tts/issues/501)) | Unvalidated long-text (AR) ([37](https://arxiv.org/html/2506.21619v1)) | Not viable today (no trainer) ([35](https://github.com/index-tts/index-tts/issues)) | Plausible — clean deps, gfx1101 listed ([38](https://github.com/index-tts/index-tts)) | Unknown | **Non-commercial** ([38](https://github.com/index-tts/index-tts)) | **FALLBACK — zero-shot inference engine only, not a fine-tune target** |
| **Fish-Speech / OpenAudio S1-mini** | Medium — bespoke loralib; open gibberish bug ([39](https://github.com/fishaudio/fish-speech/issues/1136)) | High zero-shot; fine-tune at risk ([39](https://github.com/fishaudio/fish-speech/issues/1136)) | Weak — AR, repetition "hard to solve" ([40](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini/discussions/3)) | **Hard blocker** — CUDA-only upstream, KV-cache overflows 16GB; ROCm port only tested on RDNA4 ([41](https://github.com/fishaudio/fish-speech/issues/1246)) | Unverified on RDNA3 ([41](https://github.com/fishaudio/fish-speech/issues/1246)) | ~22min for ~7h ([42](https://github.com/fishaudio/fish-speech/blob/main/docs/en/finetune.md)) | CC-BY-NC-SA-4.0 ([43](https://huggingface.co/fishaudio/openaudio-s1-mini)) | **AVOID** — stacked unsolved problems |
| **VibeVoice (0.5B/1.5B/7B)** | Low — "unofficial WIP LoRA" ([44](https://github.com/voicepowered-ai/VibeVoice-finetuning)) | High zero-shot ([45](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)) | Best *by design* (~90min single-pass) but seed-fragile, music hallucinations ([46](https://github.com/microsoft/VibeVoice/issues/185)) | Not realistic — CUDA-only tooling; 0.5B has no trainer; 1.5B needs ~16GB (no headroom) ([44](https://github.com/voicepowered-ai/VibeVoice-finetuning)) | **Demonstrated poor** — slower-than-realtime on stronger 7900 XTX ([46](https://github.com/microsoft/VibeVoice/issues/185)) | Unknown | MIT-ish (verify per-weight) ([47](https://github.com/vibevoice-community/VibeVoice)) | **AVOID** — three strikes for now; watch-list |
| **Llasa / Spark / Zonos** | Mixed — Llasa full-FT exists; Zonos has no trainer ([48](https://huggingface.co/blog/Steveeeeeeen/llasagna)) | High ([48](https://huggingface.co/blog/Steveeeeeeen/llasagna)) | Weak — AR LLM-codec, no alignment ([49](https://arxiv.org/abs/2509.19852)) | **Risky** — Llasa recipe needs FA2+bnb-8bit, ~24GB > 16GB; Spark-0.5B best-fit but short-form ([50](https://unsloth.ai/docs/get-started/install/amd)) | Partial/unverified ([50](https://unsloth.ai/docs/get-started/install/amd)) | Few hours LoRA | Varies (mostly permissive) ([48](https://huggingface.co/blog/Steveeeeeeen/llasagna)) | **AVOID** — no edge over Orpheus for this box |

---

## The ROCm / RX 7800 XT reality

This is the make-or-break axis, so here is the concrete, honest state of it. **The 2024-era "the 7800 XT isn't supported, hack around it" framing no longer applies** — but training-on-this-exact-card-with-these-exact-TTS-recipes remains *unwitnessed*, so verify empirically.

**Your GPU is gfx1101 (Navi 32), NOT gfx1102.** A popular 2026 consumer-GPU blog repeatedly mislabels the 7800 XT as gfx1102 — that is **wrong**; gfx1102 is the RX 7600 (Navi 33). AMD's own arch-spec table lists RX 7800 XT / 7700 XT (Navi 32) as gfx1101 ([rocm.docs.amd.com gpu-arch-specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html); [the blog that gets it wrong](https://www.kunalganglani.com/blog/rocm-consumer-gpu-cuda-alternative-2026)). This matters because every `PYTORCH_ROCm_ARCH` / `HSA_OVERRIDE` / build flag must target gfx1101; using gfx1102 silently produces no-op kernels.

**gfx1101 is officially ROCm-supported on native Linux as of ROCm 6.4.1 (mid-2025)** and stays listed through the current 7.x matrix ([AMD engineer @schung-amd, ROCm discussion #2599](https://github.com/ROCm/ROCm/discussions/2599); [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)). Native Linux is your axis and it is solid.

**`HSA_OVERRIDE_GFX_VERSION=11.0.0` is generally NOT needed anymore** for the core PyTorch-ROCm stack, because gfx1101 kernels are now built natively — the override (which masquerades as gfx1100) was an *older-ROCm* workaround. Keep it as a **per-library escape hatch** for any dependency that only ships gfx1100 binaries; do not set it globally by reflex ([ROCm discussion #2599](https://github.com/ROCm/ROCm/discussions/2599); [alltalk_tts #132](https://github.com/erew123/alltalk_tts/discussions/132)). Most older guides still recommend it reflexively, so this is the one place to test native-first.

**Install path (Arch — cleanest option):** Arch ships an **official `extra`-repo `python-pytorch-rocm`** (~2.12 as of 2026), which pulls `rocm-hip-sdk`, `hipblaslt`, `miopen-hip`, `amdsmi`, etc. — no pip-wheel/ROCm-version fight ([archlinux.org python-pytorch-rocm](https://archlinux.org/packages/extra/x86_64/python-pytorch-rocm/)). Alternative is AMD's pip wheels (`--index-url https://download.pytorch.org/whl/rocm6.x`). `torchaudio` is bundled and needs no special config. Caveat: Arch is not an AMD-validated distro, so when a ROCm bump breaks something the fix is wait-for / downgrade the package ([AMD pytorch-install docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)).

**Two easy-to-miss prerequisites:** a modern mainline kernel (6.10–6.12+; Arch's rolling kernel satisfies this — crucial RDNA3 amdgpu/kfd fixes landed there) and your user in the **`render` and `video`** groups with access to `/dev/kfd` + `/dev/dri` ([augustin-laurent gist](https://gist.github.com/augustin-laurent/d29f026cdb53a4dff50a400c129d3ea7); [ROCm discussion #2599](https://github.com/ROCm/ROCm/discussions/2599)).

**Which training deps work vs. break:**
- **Works:** plain PyTorch eager-mode training (conv/matmul/GAN ops), `monotonic_align` (Cython, builds against ROCm torch), DeepSpeed single-GPU (a 7800 XT user ran DeepSpeed 0.16.2 on PyTorch 2.7.0.dev+rocm6.3 — though that was *inference*; multi-GPU is where RDNA3 has problems, irrelevant to your single card) ([alltalk_tts #132](https://github.com/erew123/alltalk_tts/discussions/132); [llm-tracker AMD-GPUs](https://llm-tracker.info/howto/AMD-GPUs)).
- **Breaks / avoid:** **bitsandbytes** is the weakest link — official ROCm wheels cover gfx90a/gfx942/gfx1100 only (**not gfx1101**), source builds hit "undefined symbol" ([bitsandbytes issue #1608](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1608)). **flash-attention** CK path is broken on RDNA3 (Wave32 vs CK's Wave64); the working route is the Triton backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) or PyTorch SDPA — but **the FA *backward* pass is unsupported on RDNA3**, so for *training* use plain SDPA/eager, never flash-attn ([craftrigs ROCm 2026](https://craftrigs.com/articles/amd-rocm-local-llm-2026/); [FA issue #2452](https://github.com/Dao-AILab/flash-attention/issues/2452)). **The single biggest reason to pick StyleTTS2-family/Piper: they need neither library.** That sidesteps both landmines entirely.

**16 GB VRAM headroom:** comfortably enough to *fine-tune* small/medium single-speaker TTS — VRAM is **not** your constraint, dataset prep and lib-compat are. StyleTTS2 infers in ~2GB; F5 fine-tuned on a 12GB laptop GPU; Piper fine-tunes within 8GB ([F5 discussion #769](https://github.com/SWivid/F5-TTS/discussions/769); [dagshub StyleTTS2](https://dagshub.com/blog/styletts2); [piper TRAINING.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)). The one tight spot: StyleTTS2/Stylish-TTS **stage-2 / acoustic** training. Stylish-TTS lists 16GB as its *minimum*, so you're at the edge — set `vram_reserve`, cap `probe_batch_max`, reduce `batch_size` to ~2–4 and `max_len` ~100–200, and lengthen training rather than widening the batch ([Stylish-TTS README](https://github.com/Stylish-TTS/stylish-tts/blob/main/README.md); [StyleTTS2 discussion #128 T4 guidance](https://github.com/yl4579/StyleTTS2/discussions/128)). StyleTTS2 stage-2 is also genuinely **NaN-prone even on CUDA** ([issue #200](https://github.com/yl4579/StyleTTS2/issues/200)) — keep batch small, do a smoke test, and consider skipping the SLM adversarial stage (`joint_epoch > epochs`) if it OOMs, accepting a small quality trade.

**The honest unverified core:** no one has published a StyleTTS2 / Stylish-TTS / Piper *fine-tune* specifically on a 7800 XT / gfx1101. Every component is independently confirmed (PyTorch-ROCm training on gfx1101; these models training on consumer GPUs; gfx1101 official support), but the exact end-to-end combo is **you-go-first**. That's why the plan front-loads a smoke test.

---

## End-to-end plan

### 1) Data prep: podcast → clean single-speaker LJSpeech corpus

Build this once; it feeds any model you pick. Fixed pipeline ([TAGARELA dataset paper](https://arxiv.org/html/2603.15326v1); [HF audio-course tts_datasets](https://huggingface.co/learn/audio-course/chapter6/tts_datasets)):

1. **Music/ad removal (selective):** run **Demucs v4** or **UVR RoFormer** to pull the vocal stem on music/ad segments only — separation leaves faint artifacts, so don't blanket-apply to clean speech. Both run on torch-ROCm (~7GB) ([Demucs](https://github.com/facebookresearch/demucs); [UVR RoFormer 2025](https://vocalremover.cloud/blog/uvr-best-model-aug-2025)).
2. **Diarize + drop guests/overlap:** **pyannote speaker-diarization-3.1** (pure PyTorch, 6-8GB, demoed on ROCm by AMD) to isolate your narrator; use overlap detection to **drop co-speech** — overlap leakage is the #1 dataset corruptor ([pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1); [AMD ROCm speech blog](https://rocm.blogs.amd.com/artificial-intelligence/speech_models/README.html)).
3. **VAD-segment + transcribe + align:** **WhisperX** (large-v3) collapses transcribe+align+diarize in one tested-on-RDNA3 pass — but it needs a **ROCm-patched CTranslate2** (upstream is CUDA-only; this is the biggest install landmine). The verified RDNA3 stack: ROCm 7.2 + PyTorch 2.8.0+rocm + CTranslate2 4.1.0 ROCm-fork + faster-whisper 1.2.1 + WhisperX 3.7.4 + pyannote 3.4.0 ([whisperX discussion #1364](https://github.com/m-bain/whisperX/discussions/1364); [BoredYama whisperX-AMD fork](https://github.com/BoredYama/whisperX-AMD-ROCM7.1/tree/main)). **Fallback if CTranslate2-ROCm breaks:** plain OpenAI Whisper or whisper.cpp (`WHISPER_HIPBLAS=1`) run on ROCm with no patched deps, ~4-6x slower — fine for a one-time offline build ([danielrosehill gist](https://gist.github.com/danielrosehill/f7ae5e659a02e32d056eb7887203b7a4)).
4. **Normalize + format:** cut on word-timestamp boundaries (avoid mid-word clicks), enforce **1-15s clips** (3-10s default), resample to **24000 Hz** (match audiblez — avoids a downstream resample), loudness-normalize per clip, and write transcripts **with terminal punctuation** (`. ? !`) or the model can't learn to stop. Emit LJSpeech `metadata.csv` (`filename|text`) ([futurebeeai preprocess guide](https://www.futurebeeai.com/knowledge-hub/preprocess-tts-dataset); [voicelab dataset prep](https://voicelab.ai/how-to-prepare-a-dataset-for-neural-text-to-speech-part-1-text-preparation)).

**How many hours:** for *fine-tuning* (not from-scratch), **clip quality beats raw hours** — 30-60min clean already gives strong fidelity, a few hours is high quality, and 20min clean beats 2h noisy ([F5 fine-tune guide](https://instavar.com/blog/ai-production-stack/F5_TTS_Fine_Tuning_Voice_Cloning_Guide); [Coqui discussion #2935](https://github.com/coqui-ai/TTS/discussions/2935)). **Target ~3-10h of your cleanest clips and discard the rest.** Note Stylish-TTS specifically wants **≥25h** of pairs, so if you go that route, confirm you actually have ≥25h after cleaning ([Stylish-TTS README](https://github.com/Stylish-TTS/stylish-tts/blob/main/README.md)).

There is **no turnkey podcast-to-TTS tool** — assemble it. WhisperX collapses 3 steps; the rest is ~100 lines of glue (overlap-drop, loudness, resample, LJSpeech export). Budget ~2-3 days of setup plus unattended compute ([gokhaneraslan tts-dataset-generator as LJSpeech-export reference](https://github.com/gokhaneraslan/tts-dataset-generator)).

### 2) Fine-tune on the 7800 XT

1. **Install:** Arch `python-pytorch-rocm` (extra repo) + `espeak-ng` + `phonemizer`. Put yourself in `render`/`video` groups; confirm `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"` reports `True` and a HIP version (HIP exposes as `torch.cuda`). Try **native gfx1101 first**; only set `HSA_OVERRIDE_GFX_VERSION=11.0.0` if a specific lib fails.
2. **Build the small native dep:** `monotonic_align` (StyleTTS2) builds as a normal C extension against ROCm torch. For **Stylish-TTS**, build `k2` for CPU (`-DK2_WITH_CUDA=OFF`) or use the `torch` alignment fallback — its GPU path is never used.
3. **Smoke test FIRST:** run **1-2 epochs on a 30-60min subset** to flush out NaN / OOM / build failures *before* committing the full corpus. This is the single most important de-risking step given the unverified ROCm-train path.
4. **Full run — recipe:** StyleTTS2: start from the LibriTTS multispeaker checkpoint via `config_ft.yml`, `batch_size` 2-4, `max_len` ~100-200, mixed-precision off if NaN appears, single-GPU (DDP is broken for stage-2 anyway). Stylish-TTS: the 4-stage Acoustic(10ep)→Textual(10ep)→Style(50ep)→Duration(50ep) recipe. **Plain AdamW, bf16/fp16 — no bitsandbytes, no flash-attn.**
5. **Time/VRAM estimate:** official StyleTTS2 LJSpeech fine-tune was ~4h on **4× A100** ([discussion #128](https://github.com/yl4579/StyleTTS2/discussions/128)); on a single 16GB 7800 XT with small batches expect a **multi-day one-time run**. VRAM fits at reduced batch; the slow acoustic/stage-2 phases are the tight spots — watch OOM, set `vram_reserve`. (Piper, by contrast, fine-tunes in *hours* at `batch_size` 32 — which is why it's the de-risk baseline.)
6. **Select by listening, not by loss curve** — long-form TTS quality doesn't track loss cleanly ([stability-hallucination paper guidance](https://arxiv.org/pdf/2509.19852)).

### 3) Long-form stability scaffolding (chunk + STT-verify-retry)

audiblez **already chunks per sentence** in `gen_audio_segments` and concatenates — that's the right granularity and the reason NAR models drop in cleanly. For a NAR pick (StyleTTS2/Piper/Kokoro) this is *sufficient* and **no verify loop is required**. If you ever serve an AR/flow engine (F5/XTTS), add the proven verify harness as a thin wrapper *outside* the engine — it's engine-agnostic ([zeropointnine/tts-audiobook-tool](https://github.com/zeropointnine/tts-audiobook-tool); [petermg/Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended)):

```
for attempt in range(max_retries):
    wav = synth(chunk, speed)                    # list[np.ndarray] @24kHz
    txt = faster_whisper.transcribe(wav)         # run on CPU — CTranslate2 lacks mature ROCm GPU support
    wer = word_errors(normalize(txt), normalize(chunk))
    track best (min WER); break if wer <= threshold
# fallback: keep min-WER take, else longest-transcript / highest-similarity
```

Run the verifier Whisper on **CPU** (`small`/`base.en` is enough for WER checks) to keep the 16GB free for TTS ([integration-stability findings](https://github.com/zeropointnine/tts-audiobook-tool)). This STT-verify-and-keep-best stage is audiblez's differentiation opportunity — the existing epub-audiobook tools (ebook2audiobook, epub2tts) chunk but don't verify ([DrewThomasson/ebook2audiobook](https://github.com/DrewThomasson/ebook2audiobook)).

### 4) audiblez integration

The seam is already perfect for this. Concretely:

**a) Register the backend** in `audiblez/backends.py` — add a `BackendInfo` whose `engine` selects the new generator, keeping `torch_device` consistent (your card uses the existing `'rocm'` torch backend, which already maps to `torch_device='cuda'` because HIP exposes via `torch.cuda`). If you serve the fine-tune via ONNX (Stylish-TTS export, or Piper), you can add a lightweight `engine='onnx'` that doesn't even need torch at inference:

```python
'styletts2': BackendInfo('styletts2', 'StyleTTS2 (custom narrator)', 'styletts2', 'cuda'),
# or, for the ONNX inference path that dodges torch-ROCm entirely:
'piper':     BackendInfo('piper',     'Piper (custom narrator)',     'onnx',      None),
```

**b) Add the engine branch** in `core.build_synthesizer`, mirroring the existing `_build_mlx_synth` pattern — import the heavy lib lazily, load the fine-tuned checkpoint in the closure, return `synth(text, speed) -> list[np.ndarray]`:

```python
def _build_styletts2_synth(voice, lang_code):
    from styletts2 import tts            # lazy import, only when selected
    model = tts.load(checkpoint_for(voice))
    def synth(text, speed):
        wav = model.inference(text, speed=speed)   # 24kHz mono float32
        return [np.asarray(wav, dtype=np.float32).reshape(-1)]
    return synth
```

**c) Sample rate:** StyleTTS2/Kokoro/F5/XTTS are **24 kHz native** — return directly, no resample (audiblez's `sample_rate = 24000` already matches). **Piper emits 22,050 Hz** — the one adaptation is a `22050→24000` resample (`soxr`/`librosa`) inside the Piper `synth` shim before returning.

**d) Voice-id / lang-code handling:** the closure currently resolves the voice spec via `voicelib.voice_lang_code(voice)` / `voicelib.kokoro_voice_string(voice)`. For a custom narrator, the simplest clean approach is to **repackage the fine-tune as a Kokoro-style voicepack** so it flows through the *existing* path with a new voice id and `lang_code='a'` (American English) — no new branch needed, and you inherit Kokoro's proven, drift-free ROCm inference. Otherwise, map your narrator's voice id to its checkpoint path inside the new `_build_*_synth` closure and hardcode `lang_code='a'` for your single-English-speaker case. Everything downstream (chunking, m4b, chapter metadata) is untouched.

---

## Risks & honest caveats

- **The ROCm-train path is unwitnessed for your exact model+card.** Every building block is confirmed, but no published StyleTTS2/Stylish-TTS/Piper fine-tune on gfx1101 exists. The smoke test in step 2.3 is non-negotiable. If StyleTTS2 stage-2 won't converge on 16GB, fall back to Piper (which is far more likely to train cleanly) and accept the fidelity trade.
- **StyleTTS2 stage-2 is NaN-prone even on NVIDIA** ([issue #200](https://github.com/yl4579/StyleTTS2/issues/200)) and 16GB is exactly the floor — expect parameter tuning (batch 2-4, reduced max_len, possibly skipping SLM adversarial training). This is the genuine reason it's "strong pick" not "guaranteed easy."
- **Stylish-TTS is Alpha** with possible breaking changes — pin a commit ([README](https://github.com/Stylish-TTS/stylish-tts/blob/main/README.md)). Its long-form consistency is a *design goal*, not a benchmarked guarantee.
- **CTranslate2-ROCm (for WhisperX data prep / any verify loop) is a community fork that breaks on ROCm bumps** — keep the plain-Whisper / whisper.cpp fallback ready ([whisperX discussion #1364](https://github.com/m-bain/whisperX/discussions/1364)).
- **Long-form drift only matters if you stray from NAR.** Stick to StyleTTS2-family or Piper and drift is structurally absent; pick an AR model (F5/XTTS/GPT-SoVITS/Orpheus) and you *must* build the verify-retry harness or accept word-drop over a book.
- **Data quality is the real fidelity ceiling.** Podcast overlap leakage, surviving music beds, and inconsistent loudness silently poison the dataset and cause glitches — invest in clean diarization + overlap-drop over collecting more hours.
- **Consent / personal-use line:** this is for *your own listening only*, which is squarely within every relevant license — CPML (XTTS), CC-BY-NC (F5), MIT/Apache/GPL (StyleTTS2/Kokoro/Piper) all permit personal/hobby use ([CPML](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt); [Kokoro Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M)). The ethical line is real: do **not** redistribute the fine-tuned model or its audio, monetize it, or pass it off as the narrator's own work. Cloning a real person's voice without consent for anything beyond private use carries legal and ethical exposure — keep it personal, keep it private.

---

## Key sources (deduped)

**ROCm / gfx1101 / RDNA3:**
- https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html (gfx1101 = 7800 XT, not gfx1102)
- https://github.com/ROCm/ROCm/discussions/2599 (official gfx1101 Linux support since 6.4.1; HSA_OVERRIDE)
- https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- https://archlinux.org/packages/extra/x86_64/python-pytorch-rocm/
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
- https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1608 (bnb no gfx1101 wheel)
- https://github.com/Dao-AILab/flash-attention/issues/2452 (FA backward unsupported on RDNA3)
- https://craftrigs.com/articles/amd-rocm-local-llm-2026/ ; https://llm-tracker.info/howto/AMD-GPUs
- https://gist.github.com/augustin-laurent/d29f026cdb53a4dff50a400c129d3ea7 (Arch kernel/groups)

**Recommended model (StyleTTS2 family / Kokoro):**
- https://github.com/yl4579/StyleTTS2 ; https://github.com/yl4579/StyleTTS2/discussions/128 ; https://github.com/yl4579/StyleTTS2/issues/200
- https://github.com/Stylish-TTS/stylish-tts/blob/main/README.md ; https://github.com/Stylish-TTS/stylish-tts/blob/main/src/stylish_tts/train/cli.py
- https://github.com/IIEleven11/StyleTTS2FineTune ; https://llm-tracker.info/howto/StyleTTS-2-Setup-Guide
- https://huggingface.co/hexgrad/Kokoro-82M ; https://github.com/semidark/kikiri-tts
- https://arxiv.org/pdf/2306.07691 (NAR vs AR stability) ; https://dagshub.com/blog/styletts2

**Alternatives:**
- https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md ; https://github.com/rhasspy/piper/blob/master/TRAINING.md
- https://github.com/SWivid/F5-TTS/discussions/769 ; https://github.com/SWivid/F5-TTS/discussions/771 ; https://github.com/SWivid/F5-TTS/discussions/881
- https://github.com/erew123/alltalk_tts/discussions/132 ; https://github.com/idiap/coqui-ai-TTS ; https://github.com/coqui-ai/TTS/discussions/4146 ; https://github.com/YellowRoseCx/XTTS-WebUI-ROCm
- https://github.com/RVC-Boss/GPT-SoVITS ; https://grzegorz.ie/post/_GPT_SoVITS_ROCM ; https://github.com/RVC-Boss/GPT-SoVITS/issues/1992
- https://unsloth.ai/docs/get-started/install/amd ; https://github.com/canopyai/Orpheus-TTS/issues/276
- https://github.com/index-tts/index-tts/issues/501 ; https://github.com/fishaudio/fish-speech/issues/1246 ; https://github.com/microsoft/VibeVoice/issues/185

**Data prep + long-form scaffolding:**
- https://github.com/m-bain/whisperX/discussions/1364 ; https://github.com/BoredYama/whisperX-AMD-ROCM7.1/tree/main ; https://gist.github.com/danielrosehill/f7ae5e659a02e32d056eb7887203b7a4
- https://huggingface.co/pyannote/speaker-diarization-3.1 ; https://rocm.blogs.amd.com/artificial-intelligence/speech_models/README.html
- https://github.com/zeropointnine/tts-audiobook-tool ; https://github.com/petermg/Chatterbox-TTS-Extended ; https://github.com/DrewThomasson/ebook2audiobook
- https://arxiv.org/pdf/2509.19852 (stability hallucinations; select by listening)