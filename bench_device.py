"""CPU vs MPS benchmark for Kokoro TTS, comparing several MPS configurations.

Configs compared:
  cpu                  : baseline, plain CPU.
  mps_naive            : device='mps' passed to KPipeline, complex STFT (torch.stft).
  mps_no_complex       : device='mps' + KModel(disable_complex=True) -> conv1d STFT.
  mps_no_complex_fp16  : as above + model.half() for fp16 inference.

Run from the repo root:
    .venv/bin/python bench_device.py
"""
import os
import time
import warnings

# Silence the noisy resize/STFT deprecation warnings from torch internals;
# they don't affect correctness and drown the output.
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from kokoro import KPipeline
from kokoro.model import KModel

from audiblez.core import gen_audio_segments, load_spacy, set_espeak_library

# ~2.3k chars of public-domain prose (Poe, "The Cask of Amontillado", abridged).
TEXT = (
    "The thousand injuries of Fortunato I had borne as I best could, but when he ventured upon insult, "
    "I vowed revenge. You, who so well know the nature of my soul, will not suppose, however, that I "
    "gave utterance to a threat. At length I would be avenged; this was a point definitively settled "
    "— but the very definitiveness with which it was resolved, precluded the idea of risk. I must not "
    "only punish, but punish with impunity. A wrong is unredressed when retribution overtakes its "
    "redresser. It is equally unredressed when the avenger fails to make himself felt as such to him "
    "who has done the wrong.\n\n"
    "It must be understood that neither by word nor deed had I given Fortunato cause to doubt my "
    "good will. I continued, as was my wont, to smile in his face, and he did not perceive that my "
    "smile now was at the thought of his immolation. He had a weak point — this Fortunato — although "
    "in other regards he was a man to be respected and even feared. He prided himself on his "
    "connoisseurship in wine. Few Italians have the true virtuoso spirit. For the most part their "
    "enthusiasm is adopted to suit the time and opportunity — to practise imposture upon the British "
    "and Austrian millionaires. In painting and gemmary, Fortunato, like his countrymen, was a "
    "quack — but in the matter of old wines he was sincere. In this respect I did not differ from "
    "him materially: I was skilful in the Italian vintages myself, and bought largely whenever I "
    "could.\n\n"
    "It was about dusk, one evening during the supreme madness of the carnival season, that I "
    "encountered my friend. He accosted me with excessive warmth, for he had been drinking much. "
    "The man wore motley. He had on a tight-fitting parti-striped dress, and his head was surmounted "
    "by the conical cap and bells. I was so pleased to see him, that I thought I should never have "
    "done wringing his hand. I said to him — \"My dear Fortunato, you are luckily met. How "
    "remarkably well you are looking to-day! But I have received a pipe of what passes for Amontillado, "
    "and I have my doubts.\""
)
VOICE = "af_sky"
SAMPLE_RATE = 24000


def build_pipeline(device: str, disable_complex: bool, fp16: bool) -> KPipeline:
    """Build a KPipeline on `device`, optionally with the conv-based STFT and/or fp16."""
    repo_id = "hexgrad/Kokoro-82M"
    if disable_complex:
        # Build the model ourselves so we can pass disable_complex.
        model = KModel(repo_id=repo_id, disable_complex=True).to(device).eval()
        if fp16:
            model = model.half()
        pipeline = KPipeline(lang_code=VOICE[0], repo_id=repo_id, model=model)
    else:
        pipeline = KPipeline(lang_code=VOICE[0], repo_id=repo_id, device=device)
        if fp16:
            pipeline.model = pipeline.model.half()
    return pipeline


def benchmark(label: str, device: str, disable_complex: bool = False, fp16: bool = False) -> dict:
    print(f"\n=== {label}  (device={device}, disable_complex={disable_complex}, fp16={fp16}) ===")
    pipeline = build_pipeline(device, disable_complex, fp16)

    t_warm = time.time()
    list(pipeline("Hello world, this is a warmup sentence.", voice=VOICE, speed=1.0))
    warm_dt = time.time() - t_warm

    t0 = time.time()
    segments = gen_audio_segments(pipeline, TEXT, voice=VOICE, speed=1.0)
    dt = time.time() - t0

    total_samples = sum(len(s) for s in segments) if segments else 0
    audio_seconds = total_samples / SAMPLE_RATE
    cps = len(TEXT) / dt if dt > 0 else 0
    rtf = dt / audio_seconds if audio_seconds > 0 else float("inf")
    print(f"  warmup: {warm_dt:.2f}s")
    print(f"  measured: {len(TEXT)} chars in {dt:.2f}s -> {cps:.0f} chars/sec")
    print(f"  audio produced: {audio_seconds:.2f}s (RTF {rtf:.3f}x)")
    return dict(label=label, seconds=dt, chars_per_sec=cps, rtf=rtf, warmup=warm_dt)


def main():
    print(f"torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}")
    # Required by both backends but only matters for MPS path.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    load_spacy()
    set_espeak_library()

    results = []
    configs = [
        ("cpu",                 dict(device="cpu", disable_complex=False, fp16=False)),
        ("mps_naive",           dict(device="mps", disable_complex=False, fp16=False)),
        ("mps_no_complex",      dict(device="mps", disable_complex=True,  fp16=False)),
        # fp16 on MPS crashes with an MPSNDArrayMatrixMultiplication accumulator
        # assertion in torch 2.12 — skip it here and revisit selectively (e.g.
        # bf16, or converting only specific submodules).
        # ("mps_no_complex_fp16", dict(device="mps", disable_complex=True,  fp16=True)),
    ]
    for label, kwargs in configs:
        try:
            results.append(benchmark(label, **kwargs))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results.append(dict(label=label, seconds=float("inf"), chars_per_sec=0, rtf=float("inf"), warmup=0, error=str(e)))

    print("\n=== Summary ===")
    print(f"  {'config':<22}  {'chars/s':>8}  {'RTF':>7}  {'wall':>7}  {'warmup':>7}")
    for r in results:
        print(f"  {r['label']:<22}  {r['chars_per_sec']:>8.0f}  {r['rtf']:>7.3f}  {r['seconds']:>6.2f}s  {r['warmup']:>6.2f}s")

    cpu_wall = next((r["seconds"] for r in results if r["label"] == "cpu"), None)
    if cpu_wall and cpu_wall > 0:
        print("\nSpeedup vs CPU (>1 means MPS wins):")
        for r in results:
            if r["label"] == "cpu" or r["seconds"] == float("inf"):
                continue
            print(f"  {r['label']:<22}  {cpu_wall / r['seconds']:.2f}x")


if __name__ == "__main__":
    main()
