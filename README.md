# Sevak-Hybrid: Small Qwen Specialists + Test-Time Compute Match Phi-4-mini

**Author:** Gurpreet Singh Dhillon (Nam-toon Studio) · October 2026

Sevak-Hybrid is a **system, not a newly trained model**: a router that sends each question to a
1.5–2B open model specialised for it, and spends extra compute only when the model is unsure.
No weights were trained or modified. On held-out questions, run on one Apple-Silicon laptop with
the same harness for every system, it **matches Microsoft's Phi-4-mini (3.8B)** on math,
knowledge and code.

## Results (held-out, same harness, same questions)

| Task | Sevak-Hybrid | Phi-4-mini 3.8B (8-bit) |
|---|---|---|
| Math — GSM8K, 200 q | **93.5%** — Qwen2.5-Math-1.5B, CoT, adaptive 4→8 votes (~4.5 samples/q) | 92.5% (8 votes) · 90.5% (greedy) |
| Knowledge — MMLU, 300 q | **71.3%** — pooled A–D logits of Qwen2.5-1.5B + Qwen3.5-2B ("System 1"); if confidence < 0.7 (59% of q), Qwen3-1.7B reasons ("System 2") | 70.7% (CoT) · 61.7% (logits) |
| Code — HumanEval tasks 64–163 | **59%** — Qwen2.5-Coder-1.5B greedy | 56% (greedy) |
| Code, 8 samples + docstring self-test* | **70%** | 66% |

**How to read this honestly**
- Every gap is inside sampling noise (±3.5–5 points). The fair word is **matches**, not beats.
- "1.5B" means *each specialist* is 1.5–2B; the router loads up to ~3 B of models in total.
- Phi-4-mini was run by us in the same harness. Its model card reports different numbers
  (e.g. GSM8K 88.6%, 8-shot, Microsoft's own harness) that are not comparable to ours.
- Rows 0–99 of each dev set were used to choose models and thresholds; held-out rows were run once.
- \*The docstring-example parser was widened after the first held-out code run, so that row is
  post-hoc. The greedy code row is clean. Harness check: Coder-1.5B over all 164 HumanEval tasks
  = 71.3% vs Qwen's reported 70.7%.
- A tool-calling JSON agent (the original design) scored only 35% on GSM8K vs 75% for plain
  chain-of-thought with the same 1.5B model — format mattered more than size.
- A two-model ensemble that looked better on the tuning split (92%) was **worse** held-out (90.5%).

## What's in the box
- `src/router.py` — task detection → specialist; every answer returns `{task, answer, confidence, system}`
- `src/sevak.py` — adaptive self-consistency with calibrated confidence (high-confidence answers were
  right 97% of the time on held-out GSM8K)
- `src/mlx_llm.py` — MLX backend with batched sampling and one-pass "System 1" choice probabilities
- `scripts/` — every ablation and held-out evaluation used above; `runs/` — raw per-question results
- `test_all.py` — component + regression tests

## Limits
- English benchmarks only. Not evaluated on Punjabi or Gurmat knowledge; small models are weak at
  niche facts (it answered a Sikh-history question wrongly, at low confidence — which it flagged).
- No training was done; LoRA self-training on verified traces is future work.

## Models used (all third-party, unmodified)
Qwen2.5-Math-1.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen3.5-2B, Qwen3-1.7B
(Qwen licences apply). Baseline: microsoft/Phi-4-mini-instruct (MIT).
