"""Per-category MT-Bench acceptance-rate probe for trtllm-serve + Eagle3.

Strategy:
  - Load the 80 MT-Bench prompts (lmsys FastChat, each tagged with a category).
  - For each prompt, in strict serial order (concurrency=1):
      1. Drain /metrics (the iter-stats queue is consumed on read).
      2. Send the chat completion request and wait for it to finish.
      3. Poll /metrics again, sum accepted_draft_tokens / total_draft_tokens
         across the iterations that happened in between, and attribute the
         delta to this prompt's category.
  - Aggregate per category and print a table.

Requires the server to be started with `enable_iter_perf_stats: true` in
extra_llm_api_options (eagle.yaml in this repo already sets it).

NOTE: The PyTorch backend's /metrics endpoint is documented as beta; field
names can shift between TRT-LLM releases. This script accepts a few common
spellings (accepted_draft_tokens / acceptedDraftTokens / total_accepted_draft_tokens
and the same for total/proposed). If your build emits different keys, run with
--dump-metrics to inspect a raw sample.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
import urllib.request

import requests

MT_BENCH_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/question.jsonl"
)

ACCEPTED_KEYS = (
    "accepted_draft_tokens",
    "acceptedDraftTokens",
    "total_accepted_draft_tokens",
    "totalAcceptedDraftTokens",
)
TOTAL_KEYS = (
    "total_draft_tokens",
    "totalDraftTokens",
    "num_draft_tokens",
    "numDraftTokens",
)


def load_mtbench():
    with urllib.request.urlopen(MT_BENCH_URL, timeout=30) as r:
        text = r.read().decode("utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        # turn 1 only; FastChat stores prompts in q["turns"]
        out.append({"category": q["category"], "prompt": q["turns"][0]})
    return out


def fetch_metrics(base_url: str) -> list[dict]:
    """Return the list of iter-stat dicts currently queued, draining the queue."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"[warn] /metrics fetch failed: {e}", file=sys.stderr)
        return []
    try:
        data = r.json()
    except ValueError:
        # Some builds return text/plain Prometheus exposition; not what we need.
        print("[warn] /metrics returned non-JSON; this build likely doesn't "
              "expose iter stats as JSON. Consider --dump-metrics.", file=sys.stderr)
        return []
    if isinstance(data, dict):
        # Sometimes wrapped as {"iter_stats": [...]} or similar
        for k in ("iter_stats", "iterStats", "stats", "data"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return [data]
    if isinstance(data, list):
        return data
    return []


def extract_counts(iter_stats: list[dict]) -> tuple[int, int]:
    """Sum (accepted, total) draft tokens across iter stats, tolerating nesting."""
    accepted = 0
    total = 0

    def walk(obj):
        nonlocal accepted, total
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ACCEPTED_KEYS and isinstance(v, (int, float)):
                    accepted += int(v)
                elif k in TOTAL_KEYS and isinstance(v, (int, float)):
                    total += int(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(iter_stats)
    return accepted, total


def send_chat(base_url: str, model: str, prompt: str, max_tokens: int) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(f"{base_url}/v1/chat/completions",
                      json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return int(data.get("usage", {}).get("completion_tokens", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="optional: cap number of MT-Bench prompts (debug)")
    ap.add_argument("--dump-metrics", action="store_true",
                    help="print one raw /metrics payload and exit")
    ap.add_argument("--output", default="mtbench_acceptance.json")
    args = ap.parse_args()

    if args.dump_metrics:
        print(json.dumps(fetch_metrics(args.base_url), indent=2))
        return

    questions = load_mtbench()
    if args.limit:
        questions = questions[: args.limit]
    print(f"[info] loaded {len(questions)} MT-Bench prompts")

    by_cat = collections.defaultdict(lambda: {"accepted": 0, "total": 0,
                                              "n": 0, "completion_tokens": 0,
                                              "per_step_samples": []})

    for i, q in enumerate(questions, 1):
        fetch_metrics(args.base_url)  # drain
        t0 = time.perf_counter()
        ctoks = send_chat(args.base_url, args.model, q["prompt"], args.max_tokens)
        dt = time.perf_counter() - t0
        stats = fetch_metrics(args.base_url)
        acc, tot = extract_counts(stats)
        n_iters = len(stats) if isinstance(stats, list) else 0

        cat = q["category"]
        bucket = by_cat[cat]
        bucket["accepted"] += acc
        bucket["total"] += tot
        bucket["n"] += 1
        bucket["completion_tokens"] += ctoks
        if n_iters > 0 and ctoks > 0:
            # tokens-per-step proxy: completion_tokens / decode iterations
            bucket["per_step_samples"].append(ctoks / n_iters)

        rate = (acc / tot) if tot else float("nan")
        print(f"[{i:>2}/{len(questions)}] {cat:<12s} "
              f"completion_tok={ctoks:<5d} dt={dt:5.2f}s "
              f"accepted={acc:<6d} total={tot:<6d} rate={rate:.3f}")

    print("\n=== per-category MT-Bench acceptance ===")
    print(f"{'category':<14s} {'n':>3s} {'rate':>8s} {'tok/step':>10s} "
          f"{'avg_completion':>15s}")
    summary = {}
    for cat, b in sorted(by_cat.items()):
        rate = (b["accepted"] / b["total"]) if b["total"] else float("nan")
        tps = (statistics.mean(b["per_step_samples"])
               if b["per_step_samples"] else float("nan"))
        avg_c = b["completion_tokens"] / b["n"] if b["n"] else 0.0
        print(f"{cat:<14s} {b['n']:>3d} {rate:>8.3f} {tps:>10.3f} {avg_c:>15.1f}")
        summary[cat] = {"n": b["n"], "acceptance_rate": rate,
                        "tokens_per_step": tps,
                        "avg_completion_tokens": avg_c,
                        "accepted_draft_tokens": b["accepted"],
                        "total_draft_tokens": b["total"]}

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[info] wrote {args.output}")


if __name__ == "__main__":
    main()
