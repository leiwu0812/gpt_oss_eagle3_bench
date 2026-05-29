"""Simple async throughput / latency probe for the trtllm-serve OpenAI endpoint."""
import argparse, asyncio, json, time, statistics, aiohttp

PROMPT = ("You are a helpful assistant. Summarize the history of speculative "
          "decoding in large language models in about 200 words.")


async def one(session, url, max_tokens):
    payload = {
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(url, json=payload) as r:
        data = await r.json()
    dt = time.perf_counter() - t0
    out = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return dt, usage.get("completion_tokens", len(out.split()))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    sem = asyncio.Semaphore(a.concurrency)
    results = []
    async with aiohttp.ClientSession() as s:
        async def worker():
            async with sem:
                results.append(await one(s, a.url, a.max_tokens))
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(a.requests)])
        wall = time.perf_counter() - t0

    lat = [r[0] for r in results]
    toks = [r[1] for r in results]
    total_tok = sum(toks)
    print(json.dumps({
        "requests": a.requests,
        "concurrency": a.concurrency,
        "wall_s": round(wall, 2),
        "throughput_req_s": round(a.requests / wall, 2),
        "throughput_tok_s": round(total_tok / wall, 2),
        "latency_p50_s": round(statistics.median(lat), 3),
        "latency_p95_s": round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 3),
        "avg_completion_tokens": round(total_tok / len(toks), 1),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
