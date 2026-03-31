#!/usr/bin/env python3
"""수집된 한국어 후보 토큰을 Claude count_tokens API로 검증한다.

Bedrock 키 N개를 각각 별도 프로세스로 분리 + Anthropic 프로세스 1개.
multiprocessing으로 env var 경쟁 문제를 완전히 제거.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export BEDROCK_TOKENS="key1,key2,key3,key4"
    python verify_korean.py
"""

import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BEDROCK_MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

WORKERS_PER_KEY = 5
ANTHROPIC_WORKERS = 5
MAX_RETRIES = 4
BATCH_SIZE = 500
SANDWICH_MARKER = "§"

CANDIDATES_FILE = "morpheme_candidates.json"
CHECKPOINT_FILE = "morpheme_checkpoint.json"
RESULTS_FILE = "morpheme_verified.json"


# ---------------------------------------------------------------------------
# Worker process: Bedrock
# ---------------------------------------------------------------------------

def bedrock_worker(token: str, candidates: list[str], result_queue: mp.Queue):
    """별도 프로세스에서 Bedrock 키 하나로 검증."""
    import boto3

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    def raw(text):
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.count_tokens(
                    modelId=BEDROCK_MODEL,
                    input={"converse": {"messages": [
                        {"role": "user", "content": [{"text": text}]}
                    ]}},
                )
                return resp["inputTokens"]
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1 + attempt)
                else:
                    raise

    baseline = raw(SANDWICH_MARKER + SANDWICH_MARKER)

    def check(candidate):
        try:
            n = raw(SANDWICH_MARKER + candidate + SANDWICH_MARKER)
            return candidate, (n - baseline) == 1
        except Exception:
            return candidate, None

    with ThreadPoolExecutor(max_workers=WORKERS_PER_KEY) as pool:
        futures = {pool.submit(check, c): c for c in candidates}
        for future in as_completed(futures):
            result_queue.put(future.result())


# ---------------------------------------------------------------------------
# Worker process: Anthropic
# ---------------------------------------------------------------------------

def anthropic_worker(api_key: str, candidates: list[str], result_queue: mp.Queue):
    """별도 프로세스에서 Anthropic API로 검증."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def raw(text):
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.messages.count_tokens(
                    model=ANTHROPIC_MODEL,
                    messages=[{"role": "user", "content": text}],
                )
                return resp.input_tokens
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1 + attempt)
                else:
                    raise

    baseline = raw(SANDWICH_MARKER + SANDWICH_MARKER)

    def check(candidate):
        try:
            n = raw(SANDWICH_MARKER + candidate + SANDWICH_MARKER)
            return candidate, (n - baseline) == 1
        except Exception:
            return candidate, None

    with ThreadPoolExecutor(max_workers=ANTHROPIC_WORKERS) as pool:
        futures = {pool.submit(check, c): c for c in candidates}
        for future in as_completed(futures):
            result_queue.put(future.result())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_candidates():
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidates = []
    for c in data["korean_candidates"]:
        if not c or not c.strip():
            continue
        if any(ord(ch) < 0x20 and ch not in ('\t', '\n', '\r') for ch in c):
            continue
        candidates.append(c)
    return candidates


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("verified", [])), set(data.get("checked", []))


def save_checkpoint(verified, checked):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "verified": sorted(verified),
            "checked": sorted(checked),
        }, f, ensure_ascii=False)


def save_results(verified, checked):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "verified": sorted(verified),
            "checked": sorted(checked),
            "stats": {
                "total_verified": len(verified),
                "total_checked": len(checked),
                "hit_rate": len(verified) / max(len(checked), 1) * 100,
            },
        }, f, ensure_ascii=False, indent=2)


def main():
    bedrock_tokens = [t.strip() for t in os.environ.get("BEDROCK_TOKENS", "").split(",") if t.strip()]
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    n_bedrock = len(bedrock_tokens)
    has_anthropic = bool(anthropic_key)
    total_channels = n_bedrock + (1 if has_anthropic else 0)

    print(f"\n=== 한국어 토큰 검증 ===")
    print(f"  Bedrock 키 {n_bedrock}개 × {WORKERS_PER_KEY}w = {n_bedrock * WORKERS_PER_KEY}w")
    if has_anthropic:
        print(f"  Anthropic {ANTHROPIC_WORKERS}w")
    print(f"  총 {n_bedrock * WORKERS_PER_KEY + (ANTHROPIC_WORKERS if has_anthropic else 0)} workers")

    all_candidates = load_candidates()
    print(f"\n  전체 후보: {len(all_candidates):,}개")

    verified, checked = load_checkpoint()
    if checked:
        print(f"  체크포인트 복원: {len(checked):,}개 검증됨, {len(verified):,}개 히트")

    remaining = [c for c in all_candidates if c not in checked]
    print(f"  미검증 후보: {len(remaining):,}개")

    if not remaining:
        print("\n  모든 후보가 이미 검증됨!")
        save_results(verified, checked)
        return

    # 빈도 기반 정렬
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    sources = data.get("korean_with_sources", {})
    remaining.sort(key=lambda t: len(sources.get(t, [])), reverse=True)

    # 채널별 균등 분배
    chunks = [[] for _ in range(total_channels)]
    for i, c in enumerate(remaining):
        chunks[i % total_channels].append(c)

    for i, chunk in enumerate(chunks):
        label = f"Bedrock#{i+1}" if i < n_bedrock else "Anthropic"
        print(f"  {label}: {len(chunk):,}개")

    # 결과 큐
    result_queue = mp.Queue()

    # 프로세스 시작
    processes = []
    for i in range(n_bedrock):
        p = mp.Process(target=bedrock_worker, args=(bedrock_tokens[i], chunks[i], result_queue))
        p.start()
        processes.append(p)
        print(f"  Bedrock#{i+1} 프로세스 시작 (PID {p.pid})")

    if has_anthropic:
        p = mp.Process(target=anthropic_worker, args=(anthropic_key, chunks[n_bedrock], result_queue))
        p.start()
        processes.append(p)
        print(f"  Anthropic 프로세스 시작 (PID {p.pid})")

    print(f"\n  검증 시작...\n")

    # 결과 수집
    total = len(remaining)
    done = 0
    hits = 0
    errors = 0
    batch_count = 0
    start_time = time.time()

    alive = True
    while alive:
        # 큐에서 결과 꺼내기
        try:
            candidate, result = result_queue.get(timeout=1)
        except Exception:
            # 모든 프로세스가 끝났는지 확인
            alive = any(p.is_alive() for p in processes)
            continue

        checked.add(candidate)
        if result is True:
            verified.add(candidate)
            hits += 1
        elif result is None:
            errors += 1

        done += 1
        batch_count += 1
        if batch_count >= BATCH_SIZE:
            save_checkpoint(verified, checked)
            batch_count = 0

        if done % 200 == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            checked_ok = done - errors
            hit_rate = hits / max(checked_ok, 1) * 100
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"\r  [{done:>7,}/{total:,}] "
                f"hits={hits:,} ({hit_rate:.1f}%) "
                f"err={errors} "
                f"rate={rate:.0f}/s "
                f"ETA={eta/60:.1f}min",
                end="", flush=True,
            )

    # 큐에 남은 결과 비우기
    while not result_queue.empty():
        try:
            candidate, result = result_queue.get_nowait()
            checked.add(candidate)
            if result is True:
                verified.add(candidate)
                hits += 1
            done += 1
        except Exception:
            break

    for p in processes:
        p.join(timeout=5)

    print("\n")
    save_checkpoint(verified, checked)
    save_results(verified, checked)

    print(f"\n=== 검증 완료 ===")
    print(f"  검증 완료: {len(checked):,}개")
    print(f"  히트 (단일 토큰): {len(verified):,}개")
    print(f"  히트율: {len(verified)/max(len(checked),1)*100:.1f}%")
    print(f"  결과: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
