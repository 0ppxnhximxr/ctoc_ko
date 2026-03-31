#!/usr/bin/env python3
"""수집된 한국어 후보 토큰을 Claude count_tokens API로 검증한다.

Sandwich counting 기법으로 각 후보가 Claude에서 단일 토큰인지 확인.
Bedrock(70%)과 Anthropic(30%)을 병렬로 분담 처리.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export BEDROCK_TOKEN=ABSK...
    python verify_korean.py
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BEDROCK_TOKEN = os.environ.get("BEDROCK_TOKEN", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

BEDROCK_MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

BEDROCK_WORKERS = 40
ANTHROPIC_WORKERS = 10
MAX_RETRIES = 4
BATCH_SIZE = 500
SANDWICH_MARKER = "§"

CANDIDATES_FILE = "collected_vocab.json"
CHECKPOINT_FILE = "verify_checkpoint.json"
RESULTS_FILE = "verified_korean.json"

# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_TOKEN
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

anthropic_client = None
if ANTHROPIC_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ---------------------------------------------------------------------------
# Sandwich counting with retry
# ---------------------------------------------------------------------------

def _bedrock_raw(text: str) -> int:
    for attempt in range(MAX_RETRIES):
        try:
            resp = bedrock_client.count_tokens(
                modelId=BEDROCK_MODEL,
                input={
                    "converse": {
                        "messages": [
                            {"role": "user", "content": [{"text": text}]}
                        ]
                    }
                },
            )
            return resp["inputTokens"]
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def _anthropic_raw(text: str) -> int:
    for attempt in range(MAX_RETRIES):
        try:
            resp = anthropic_client.messages.count_tokens(
                model=ANTHROPIC_MODEL,
                messages=[{"role": "user", "content": text}],
            )
            return resp.input_tokens
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


_baseline_cache = {}
_baseline_lock = threading.Lock()


def get_baseline(raw_fn, name: str) -> int:
    with _baseline_lock:
        if name not in _baseline_cache:
            _baseline_cache[name] = raw_fn(SANDWICH_MARKER + SANDWICH_MARKER)
        return _baseline_cache[name]


def check_bedrock(candidate: str) -> tuple[str, bool | None]:
    """Bedrock으로 단일 토큰 여부 확인."""
    try:
        baseline = get_baseline(_bedrock_raw, "bedrock")
        sandwiched = _bedrock_raw(SANDWICH_MARKER + candidate + SANDWICH_MARKER)
        return candidate, (sandwiched - baseline) == 1
    except Exception:
        return candidate, None


def check_anthropic(candidate: str) -> tuple[str, bool | None]:
    """Anthropic으로 단일 토큰 여부 확인."""
    try:
        baseline = get_baseline(_anthropic_raw, "anthropic")
        sandwiched = _anthropic_raw(SANDWICH_MARKER + candidate + SANDWICH_MARKER)
        return candidate, (sandwiched - baseline) == 1
    except Exception:
        return candidate, None


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.hits = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, is_hit: bool | None):
        with self.lock:
            self.done += 1
            if is_hit is True:
                self.hits += 1
            elif is_hit is None:
                self.errors += 1

            if self.done % 100 == 0 or self.done == self.total:
                elapsed = time.time() - self.start_time
                rate = self.done / elapsed if elapsed > 0 else 0
                checked_ok = self.done - self.errors
                hit_rate = self.hits / max(checked_ok, 1) * 100
                eta = (self.total - self.done) / rate if rate > 0 else 0
                print(
                    f"\r  [{self.done:>7,}/{self.total:,}] "
                    f"hits={self.hits:,} ({hit_rate:.1f}%) "
                    f"err={self.errors} "
                    f"rate={rate:.0f}/s "
                    f"ETA={eta/60:.1f}min",
                    end="", flush=True,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_candidates() -> list[str]:
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


def load_checkpoint() -> tuple[set[str], set[str]]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("verified", [])), set(data.get("checked", []))


def save_checkpoint(verified: set[str], checked: set[str]):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "verified": sorted(verified),
            "checked": sorted(checked),
        }, f, ensure_ascii=False)


def save_results(verified: set[str], checked: set[str]):
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
    print("=== 한국어 토큰 검증 시작 ===\n")

    all_candidates = load_candidates()
    print(f"  전체 후보: {len(all_candidates):,}개")

    verified, checked = load_checkpoint()
    if checked:
        print(f"  체크포인트 복원: {len(checked):,}개 검증됨, {len(verified):,}개 히트")

    remaining = [c for c in all_candidates if c not in checked]
    print(f"  미검증 후보: {len(remaining):,}개")

    if not remaining:
        print("\n  모든 후보가 이미 검증됨!")
        save_results(verified, checked)
        return

    # 빈도 기반 정렬 (여러 모델에서 등장하는 토큰 우선)
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    sources = data.get("korean_with_sources", {})
    remaining.sort(key=lambda t: len(sources.get(t, [])), reverse=True)

    # 70/30 분할
    split_idx = int(len(remaining) * 0.7)
    bedrock_batch = remaining[:split_idx]
    anthropic_batch = remaining[split_idx:]

    # 베이스라인 워밍업
    print("\n  베이스라인 측정 중...", end=" ", flush=True)
    b_baseline = get_baseline(_bedrock_raw, "bedrock")
    print(f"Bedrock={b_baseline}", end=" ", flush=True)
    if anthropic_client:
        a_baseline = get_baseline(_anthropic_raw, "anthropic")
        print(f"Anthropic={a_baseline}")
    else:
        print("(Anthropic 없음, Bedrock만 사용)")
        bedrock_batch = remaining
        anthropic_batch = []

    print(f"\n  Bedrock {BEDROCK_WORKERS}w × {len(bedrock_batch):,}개"
          f" + Anthropic {ANTHROPIC_WORKERS}w × {len(anthropic_batch):,}개\n")

    progress = Progress(len(remaining))
    result_lock = threading.Lock()
    batch_count = 0

    def on_result(candidate: str, result: bool | None):
        nonlocal batch_count
        with result_lock:
            checked.add(candidate)
            if result is True:
                verified.add(candidate)
            batch_count += 1
            if batch_count % BATCH_SIZE == 0:
                save_checkpoint(verified, checked)
        progress.update(result)

    # 두 풀을 동시에 실행
    bedrock_pool = ThreadPoolExecutor(max_workers=BEDROCK_WORKERS, thread_name_prefix="bedrock")
    anthropic_pool = ThreadPoolExecutor(max_workers=ANTHROPIC_WORKERS, thread_name_prefix="anthropic")

    all_futures = {}
    for c in bedrock_batch:
        f = bedrock_pool.submit(check_bedrock, c)
        all_futures[f] = c
    for c in anthropic_batch:
        f = anthropic_pool.submit(check_anthropic, c)
        all_futures[f] = c

    for future in as_completed(all_futures):
        candidate, result = future.result()
        on_result(candidate, result)

    bedrock_pool.shutdown(wait=False)
    anthropic_pool.shutdown(wait=False)

    # 최종 저장
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
