#!/usr/bin/env python3
"""
Compare ctoc tokenizer vs Anthropic count_tokens API on Korean NSMC data.
"""

import csv
import os
import random
import subprocess
import tempfile
import time
import re

import anthropic

CTOC_BIN = os.path.join(os.path.dirname(__file__), "ctoc")
TRAIN_CSV = "/home/user/12312313123123/nsmc_train.csv"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def load_samples(path: str, n: int, seed: int) -> list[str]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["review"].strip()
            if text:
                rows.append(text)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def ctoc_count(text: str) -> int:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     encoding="utf-8", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        result = subprocess.run(
            [CTOC_BIN, "--by-file", tmp],
            capture_output=True, text=True, check=True
        )
        # Parse token count from output line: "<path>  .txt  <N>"
        for line in result.stdout.splitlines():
            m = re.search(r"\s+(\d+)\s*$", line)
            if m and tmp in line:
                return int(m.group(1))
    finally:
        os.unlink(tmp)
    raise ValueError(f"Could not parse ctoc output:\n{result.stdout}")


def anthropic_count(client: anthropic.Anthropic, text: str) -> int:
    resp = client.messages.count_tokens(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading {SAMPLE_SIZE} samples from NSMC train set...")
    samples = load_samples(TRAIN_CSV, SAMPLE_SIZE, RANDOM_SEED)

    results = []
    errors = 0

    for i, text in enumerate(samples, 1):
        try:
            c_count = ctoc_count(text)
            a_count = anthropic_count(client, text)
            diff = c_count - a_count
            pct_err = abs(diff) / a_count * 100 if a_count else 0
            results.append((text, c_count, a_count, diff, pct_err))

            status = "OK" if pct_err < 5 else ("WARN" if pct_err < 15 else "ERR")
            print(f"[{i:3d}/{SAMPLE_SIZE}] ctoc={c_count:4d} api={a_count:4d} "
                  f"diff={diff:+4d} ({pct_err:5.1f}%)  [{status}]  {text[:40]!r}")

            # Respect rate limits
            time.sleep(0.3)

        except Exception as e:
            print(f"[{i:3d}/{SAMPLE_SIZE}] ERROR: {e}  text={text[:40]!r}")
            errors += 1

    if not results:
        print("No results collected.")
        return

    pct_errors = [r[4] for r in results]
    exact = sum(1 for r in results if r[3] == 0)
    within5 = sum(1 for r in results if r[4] <= 5)
    within10 = sum(1 for r in results if r[4] <= 10)
    mean_err = sum(pct_errors) / len(pct_errors)
    over_count = sum(1 for r in results if r[3] > 0)
    under_count = sum(1 for r in results if r[3] < 0)

    print()
    print("=" * 60)
    print("  ctoc Korean Accuracy Report (vs Anthropic count_tokens)")
    print("=" * 60)
    print(f"  Samples tested   : {len(results)}")
    print(f"  Errors skipped   : {errors}")
    print(f"  Exact match      : {exact:4d} ({exact/len(results)*100:.1f}%)")
    print(f"  Within  5% error : {within5:4d} ({within5/len(results)*100:.1f}%)")
    print(f"  Within 10% error : {within10:4d} ({within10/len(results)*100:.1f}%)")
    print(f"  Mean abs % error : {mean_err:.2f}%")
    print(f"  Over-count       : {over_count:4d}")
    print(f"  Under-count      : {under_count:4d}")
    print("=" * 60)

    # Worst cases
    worst = sorted(results, key=lambda r: r[4], reverse=True)[:5]
    print("\n  Top 5 worst cases:")
    for text, c, a, d, p in worst:
        print(f"    ctoc={c} api={a} diff={d:+d} ({p:.1f}%)  {text[:50]!r}")


if __name__ == "__main__":
    main()
