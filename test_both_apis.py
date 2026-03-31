#!/usr/bin/env python3
"""Anthropic 직접 API와 AWS Bedrock 두 채널로 count_tokens 비교 테스트."""

import json
import time

import anthropic
import boto3

import os

anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", os.environ.get("BEDROCK_TOKEN", ""))
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
BEDROCK_MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"
BASELINE = 7  # framing tokens


def count_anthropic(text: str) -> int:
    resp = anthropic_client.messages.count_tokens(
        model=ANTHROPIC_MODEL,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens - BASELINE


def count_bedrock(text: str) -> int:
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
    return resp["inputTokens"] - BASELINE


def main():
    with open("test_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    results = []
    total_anthropic = 0
    total_bedrock = 0
    total_chars = 0
    total_count = 0

    for category, texts in corpus.items():
        print(f"\n{'='*70}")
        print(f" {category}")
        print(f"{'='*70}")
        print(f"{'텍스트':<45} {'글자':>4} {'Anth':>5} {'Bedr':>5} {'일치':>4} {'글자/토큰':>8}")
        print(f"{'-'*70}")

        cat_anthropic = 0
        cat_bedrock = 0
        cat_chars = 0

        for text in texts:
            display = text.replace("\n", "\\n")
            if len(display) > 42:
                display = display[:42] + "..."

            chars = len(text)

            try:
                a_count = count_anthropic(text)
            except Exception as e:
                a_count = -1

            try:
                b_count = count_bedrock(text)
            except Exception as e:
                b_count = -1

            match = "O" if a_count == b_count else "X"
            ratio = f"{chars/a_count:.2f}" if a_count > 0 else "N/A"

            print(f"{display:<45} {chars:>4} {a_count:>5} {b_count:>5} {match:>4} {ratio:>8}")

            results.append({
                "category": category,
                "text": text,
                "chars": chars,
                "anthropic_tokens": a_count,
                "bedrock_tokens": b_count,
                "match": a_count == b_count,
            })

            if a_count > 0:
                cat_anthropic += a_count
                cat_chars += chars
            if b_count > 0:
                cat_bedrock += b_count

        if cat_anthropic > 0:
            print(f"{'-'*70}")
            print(f"{'소계':<45} {cat_chars:>4} {cat_anthropic:>5} {cat_bedrock:>5} {'':>4} {cat_chars/cat_anthropic:>8.2f}")
            total_anthropic += cat_anthropic
            total_bedrock += cat_bedrock
            total_chars += cat_chars

    # 전체 요약
    print(f"\n{'='*70}")
    print(f" 전체 요약")
    print(f"{'='*70}")
    total_texts = len(results)
    matches = sum(1 for r in results if r["match"])
    print(f"  총 텍스트 수:     {total_texts}")
    print(f"  API 일치:         {matches}/{total_texts}")
    print(f"  총 글자 수:       {total_chars:,}")
    print(f"  Anthropic 토큰:   {total_anthropic:,}")
    print(f"  Bedrock 토큰:     {total_bedrock:,}")
    if total_anthropic > 0:
        print(f"  평균 글자/토큰:   {total_chars/total_anthropic:.2f}")

    # 카테고리별 요약
    print(f"\n{'카테고리':<25} {'글자':>6} {'토큰':>6} {'글자/토큰':>8}")
    print(f"{'-'*48}")
    for category in corpus:
        cat_results = [r for r in results if r["category"] == category and r["anthropic_tokens"] > 0]
        c = sum(r["chars"] for r in cat_results)
        t = sum(r["anthropic_tokens"] for r in cat_results)
        if t > 0:
            print(f"{category:<25} {c:>6} {t:>6} {c/t:>8.2f}")

    # 결과 저장
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 test_results.json에 저장되었습니다.")


if __name__ == "__main__":
    main()
