#!/usr/bin/env python3
"""Claude count_tokens API 호출 스크립트."""

import os

import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def count_tokens(text: str, model: str = "claude-sonnet-4-20250514") -> int:
    """텍스트의 토큰 수를 반환한다."""
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens


if __name__ == "__main__":
    # 테스트
    tests = [
        "Hello, world!",
        "안녕하세요",
        "한국어 토크나이저 테스트입니다.",
        "a",
    ]

    for text in tests:
        tokens = count_tokens(text)
        print(f"{tokens:>4} tokens | {text!r}")
