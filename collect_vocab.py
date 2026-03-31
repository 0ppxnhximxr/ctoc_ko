#!/usr/bin/env python3
"""다양한 오픈웨이트 모델의 토크나이저에서 한국어 포함 토큰을 수집한다.

Usage:
    pip install transformers sentencepiece protobuf tiktoken
    python collect_vocab.py
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# 한글 범위 판별
# ---------------------------------------------------------------------------

def contains_korean(s: str) -> bool:
    """문자열에 한글(완성형, 자모, 호환자모)이 포함되어 있으면 True."""
    for ch in s:
        cp = ord(ch)
        if (0xAC00 <= cp <= 0xD7A3      # 완성형 음절
            or 0x1100 <= cp <= 0x11FF    # 자모
            or 0x3130 <= cp <= 0x318F    # 호환자모 (ㄱ-ㅎ, ㅏ-ㅣ)
            or 0xA960 <= cp <= 0xA97F    # 자모 확장-A
            or 0xD7B0 <= cp <= 0xD7FF   # 자모 확장-B
            or 0xFFA0 <= cp <= 0xFFDC): # 반각 자모
            return True
    return False


def is_cjk_or_unicode(s: str) -> bool:
    """문자열에 비ASCII 유니코드가 포함되어 있으면 True."""
    return any(ord(ch) > 127 for ch in s)


def categorize_token(s: str) -> str:
    """토큰의 카테고리를 분류한다."""
    if contains_korean(s):
        return "korean"
    elif is_cjk_or_unicode(s):
        return "unicode_other"
    else:
        return "ascii"


# ---------------------------------------------------------------------------
# 토크나이저 로딩 + vocab 추출
# ---------------------------------------------------------------------------

def extract_from_transformers(model_id: str, trust_remote_code: bool = False) -> set[str]:
    """HuggingFace transformers AutoTokenizer에서 vocab을 추출한다."""
    from transformers import AutoTokenizer

    print(f"  Loading {model_id}...", end=" ", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return set()

    vocab = set()
    vocab_dict = tok.get_vocab()
    for token_str, token_id in vocab_dict.items():
        try:
            # decode를 통해 실제 문자열 복원
            decoded = tok.decode([token_id])
            if decoded and decoded != "\ufffd":
                vocab.add(decoded)
        except Exception:
            pass

        # 원본 토큰 문자열도 추가 (공백 접두어 등 보존)
        cleaned = token_str.replace("▁", " ").replace("Ġ", " ")
        if cleaned and "\ufffd" not in cleaned:
            vocab.add(cleaned)

    print(f"OK ({len(vocab):,} tokens)")
    return vocab


def extract_from_tiktoken(encoding_name: str) -> set[str]:
    """tiktoken 인코딩에서 vocab을 추출한다."""
    import tiktoken

    print(f"  Loading tiktoken {encoding_name}...", end=" ", flush=True)
    enc = tiktoken.get_encoding(encoding_name)

    vocab = set()
    # tiktoken의 _mergeable_ranks에서 토큰 추출
    for token_bytes in enc._mergeable_ranks:
        try:
            s = token_bytes.decode("utf-8")
            if s and "\ufffd" not in s:
                vocab.add(s)
        except (UnicodeDecodeError, ValueError):
            pass

    # decode 방식으로도 추출
    max_id = enc.n_vocab
    for token_id in range(min(max_id, 300000)):
        try:
            decoded = enc.decode([token_id])
            if decoded and "\ufffd" not in decoded:
                vocab.add(decoded)
        except Exception:
            pass

    print(f"OK ({len(vocab):,} tokens)")
    return vocab


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

MODELS = {
    # --- 대형 다국어 ---
    "gemma3":       ("google/gemma-3-4b-it", False),
    "qwen2.5":      ("Qwen/Qwen2.5-0.5B", False),
    "llama4":       ("meta-llama/Llama-4-Scout-17B-16E-Instruct", False),
    "deepseek_v3":  ("deepseek-ai/DeepSeek-V3", False),
    "phi4_mini":    ("microsoft/Phi-4-mini-instruct", False),

    # --- 한국어 특화 ---
    "k_exaone":     ("LGAI-EXAONE/K-EXAONE-236B-A23B", True),
    "solar_open":   ("upstage/Solar-Open-100B", False),
    "exaone3":      ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", True),

    # --- 다국어 중형 ---
    "mistral_nemo": ("mistralai/Mistral-Nemo-Instruct-2407", False),
    "yi":           ("01-ai/Yi-1.5-6B", False),

    # --- 최신 모델 ---
    "glm5":         ("zai-org/GLM-5", True),
    "kimi_k2.5":    ("moonshotai/Kimi-K2.5", True),
    "minimax_m2.5": ("MiniMaxAI/MiniMax-M2.5", True),

    # --- 기존 ctoc에서 사용한 CJK 모델 ---
    "glm4":         ("THUDM/glm-4-9b-chat", True),
    "baichuan2":    ("baichuan-inc/Baichuan2-7B-Base", True),
    "bloom":        ("bigscience/bloom", False),
}

TIKTOKEN_ENCODINGS = {
    "cl100k_base": "cl100k_base",   # GPT-4
    "o200k_base":  "o200k_base",    # GPT-4o
}


def main():
    all_korean: dict[str, set[str]] = {}     # token -> set of sources
    all_unicode: dict[str, set[str]] = {}    # non-Korean unicode tokens
    stats = {}

    # --- tiktoken ---
    print("\n=== tiktoken ===")
    for name, enc_name in TIKTOKEN_ENCODINGS.items():
        try:
            vocab = extract_from_tiktoken(enc_name)
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            continue

        korean_count = 0
        unicode_count = 0
        for token in vocab:
            if contains_korean(token):
                all_korean.setdefault(token, set()).add(name)
                korean_count += 1
            elif is_cjk_or_unicode(token):
                all_unicode.setdefault(token, set()).add(name)
                unicode_count += 1

        stats[name] = {
            "total": len(vocab),
            "korean": korean_count,
            "unicode_other": unicode_count,
        }

    # --- HuggingFace models ---
    print("\n=== HuggingFace Models ===")
    for name, (model_id, trust_remote) in MODELS.items():
        try:
            vocab = extract_from_transformers(model_id, trust_remote_code=trust_remote)
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            continue

        if not vocab:
            continue

        korean_count = 0
        unicode_count = 0
        for token in vocab:
            if contains_korean(token):
                all_korean.setdefault(token, set()).add(name)
                korean_count += 1
            elif is_cjk_or_unicode(token):
                all_unicode.setdefault(token, set()).add(name)
                unicode_count += 1

        stats[name] = {
            "total": len(vocab),
            "korean": korean_count,
            "unicode_other": unicode_count,
        }

    # --- 통계 출력 ---
    print(f"\n{'='*70}")
    print(f" 토크나이저별 통계")
    print(f"{'='*70}")
    print(f"{'모델':<20} {'전체':>8} {'한국어':>8} {'기타유니코드':>10}")
    print(f"{'-'*70}")
    for name, s in sorted(stats.items(), key=lambda x: x[1]["korean"], reverse=True):
        print(f"{name:<20} {s['total']:>8,} {s['korean']:>8,} {s['unicode_other']:>10,}")

    # --- 한국어 토큰 빈도 분석 ---
    print(f"\n{'='*70}")
    print(f" 한국어 토큰 수집 결과")
    print(f"{'='*70}")
    print(f"  고유 한국어 토큰 수: {len(all_korean):,}")
    print(f"  고유 기타 유니코드 토큰 수: {len(all_unicode):,}")

    # 여러 모델에서 공통으로 등장하는 토큰
    freq = Counter({t: len(srcs) for t, srcs in all_korean.items()})
    print(f"\n  출현 빈도별 한국어 토큰 분포:")
    for n in range(1, max(freq.values()) + 1):
        count = sum(1 for f in freq.values() if f == n)
        if count > 0:
            print(f"    {n}개 모델에서 등장: {count:,}개 토큰")

    # 상위 샘플
    print(f"\n  가장 많은 모델에서 등장하는 한국어 토큰 TOP 30:")
    for token, count in freq.most_common(30):
        sources = ", ".join(sorted(all_korean[token]))
        display = repr(token)
        if len(display) > 30:
            display = display[:30] + "..."
        print(f"    {display:<35} {count}개 모델 | {sources}")

    # --- 결과 저장 ---
    output = {
        "korean_candidates": sorted(all_korean.keys()),
        "unicode_candidates": sorted(all_unicode.keys()),
        "korean_with_sources": {
            token: sorted(sources)
            for token, sources in sorted(all_korean.items())
        },
        "stats": stats,
    }

    out_path = "collected_vocab.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 {out_path}에 저장되었습니다.")
    print(f"  한국어 후보: {len(output['korean_candidates']):,}개")
    print(f"  기타 유니코드 후보: {len(output['unicode_candidates']):,}개")


if __name__ == "__main__":
    main()
