"""
评测指标
- Rouge-1 F1
- BLEU (fugashi 分词)
- 语义相似度 (bge-m3 cosine)
"""
from __future__ import annotations

import numpy as np
from rouge_score import rouge_scorer

import fugashi
from codes.embedding import encode_sync

_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
_tagger = fugashi.Tagger()


def _tokenize_ja(text: str) -> str:
    """fugashi 分词，返回空格分隔字符串（rouge/bleu 用）"""
    return " ".join(w.surface for w in _tagger(text) if w.surface.strip())


def rouge1_f1(hypothesis: str, reference: str) -> float:
    h = _tokenize_ja(hypothesis)
    r = _tokenize_ja(reference)
    score = _rouge.score(r, h)
    return round(score["rouge1"].fmeasure, 4)


def bleu(hypothesis: str, reference: str) -> float:
    from sacrebleu.metrics import BLEU as SacreBLEU
    metric = SacreBLEU(tokenize="char")
    result = metric.sentence_score(hypothesis, [reference])
    return round(result.score / 100, 4)  # 归一化到 0-1


def semantic_similarity(hypothesis: str, reference: str) -> float:
    vecs = encode_sync([hypothesis, reference], normalize=True)
    sim = float(np.dot(vecs[0], vecs[1]))
    return round(sim, 4)


def compute_all(hypothesis: str, reference: str) -> dict[str, float]:
    return {
        "rouge1_f1": rouge1_f1(hypothesis, reference),
        "bleu": bleu(hypothesis, reference),
        "semantic_sim": semantic_similarity(hypothesis, reference),
    }
