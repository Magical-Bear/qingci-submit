"""
BM25 + 向量混合检索
BM25 0.3 / 向量 0.7
"""
import asyncio
import pickle
from functools import lru_cache

import faiss
import fugashi
import numpy as np
from rank_bm25 import BM25Okapi

from codes.config import BM25_PATH, settings
from codes.embedding import encode
from codes.vector_store import load_index, search as vector_search


# --------------- BM25 构建/加载 ---------------

def build_bm25(corpus: list[str]) -> BM25Okapi:
    tokenized = [_tokenize(q) for q in corpus]
    return BM25Okapi(tokenized)


def save_bm25(bm25: BM25Okapi) -> None:
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)


@lru_cache(maxsize=1)
def _load_bm25() -> BM25Okapi:
    if not BM25_PATH.exists():
        raise FileNotFoundError("BM25 索引不存在，请先运行 python -m codes.vector_store")
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


# --------------- 分词 (fugashi/MeCab) ---------------

_tagger: fugashi.Tagger | None = None


def _get_tagger() -> fugashi.Tagger:
    global _tagger
    if _tagger is None:
        _tagger = fugashi.Tagger()
    return _tagger


def _tokenize(text: str) -> list[str]:
    tagger = _get_tagger()
    return [word.surface for word in tagger(text) if word.surface.strip()]


# --------------- 混合检索主函数 ---------------

async def hybrid_search(
    query: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    BM25 + 向量混合检索
    返回 top_k 结果，按融合分数降序
    """
    k = top_k or settings.top_k
    index, metadata = load_index()
    bm25 = _load_bm25()

    # 并发执行 BM25 检索 + 向量编码
    query_vec, bm25_scores = await asyncio.gather(
        encode(query),
        asyncio.to_thread(_bm25_scores, bm25, query),
    )

    # 向量检索（取 top_k * 3 候选，后续融合排序）
    candidate_k = min(k * 3, len(metadata))
    vector_results = await vector_search(query_vec[0], index, metadata, top_k=candidate_k)

    # 归一化向量分数
    v_scores = np.array([r["vector_score"] for r in vector_results])
    v_scores = _minmax_norm(v_scores)

    # 归一化 BM25 分数（全量）
    b_scores_norm = _minmax_norm(bm25_scores)

    # 融合
    fused: dict[int, dict] = {}
    for i, r in enumerate(vector_results):
        # 找到该条在全量 metadata 中的原始 index
        orig_idx = _find_idx(metadata, r)
        bm25_s = float(b_scores_norm[orig_idx])
        v_s = float(v_scores[i])
        fused[orig_idx] = {
            **r,
            "vector_score": v_s,
            "bm25_score": bm25_s,
            "fused_score": settings.vector_weight * v_s + settings.bm25_weight * bm25_s,
        }

    # 对未出现在向量结果中但 BM25 top-k 的条目补充进来
    top_bm25_idx = np.argsort(b_scores_norm)[::-1][:candidate_k]
    for idx in top_bm25_idx:
        if idx not in fused:
            bm25_s = float(b_scores_norm[idx])
            fused[idx] = {
                **metadata[idx],
                "vector_score": 0.0,
                "bm25_score": bm25_s,
                "fused_score": settings.bm25_weight * bm25_s,
            }

    ranked = sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)
    return ranked[:k]


def _bm25_scores(bm25: BM25Okapi, query: str) -> np.ndarray:
    tokens = _tokenize(query)
    return np.array(bm25.get_scores(tokens), dtype=np.float32)


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _find_idx(metadata: list[dict], record: dict) -> int:
    """根据 mail_id + question 找原始索引（O(n)，数据量小无所谓）"""
    for i, m in enumerate(metadata):
        if m["mail_id"] == record["mail_id"] and m["question"] == record["question"]:
            return i
    return 0
