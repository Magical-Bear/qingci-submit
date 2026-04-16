"""
bge-m3 稀疏 + 向量混合检索
稀疏 0.3 / 向量 0.7（使用 bge-m3 sparse_linear.pt，替代 BM25）
"""
import asyncio
import pickle
from functools import lru_cache

import numpy as np

from codes.config import SPARSE_PATH, settings
from codes.embedding import encode_both
from codes.vector_store import load_index, search as vector_search


# --------------- 稀疏向量加载 ---------------

@lru_cache(maxsize=1)
def _load_sparse_vectors() -> list[dict]:
    if not SPARSE_PATH.exists():
        raise FileNotFoundError(
            "稀疏向量索引不存在，请先运行 python -m codes.vector_store"
        )
    with open(SPARSE_PATH, "rb") as f:
        return pickle.load(f)


# --------------- 稀疏相似度计算 ---------------

def _sparse_scores(query_sparse: dict, corpus: list[dict]) -> np.ndarray:
    """计算 query 与语料库中每条文档的稀疏点积分数（SPLADE 风格）"""
    scores = np.zeros(len(corpus), dtype=np.float32)
    q_keys = set(query_sparse)
    for i, doc_sparse in enumerate(corpus):
        shared = q_keys & set(doc_sparse)
        if shared:
            scores[i] = sum(query_sparse[t] * doc_sparse[t] for t in shared)
    return scores


# --------------- 混合检索主函数 ---------------

async def hybrid_search(
    query: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    bge-m3 稀疏 + 向量混合检索
    返回 top_k 结果，按融合分数降序
    """
    k = top_k or settings.top_k
    index, metadata = load_index()
    sparse_corpus = _load_sparse_vectors()

    # 单次 to_thread 完成稠密 + 稀疏编码（避免 tokenizer 并发冲突）
    query_dense, query_sparse_list = await encode_both(query)
    query_sparse = query_sparse_list[0]

    # 向量检索（取 top_k * 3 候选，后续融合排序）
    candidate_k = min(k * 3, len(metadata))
    vector_results = await vector_search(query_dense[0], index, metadata, top_k=candidate_k)

    # 稀疏分数（全量）
    s_scores_raw = await asyncio.to_thread(_sparse_scores, query_sparse, sparse_corpus)

    # 归一化向量分数
    v_scores = np.array([r["vector_score"] for r in vector_results])
    v_scores = _minmax_norm(v_scores)

    # 归一化稀疏分数（全量）
    s_scores_norm = _minmax_norm(s_scores_raw)

    # 融合
    fused: dict[int, dict] = {}
    for i, r in enumerate(vector_results):
        orig_idx = _find_idx(metadata, r)
        s_s = float(s_scores_norm[orig_idx])
        v_s = float(v_scores[i])
        fused[orig_idx] = {
            **r,
            "vector_score": v_s,
            "sparse_score": s_s,
            "fused_score": settings.dense_weight * v_s + settings.sparse_weight * s_s,
        }

    # 对未出现在向量结果中但稀疏 top-k 的条目补充进来
    top_sparse_idx = np.argsort(s_scores_norm)[::-1][:candidate_k]
    for idx in top_sparse_idx:
        if idx not in fused:
            s_s = float(s_scores_norm[idx])
            fused[idx] = {
                **metadata[idx],
                "vector_score": 0.0,
                "sparse_score": s_s,
                "fused_score": settings.sparse_weight * s_s,
            }

    ranked = sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)
    return ranked[:k]


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
