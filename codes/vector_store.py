"""
FAISS IndexFlatIP 构建 / 保存 / 加载
只索引 question_content（核心问题内容，去掉表单结构）
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"       # 防止 faiss OpenMP 与 PyTorch OpenMP 冲突（macOS ARM）
os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
import pickle
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from codes.config import (
    DATASET_DIR,
    FAISS_INDEX_PATH,
    SPARSE_PATH,
    METADATA_PATH,
    INDEX_DIR,
)
from codes.embedding import encode_sync


def build_index(csv_path: Optional[str] = None, force: bool = False) -> None:
    """从 train.csv 构建 FAISS 索引、bge-m3 稀疏向量索引、metadata 并落盘。

    已存在的文件默认跳过，force=True 强制重建所有文件。
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    need_faiss = force or not FAISS_INDEX_PATH.exists()
    need_sparse = force or not SPARSE_PATH.exists()
    need_meta = force or not METADATA_PATH.exists()

    if not (need_faiss or need_sparse or need_meta):
        print("所有索引文件已存在，跳过构建。使用 force=True 强制重建。")
        return

    path = csv_path or str(DATASET_DIR / "train.csv")
    df = pd.read_csv(path)

    # 过滤 question 为空的行（NaN 会导致 encoder 崩溃）
    df = df.dropna(subset=["question"])
    questions = df["question"].tolist()
    metadata = df[["mail_id", "question", "answer", "inquiry_type"]].to_dict("records")

    # ---- metadata ----
    if need_meta:
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        print(f"metadata 已保存: {len(metadata)} 条")
    else:
        print("metadata 已存在，跳过。")

    # ---- 稀疏向量 (bge-m3 sparse_linear) ----
    if need_sparse:
        from codes.embedding import encode_sparse_sync
        print(f"构建 bge-m3 稀疏向量索引 ({len(questions)} 条)...")
        sparse_vecs: list[dict] = []
        for i in range(0, len(questions), 64):
            sparse_vecs.extend(encode_sparse_sync(questions[i: i + 64], batch_size=64))
        with open(SPARSE_PATH, "wb") as f:
            pickle.dump(sparse_vecs, f)
        print(f"稀疏向量索引已保存: {SPARSE_PATH} ({len(sparse_vecs)} 条)")
    else:
        print("稀疏向量索引已存在，跳过。")

    # ---- FAISS ----
    if need_faiss:
        print(f"编码 {len(questions)} 条问题...")
        # 手动分批编码，避免 sentence-transformers 内部多进程在 macOS 触发 segfault
        chunk_size = 64
        chunks = [encode_sync(questions[i:i+chunk_size]) for i in range(0, len(questions), chunk_size)]
        vecs = np.vstack(chunks).astype(np.float32)  # (N, 1024) float32

        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs.astype(np.float32))

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        print(f"FAISS 索引已保存: {index.ntotal} 条, dim={dim}")
    else:
        print("FAISS 索引已存在，跳过。")


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS 索引不存在，请先运行 python -m codes.vector_store")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


async def search(
    query_vec: np.ndarray,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """向量检索，返回 top_k 结果，附带 cosine score"""

    def _search():
        scores, indices = index.search(query_vec.reshape(1, -1).astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(metadata[idx])
            item["vector_score"] = float(score)
            results.append(item)
        return results

    return await asyncio.to_thread(_search)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重建所有索引文件")
    args = parser.parse_args()
    build_index(force=args.force)
