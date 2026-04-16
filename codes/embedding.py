"""
bge-m3 单例加载 + 编码
重任务(批量编码)扔进程池，轻任务直接 asyncio.to_thread
"""
import asyncio
from functools import lru_cache
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

from codes.config import WEIGHTS_DIR


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(str(WEIGHTS_DIR), device="cpu")


def encode_sync(texts: Union[str, list[str]], normalize: bool = True, batch_size: int = 64) -> np.ndarray:
    model = _load_model()
    if isinstance(texts, str):
        texts = [texts]
    vecs = model.encode(
        texts,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    return vecs


async def encode(texts: Union[str, list[str]], normalize: bool = True) -> np.ndarray:
    """异步编码，防止阻塞事件循环"""
    return await asyncio.to_thread(encode_sync, texts, normalize)


@lru_cache(maxsize=1)
def _load_sparse_linear():
    """加载并缓存 bge-m3 稀疏头 (sparse_linear.pt → nn.Linear(1024, 1))"""
    import torch
    import torch.nn as nn
    state = torch.load(WEIGHTS_DIR / "sparse_linear.pt", map_location="cpu")
    layer = nn.Linear(1024, 1, bias=True)
    layer.load_state_dict(state)
    layer.eval()
    return layer


def encode_sparse_sync(
    texts: Union[str, list[str]],
    batch_size: int = 64,
) -> list[dict]:
    """
    使用 bge-m3 稀疏头计算 SPLADE 风格的词法权重。
    返回每条文本的 {token_id: max_weight} 字典列表。
    复用已缓存的 SentenceTransformer tokenizer 和 base model，不重复加载。
    """
    import torch
    if isinstance(texts, str):
        texts = [texts]

    st_model = _load_model()
    tokenizer = st_model[0].tokenizer
    base_model = st_model[0].auto_model
    sparse_linear = _load_sparse_linear()

    all_results: list[dict] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = base_model(**inputs)
            token_emb = outputs.last_hidden_state.float()    # (B, L, 1024)
            tok_w = torch.relu(sparse_linear(token_emb))     # (B, L, 1)

        attention_mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]

        for b in range(len(batch)):
            lexical: dict[int, float] = {}
            for pos in range(input_ids.shape[1]):
                if attention_mask[b, pos] == 0:
                    continue
                tok_id = int(input_ids[b, pos].item())
                weight = float(tok_w[b, pos, 0].item())
                if weight > 0.0:
                    if tok_id not in lexical or lexical[tok_id] < weight:
                        lexical[tok_id] = weight
            all_results.append(lexical)

    return all_results


async def encode_sparse(
    texts: Union[str, list[str]],
    batch_size: int = 64,
) -> list[dict]:
    """异步稀疏编码，防止阻塞事件循环"""
    return await asyncio.to_thread(encode_sparse_sync, texts, batch_size)


def encode_both_sync(
    texts: Union[str, list[str]],
    normalize: bool = True,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[dict]]:
    """
    在同一线程中顺序执行稠密编码和稀疏编码。
    避免两个 to_thread 并发访问同一 tokenizer（Rust borrow checker 会 panic）。
    返回 (dense_vecs, sparse_dicts)。
    """
    dense = encode_sync(texts, normalize=normalize, batch_size=batch_size)
    sparse = encode_sparse_sync(texts, batch_size=batch_size)
    return dense, sparse


async def encode_both(
    texts: Union[str, list[str]],
    normalize: bool = True,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[dict]]:
    """异步版：单次 to_thread 完成稠密 + 稀疏编码，避免 tokenizer 并发冲突"""
    return await asyncio.to_thread(encode_both_sync, texts, normalize, batch_size)
