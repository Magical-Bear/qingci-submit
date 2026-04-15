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
