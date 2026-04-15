"""
FastAPI 应用入口 — lifespan 管理资源
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from codes.config import settings
from codes.embedding import _load_model       # 预热模型
from codes.service.routes import router
from codes.service.session import store
from codes.vector_store import load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：预热 embedding 模型 + 加载 FAISS 索引 + 启动 session 清理
    print("正在加载 bge-m3 模型...")
    _load_model()
    print("正在加载 FAISS 索引...")
    try:
        load_index()
        print("索引加载完成")
    except FileNotFoundError as e:
        print(f"[警告] {e}")
        print("请先运行: uv run python -m codes.vector_store")
    store.start()
    yield
    # 关闭：停止 session 清理
    store.stop()


app = FastAPI(
    title="ぽちゃガチョ！ 客服系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
