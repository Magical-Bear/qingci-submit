"""
UUID 会话管理器 — 内存存储，TTL 自动清理
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Literal

from codes.config import settings


class SessionStore:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._ttl = timedelta(hours=settings.session_ttl_hours)
        self._cleanup_task: asyncio.Task | None = None

    def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def stop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()

    def create(self) -> str:
        sid = str(uuid.uuid4())
        self._store[sid] = {
            "session_id": sid,
            "intent": "未知",
            "state": "DRAFT_PENDING",
            "turns": [],
            "pipeline_state": {},   # 存 pipeline 中间状态，供 finalize/followup 使用
            "created_at": datetime.utcnow(),
        }
        return sid

    def get(self, sid: str) -> dict | None:
        session = self._store.get(sid)
        if session is None:
            return None
        if datetime.utcnow() - session["created_at"] > self._ttl:
            del self._store[sid]
            return None
        return session

    def update(self, sid: str, **kwargs: object) -> None:
        if sid in self._store:
            self._store[sid].update(kwargs)

    def append_turn(self, sid: str, role: str, content_zh: str) -> None:
        session = self._store.get(sid)
        if session:
            session["turns"].append({
                "role": role,
                "content_zh": content_zh,
                "timestamp": datetime.utcnow(),
            })

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(3600)
            now = datetime.utcnow()
            expired = [
                sid for sid, s in self._store.items()
                if now - s["created_at"] > self._ttl
            ]
            for sid in expired:
                del self._store[sid]


# 全局单例，在 lifespan 中启动/停止
store = SessionStore()
