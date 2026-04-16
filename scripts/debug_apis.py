"""
单接口逐个调试脚本 — 逐步测试每个 API，打印详细耗时。
运行前确保服务已启动: uv run uvicorn codes.service.app:app --reload --port 8000

用法:
    uv run python scripts/debug_apis.py                # 全部接口
    uv run python scripts/debug_apis.py new            # 只测 /api/ticket/new
    uv run python scripts/debug_apis.py finalize       # 只测 /api/ticket/finalize
    uv run python scripts/debug_apis.py followup       # 只测 /api/ticket/followup
    uv run python scripts/debug_apis.py session        # 只测 /api/session/{id}
    uv run python scripts/debug_apis.py health         # 只测 /api/health
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 300.0  # 单次请求最长等待时间（秒）

QUESTION = "ゲームが起動しません。iPhoneを使用しています。"
DRAFT_ZH = "感谢您的来信。关于游戏无法启动的问题，请先尝试重启应用程序，如问题持续请联系我们。"
FOLLOWUP = "再起動しましたが、まだ起動しません。"


def fmt(seconds: float) -> str:
    return f"{seconds:.2f}s"


async def call(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    label: str,
    **kwargs: Any,
) -> tuple[float, httpx.Response]:
    print(f"\n{'='*60}")
    print(f"  {method} {url}  [{label}]")
    if "json" in kwargs:
        body_preview = str(kwargs["json"])[:120]
        print(f"  body: {body_preview}")
    print(f"  ...")
    t0 = time.perf_counter()
    resp = await client.request(method, url, **kwargs)
    elapsed = time.perf_counter() - t0
    status_mark = "OK" if resp.status_code < 400 else "FAIL"
    print(f"  [{status_mark}] {resp.status_code}  耗时: {fmt(elapsed)}")
    try:
        body = resp.json()
        preview = str(body)[:300]
        print(f"  响应: {preview}")
    except Exception:
        print(f"  响应(raw): {resp.text[:200]}")
    return elapsed, resp


# ----------------------------------------------------------------
# 各接口测试函数
# ----------------------------------------------------------------

async def test_health(client: httpx.AsyncClient) -> None:
    print("\n" + "#"*60)
    print("# GET /api/health")
    print("#"*60)
    elapsed, resp = await call(client, "GET", "/api/health", "health check")
    assert resp.status_code == 200, f"health 失败: {resp.text}"
    print(f"\n[PASS] health  耗时={fmt(elapsed)}")


async def test_new_ticket(client: httpx.AsyncClient) -> str:
    """返回创建的 session_id"""
    print("\n" + "#"*60)
    print("# POST /api/ticket/new")
    print("#"*60)
    elapsed, resp = await call(
        client, "POST", "/api/ticket/new", "new ticket",
        json={"question": QUESTION},
    )
    assert resp.status_code == 200, f"new ticket 失败: {resp.text}"
    sid = resp.json()["session_id"]
    print(f"\n[PASS] new_ticket  耗时={fmt(elapsed)}  session_id={sid}")
    return sid


async def test_finalize(client: httpx.AsyncClient, session_id: str | None = None) -> str:
    print("\n" + "#"*60)
    print("# POST /api/ticket/finalize")
    print("#"*60)
    if session_id is None:
        print("  → 先创建 session...")
        t0 = time.perf_counter()
        resp = await client.post("/api/ticket/new", json={"question": QUESTION})
        assert resp.status_code == 200, f"创建 session 失败: {resp.text}"
        session_id = resp.json()["session_id"]
        print(f"  session 创建耗时: {fmt(time.perf_counter() - t0)}  id={session_id}")

    elapsed, resp = await call(
        client, "POST", "/api/ticket/finalize", "finalize",
        json={"session_id": session_id, "draft_zh": DRAFT_ZH},
    )
    assert resp.status_code == 200, f"finalize 失败: {resp.text}"
    print(f"\n[PASS] finalize  耗时={fmt(elapsed)}")
    return session_id


async def test_followup(client: httpx.AsyncClient, session_id: str | None = None) -> None:
    print("\n" + "#"*60)
    print("# POST /api/ticket/followup")
    print("#"*60)
    if session_id is None:
        print("  → 先 new + finalize 建立 AWAITING_PLAYER 状态...")
        t0 = time.perf_counter()
        resp = await client.post("/api/ticket/new", json={"question": QUESTION})
        assert resp.status_code == 200
        sid = resp.json()["session_id"]
        resp2 = await client.post(
            "/api/ticket/finalize",
            json={"session_id": sid, "draft_zh": DRAFT_ZH},
        )
        assert resp2.status_code == 200, f"finalize 失败: {resp2.text}"
        print(f"  前置 new+finalize 耗时: {fmt(time.perf_counter() - t0)}  id={sid}")
        session_id = sid

    elapsed, resp = await call(
        client, "POST", "/api/ticket/followup", "followup",
        json={"session_id": session_id, "message": FOLLOWUP},
    )
    assert resp.status_code == 200, f"followup 失败: {resp.text}"
    print(f"\n[PASS] followup  耗时={fmt(elapsed)}")


async def test_get_session(client: httpx.AsyncClient, session_id: str | None = None) -> None:
    print("\n" + "#"*60)
    print("# GET /api/session/{id}")
    print("#"*60)
    if session_id is None:
        resp = await client.post("/api/ticket/new", json={"question": QUESTION})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        print(f"  session_id={session_id}")

    elapsed, resp = await call(
        client, "GET", f"/api/session/{session_id}", "get session",
    )
    assert resp.status_code == 200, f"get_session 失败: {resp.text}"
    print(f"\n[PASS] get_session  耗时={fmt(elapsed)}")


# ----------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------

async def main(target: str = "all") -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        # 先确认服务可达
        try:
            pong = await client.get("/api/health")
            assert pong.status_code == 200
            print(f"服务已就绪: {BASE_URL}")
        except Exception as e:
            print(f"[ERROR] 无法连接服务 {BASE_URL}: {e}")
            print("请先运行: uv run uvicorn codes.service.app:app --reload --port 8000")
            sys.exit(1)

        sid: str | None = None

        if target in ("all", "health"):
            await test_health(client)

        if target in ("all", "new"):
            sid = await test_new_ticket(client)

        if target in ("all", "finalize"):
            await test_finalize(client, sid if target == "all" else None)
            # 如果是全量测试，重新建一个 session 保持状态干净
            if target == "all":
                resp = await client.post("/api/ticket/new", json={"question": QUESTION})
                assert resp.status_code == 200
                sid = resp.json()["session_id"]
                await client.post(
                    "/api/ticket/finalize",
                    json={"session_id": sid, "draft_zh": DRAFT_ZH},
                )

        if target in ("all", "followup"):
            await test_followup(client, sid if target == "all" else None)

        if target in ("all", "session"):
            await test_get_session(client)

        print("\n" + "="*60)
        print("全部指定接口测试完成")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    asyncio.run(main(target))
