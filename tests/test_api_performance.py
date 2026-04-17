"""
接口性能测试
每个接口测试 5 次，统计平均耗时与响应时间波动，生成 Markdown 报告。
运行前须确保服务已启动: uv run uvicorn codes.service.app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

import httpx
import pytest

BASE_URL = "http://localhost:8000"
RUNS = 5

# ---------- 测试用例数据 ----------

NEW_TICKET_QUESTIONS = [
    "ゲームが起動しません。iPhoneを使用しています。",
    "課金したのにアイテムが届きません。",
    "アカウントを機種変更したいのですが、データは引き継げますか？",
    "ゲームの操作方法について教えてください。",
    "ログインできなくなりました。パスワードを忘れてしまいました。",
]

FOLLOWUP_MESSAGES = [
    "再起動しましたが、まだ起動しません。",
    "それでもアイテムが届いていません。",
    "引き継ぎコードはどこで確認できますか？",
    "もう少し詳しく教えていただけますか？",
    "パスワードリセットのメールが届きません。",
]

DRAFT_ZH_TEMPLATES = [
    "感谢您的来信。关于游戏无法启动的问题，请先尝试重启应用程序，如问题持续请联系我们。",
    "感谢您的联系。关于充值后未收到道具，我们将立即为您核查，请稍候。",
    "感谢您的咨询。关于账号数据继承，请在旧设备上获取转移码后，在新设备上输入即可。",
    "感谢您的咨询。请参考游戏内教程，如有其他问题随时联系我们。",
    "感谢您的联系。请点击登录页面的「忘记密码」，通过注册邮箱重置密码。",
]


# ---------- 辅助函数 ----------

async def timed_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> tuple[float, httpx.Response]:
    """执行一次 HTTP 请求并返回 (耗时秒数, response)"""
    start = time.perf_counter()
    resp = await client.request(method, url, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, resp


def stats(times: list[float]) -> dict[str, float]:
    """计算基本统计量（秒）"""
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "mean": mean,
        "min": min(times),
        "max": max(times),
        "stdev": stdev,
        "p50": statistics.median(times),
    }


# ---------- Fixture ----------

@pytest.fixture(scope="session")
def perf_results() -> dict[str, dict]:
    """收集所有接口的性能数据，供最后生成报告使用"""
    return {}


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ---------- 测试类 ----------

@pytest.mark.asyncio
class TestHealthPerformance:
    """GET /api/health — 5 次独立调用"""

    async def test_health_5_runs(self, perf_results: dict) -> None:
        times: list[float] = []
        statuses: list[int] = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            for _ in range(RUNS):
                elapsed, resp = await timed_request(client, "GET", "/api/health")
                times.append(elapsed)
                statuses.append(resp.status_code)

        assert all(s == 200 for s in statuses), f"部分 health 请求失败: {statuses}"
        perf_results["GET /api/health"] = {
            "runs": RUNS,
            "times": times,
            **stats(times),
        }


@pytest.mark.asyncio
class TestNewTicketPerformance:
    """POST /api/ticket/new — 5 次独立调用（不同问题）"""

    async def test_new_ticket_5_runs(self, perf_results: dict) -> None:
        times: list[float] = []
        statuses: list[int] = []
        session_ids: list[str] = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=180.0) as client:
            for i in range(RUNS):
                payload = {"question": NEW_TICKET_QUESTIONS[i]}
                elapsed, resp = await timed_request(
                    client, "POST", "/api/ticket/new", json=payload
                )
                times.append(elapsed)
                statuses.append(resp.status_code)
                if resp.status_code == 200:
                    session_ids.append(resp.json()["session_id"])

        assert all(s == 200 for s in statuses), f"部分 new ticket 请求失败: {statuses}"
        perf_results["POST /api/ticket/new"] = {
            "runs": RUNS,
            "times": times,
            "session_ids": session_ids,
            **stats(times),
        }


@pytest.mark.asyncio
class TestFinalizePerformance:
    """POST /api/ticket/finalize — 为每次测试先创建独立 session"""

    async def _create_session(
        self, client: httpx.AsyncClient, question: str
    ) -> str:
        resp = await client.post(
            "/api/ticket/new", json={"question": question}
        )
        assert resp.status_code == 200, f"创建 session 失败: {resp.text}"
        return resp.json()["session_id"]

    async def test_finalize_5_runs(self, perf_results: dict) -> None:
        times: list[float] = []
        statuses: list[int] = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=180.0) as client:
            for i in range(RUNS):
                sid = await self._create_session(client, NEW_TICKET_QUESTIONS[i])
                payload = {"session_id": sid, "draft_zh": DRAFT_ZH_TEMPLATES[i]}
                elapsed, resp = await timed_request(
                    client, "POST", "/api/ticket/finalize", json=payload
                )
                times.append(elapsed)
                statuses.append(resp.status_code)

        assert all(s == 200 for s in statuses), f"部分 finalize 请求失败: {statuses}"
        perf_results["POST /api/ticket/finalize"] = {
            "runs": RUNS,
            "times": times,
            **stats(times),
        }


@pytest.mark.asyncio
class TestFollowupPerformance:
    """POST /api/ticket/followup — 每次先 new→finalize 建立 AWAITING_PLAYER 状态"""

    async def _prepare_session(
        self, client: httpx.AsyncClient, idx: int
    ) -> str:
        resp = await client.post(
            "/api/ticket/new",
            json={"question": NEW_TICKET_QUESTIONS[idx]},
        )
        assert resp.status_code == 200
        sid = resp.json()["session_id"]

        resp2 = await client.post(
            "/api/ticket/finalize",
            json={"session_id": sid, "draft_zh": DRAFT_ZH_TEMPLATES[idx]},
        )
        assert resp2.status_code == 200, f"finalize 失败: {resp2.text}"
        return sid

    async def test_followup_5_runs(self, perf_results: dict) -> None:
        times: list[float] = []
        statuses: list[int] = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=300.0) as client:
            for i in range(RUNS):
                sid = await self._prepare_session(client, i)
                payload = {"session_id": sid, "message": FOLLOWUP_MESSAGES[i]}
                elapsed, resp = await timed_request(
                    client, "POST", "/api/ticket/followup", json=payload
                )
                times.append(elapsed)
                statuses.append(resp.status_code)

        assert all(s == 200 for s in statuses), f"部分 followup 请求失败: {statuses}"
        perf_results["POST /api/ticket/followup"] = {
            "runs": RUNS,
            "times": times,
            **stats(times),
        }


@pytest.mark.asyncio
class TestGetSessionPerformance:
    """GET /api/session/{id} — 先创建 session，再查询 5 次"""

    async def test_get_session_5_runs(self, perf_results: dict) -> None:
        times: list[float] = []
        statuses: list[int] = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
            # 创建一个 session 供后续查询
            resp = await client.post(
                "/api/ticket/new",
                json={"question": NEW_TICKET_QUESTIONS[0]},
            )
            assert resp.status_code == 200, f"创建 session 失败: {resp.text}"
            sid = resp.json()["session_id"]

            for _ in range(RUNS):
                elapsed, resp = await timed_request(
                    client, "GET", f"/api/session/{sid}"
                )
                times.append(elapsed)
                statuses.append(resp.status_code)

        assert all(s == 200 for s in statuses), f"部分 get_session 请求失败: {statuses}"
        perf_results["GET /api/session/{id}"] = {
            "runs": RUNS,
            "times": times,
            **stats(times),
        }


# ---------- 404 / 错误场景耗时 ----------

@pytest.mark.asyncio
class TestErrorResponsePerformance:
    """错误场景接口响应速度（不计入主报告，仅作参考）"""

    async def test_404_session_response_time(self, perf_results: dict) -> None:
        times: list[float] = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for _ in range(RUNS):
                elapsed, resp = await timed_request(
                    client, "GET", "/api/session/nonexistent-uuid-00000000"
                )
                times.append(elapsed)
                assert resp.status_code == 404

        perf_results["GET /api/session/{id} (404)"] = {
            "runs": RUNS,
            "times": times,
            **stats(times),
        }
