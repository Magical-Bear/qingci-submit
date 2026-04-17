"""
conftest.py — 共享 fixture + 测试结束后生成性能报告
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest


# ---------- 共享 perf_results fixture ----------

@pytest.fixture(scope="session")
def perf_results() -> dict:
    """跨测试类共享的性能数据收集器"""
    return {}


# ---------- 报告生成 ----------

def _fmt(seconds: float) -> str:
    """毫秒，保留 1 位小数"""
    return f"{seconds * 1000:.1f} ms"


def _bar(value: float, max_val: float, width: int = 20) -> str:
    """ASCII 进度条，用于直观对比各接口耗时"""
    if max_val == 0:
        return "-" * width
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def generate_report(perf_results: dict, report_dir: Path) -> Path:
    now = datetime.now()
    filename = now.strftime("%y-%m-%d-%H-%M") + ".md"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / filename

    lines: list[str] = []

    lines.append("# ぽちゃガチョ！ 客服系统 API 性能测试报告")
    lines.append("")
    lines.append(f"**测试时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**服务地址**: `http://localhost:8000`")
    lines.append(f"**每接口运行次数**: 5")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not perf_results:
        lines.append("> 无性能数据，请确认测试已正常运行。")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    # 汇总表
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 接口 | 均值 | 最小 | 最大 | 标准差 | P50 |")
    lines.append("|------|------|------|------|--------|-----|")

    for endpoint, data in perf_results.items():
        lines.append(
            f"| `{endpoint}` "
            f"| {_fmt(data['mean'])} "
            f"| {_fmt(data['min'])} "
            f"| {_fmt(data['max'])} "
            f"| {_fmt(data['stdev'])} "
            f"| {_fmt(data['p50'])} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # 各接口详情
    lines.append("## 各接口详情")
    lines.append("")

    max_mean = max((d["mean"] for d in perf_results.values()), default=1.0)

    for endpoint, data in perf_results.items():
        lines.append(f"### `{endpoint}`")
        lines.append("")

        # 可视化均值对比条
        bar = _bar(data["mean"], max_mean)
        lines.append(f"响应时间分布 (相对最慢接口): `{bar}` {_fmt(data['mean'])}")
        lines.append("")

        # 每次请求耗时
        lines.append("| # | 耗时 |")
        lines.append("|---|------|")
        for i, t in enumerate(data["times"], 1):
            lines.append(f"| {i} | {_fmt(t)} |")

        lines.append("")
        lines.append(f"- **均值**: {_fmt(data['mean'])}")
        lines.append(f"- **最小**: {_fmt(data['min'])}")
        lines.append(f"- **最大**: {_fmt(data['max'])}")
        lines.append(f"- **标准差**: {_fmt(data['stdev'])}（波动指标，越小越稳定）")
        lines.append(f"- **P50（中位数）**: {_fmt(data['p50'])}")

        # 波动评级
        if data["mean"] > 0:
            cv = data["stdev"] / data["mean"]  # 变异系数
            if cv < 0.1:
                stability = "稳定 ✓"
            elif cv < 0.25:
                stability = "一般"
            else:
                stability = "波动较大 ⚠"
        else:
            stability = "N/A"
        lines.append(f"- **稳定性**: {stability}（变异系数 CV = {data['stdev'] / data['mean']:.2%}）")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 耗时包含网络往返 + 服务端处理（翻译/LLM 调用/RAG 检索）")
    lines.append("- `finalize` / `followup` 测试前会先调用 `new` 建立 session，准备阶段耗时不计入统计")
    lines.append("- `GET /api/session/{id} (404)` 为错误路径基准，耗时极低属预期")
    lines.append("- 标准差越小、变异系数（CV）越低，表示响应时间越稳定")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """所有测试结束后触发，生成性能报告"""
    # 从 session 中取出 perf_results fixture 的值
    perf_results: dict = {}
    for item in session.items:
        fixture_manager = item.session._fixturemanager  # type: ignore[attr-defined]
        break
    else:
        return

    # 通过 fixture cache 获取 session-scoped perf_results
    try:
        cached = fixture_manager._arg2fixturedefs.get("perf_results")  # type: ignore[attr-defined]
        if cached:
            for fd in cached:
                if hasattr(fd, "cached_result") and fd.cached_result is not None:
                    perf_results = fd.cached_result[0]
                    break
    except Exception:
        pass

    if not perf_results:
        print("\n[report] perf_results 为空，跳过报告生成。")
        return

    root = Path(__file__).parent.parent
    report_dir = root / "report" / "apis"
    report_path = generate_report(perf_results, report_dir)
    print(f"\n[report] 性能报告已生成: {report_path}")
