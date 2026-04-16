"""
全流程编排 — LangGraph StateGraph (Pipeline 模式，非 ReAct Agent)

节点流程:
  新工单: translate_and_classify → retrieve → generate_draft
  确认:   finalize → translate_zh_to_ja
  追加:   translate_and_classify_followup → retrieve → generate_followup_draft
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from codes import baidu_translate, llm, prompts
from codes.config import settings
from codes.retriever import hybrid_search

_logger = logging.getLogger(__name__)


def _t(label: str, start: float) -> None:
    """如果 ENABLE_TIMING=true，打印节点耗时。"""
    if settings.enable_timing:
        _logger.warning("[TIMING] %-28s %.3fs", label, time.perf_counter() - start)


# ============================================================
# State 定义
# ============================================================

class TicketState(TypedDict, total=False):
    # 输入
    question_ja: str                  # 玩家日语原文
    draft_zh_confirmed: str           # 客服确认后的中文草稿
    followup_ja: str                  # 追加日语消息

    # 中间产物
    question_zh: str                  # 翻译后中文
    intent: str                       # 意图分类
    similar_cases: list[dict]         # RAG 检索结果
    followup_zh: str                  # 追加消息中文
    followup_type: str                # 追加类型

    # 输出
    draft_zh: str                     # 中文草稿 (待客服确认)
    reply_ja: str                     # 最终日语回复
    confidence: Literal["high", "normal"]  # 高置信直接匹配标记

    # 多轮历史 (role: "player"|"staff", content_zh: str)
    conversation_history: list[dict]

    # 内部状态
    _step: str                        # 当前执行步骤标记


# ============================================================
# 节点实现
# ============================================================

async def node_translate_and_classify(state: TicketState) -> dict:
    """百度翻译(JA→ZH) 与 Kimi 意图分类并行执行"""
    _start = time.perf_counter()
    question_ja = state["question_ja"]

    translate_task = baidu_translate.translate_ja_to_zh(question_ja)
    classify_task = llm.chat(prompts.classify_intent(question_ja))
    question_zh, intent_raw = await asyncio.gather(translate_task, classify_task)

    intent_ja = _normalize_intent(intent_raw)
    intent = _INTENT_ZH_MAP.get(intent_ja, intent_ja)
    _t("node_translate_and_classify", _start)
    return {"question_zh": question_zh, "intent": intent, "_step": "translated_and_classified"}


async def node_retrieve(state: TicketState) -> dict:
    _start = time.perf_counter()
    query = state["question_zh"]
    # 如果是追加消息，拼接初始问题提升召回
    if "followup_zh" in state and state["followup_zh"]:
        query = f"{state['question_zh']} {state['followup_zh']}"

    cases = await hybrid_search(query, top_k=settings.top_k)

    # 高置信判断
    confidence: Literal["high", "normal"] = "normal"
    if (
        cases
        and cases[0].get("vector_score", 0) > settings.high_confidence_threshold
        and cases[0].get("inquiry_type", "") == state.get("intent", "")
    ):
        confidence = "high"

    _t("node_retrieve", _start)
    return {"similar_cases": cases, "confidence": confidence, "_step": "retrieved"}


async def node_generate_draft(state: TicketState) -> dict:
    _start = time.perf_counter()
    # 高置信直接用历史回复翻成中文草稿，跳过 LLM 生成
    if state.get("confidence") == "high":
        top_case = state["similar_cases"][0]
        draft_zh = await baidu_translate.translate_ja_to_zh(top_case["answer"])
        _t("node_generate_draft (high-conf translate)", _start)
        return {"draft_zh": draft_zh, "_step": "draft_generated"}

    messages = prompts.generate_draft_zh(
        question_zh=state["question_zh"],
        intent=state["intent"],
        similar_cases=state["similar_cases"],
        conversation_history=state.get("conversation_history"),
    )
    draft_zh = await llm.chat(messages)
    _t("node_generate_draft", _start)
    return {"draft_zh": draft_zh, "_step": "draft_generated"}


async def node_finalize(state: TicketState) -> dict:
    """客服确认草稿 → 翻译成日语敬语"""
    _start = time.perf_counter()
    messages = prompts.translate_zh_to_ja_polite(state["draft_zh_confirmed"])
    reply_ja = await llm.chat(messages)

    # 追加到对话历史
    history = list(state.get("conversation_history") or [])
    history.append({"role": "player", "content_zh": state["question_zh"]})
    history.append({"role": "staff", "content_zh": state["draft_zh_confirmed"]})

    _t("node_finalize", _start)
    return {
        "reply_ja": reply_ja,
        "conversation_history": history,
        "_step": "finalized",
    }


async def node_translate_and_classify_followup(state: TicketState) -> dict:
    """百度翻译(JA→ZH) 与 Kimi 追加分类并行执行"""
    _start = time.perf_counter()
    followup_ja = state["followup_ja"]

    history = state.get("conversation_history") or []
    history_summary = " → ".join(
        f"[{t['role']}] {t['content_zh'][:30]}..." for t in history[-4:]
    )

    translate_task = baidu_translate.translate_ja_to_zh(followup_ja)
    classify_task = llm.chat(prompts.classify_followup(followup_ja, history_summary))
    followup_zh, followup_type_raw = await asyncio.gather(translate_task, classify_task)

    followup_type_ja = followup_type_raw.strip()
    followup_type = _FOLLOWUP_TYPE_ZH_MAP.get(followup_type_ja, followup_type_ja)
    _t("node_translate_and_classify_followup", _start)
    return {
        "followup_zh": followup_zh,
        "followup_type": followup_type,
        "_step": "followup_translated_and_classified",
    }


# ============================================================
# 意图归一化
# ============================================================

_INTENT_MAP = {
    "不具合": "不具合",
    "bug": "不具合",
    "故障": "不具合",
    "購入": "購入問題",
    "购买": "購入問題",
    "意見": "意見建議",
    "建议": "意見建議",
    "データ": "データ継承",
    "数据": "データ継承",
    "继承": "データ継承",
}

# 日语意图标签 → 中文（供客服阅读）
_INTENT_ZH_MAP: dict[str, str] = {
    "不具合":   "故障问题",
    "購入問題": "购买问题",
    "意見建議": "意见建议",
    "データ継承": "数据继承",
    "未知":     "未知",
}

# 日语追加类型标签 → 中文（供客服阅读）
_FOLLOWUP_TYPE_ZH_MAP: dict[str, str] = {
    "補充情報":   "补充信息",
    "問題未解決": "问题未解决",
    "新問題":     "新问题",
    "確認":       "确认",
}


def _normalize_intent(raw: str) -> str:
    raw = raw.strip()
    for key, val in _INTENT_MAP.items():
        if key in raw:
            return val
    return raw if raw in prompts.INTENT_LABELS else "未知"


# ============================================================
# 图构建
# ============================================================

def _build_new_ticket_graph() -> Any:
    """新工单处理图: translate_and_classify → retrieve → generate_draft"""
    g = StateGraph(TicketState)
    g.add_node("translate_and_classify", node_translate_and_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate_draft", node_generate_draft)

    g.set_entry_point("translate_and_classify")
    g.add_edge("translate_and_classify", "retrieve")
    g.add_edge("retrieve", "generate_draft")
    g.add_edge("generate_draft", END)

    return g.compile(checkpointer=MemorySaver())


def _build_finalize_graph() -> Any:
    """确认草稿图: finalize (translate zh→ja)"""
    g = StateGraph(TicketState)
    g.add_node("finalize", node_finalize)
    g.set_entry_point("finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())


def _build_followup_graph() -> Any:
    """追加消息图: translate_and_classify_followup → retrieve → generate_draft"""
    g = StateGraph(TicketState)
    g.add_node("translate_and_classify_followup", node_translate_and_classify_followup)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate_draft", node_generate_draft)

    g.set_entry_point("translate_and_classify_followup")
    g.add_edge("translate_and_classify_followup", "retrieve")
    g.add_edge("retrieve", "generate_draft")
    g.add_edge("generate_draft", END)

    return g.compile(checkpointer=MemorySaver())


# 全局图实例（服务启动时初始化一次）
new_ticket_graph = _build_new_ticket_graph()
finalize_graph = _build_finalize_graph()
followup_graph = _build_followup_graph()


# ============================================================
# 公开 API
# ============================================================

async def process_new_ticket(
    session_id: str,
    question_ja: str,
) -> dict:
    config = {"configurable": {"thread_id": f"{session_id}_new"}}
    result = await new_ticket_graph.ainvoke(
        {"question_ja": question_ja, "conversation_history": []},
        config=config,
    )
    return {
        "question_zh": result["question_zh"],
        "intent": result["intent"],
        "similar_cases": result["similar_cases"],
        "draft_zh": result["draft_zh"],
        "confidence": result.get("confidence", "normal"),
    }


async def process_finalize(
    session_id: str,
    draft_zh_confirmed: str,
    current_state: dict,
) -> dict:
    config = {"configurable": {"thread_id": f"{session_id}_finalize"}}
    result = await finalize_graph.ainvoke(
        {
            **current_state,
            "draft_zh_confirmed": draft_zh_confirmed,
        },
        config=config,
    )
    return {
        "reply_ja": result["reply_ja"],
        "conversation_history": result["conversation_history"],
    }


async def process_followup(
    session_id: str,
    followup_ja: str,
    current_state: dict,
) -> dict:
    config = {"configurable": {"thread_id": f"{session_id}_followup"}}
    result = await followup_graph.ainvoke(
        {
            **current_state,
            "followup_ja": followup_ja,
        },
        config=config,
    )
    return {
        "followup_zh": result["followup_zh"],
        "followup_type": result.get("followup_type", ""),
        "similar_cases": result["similar_cases"],
        "draft_zh": result["draft_zh"],
        "confidence": result.get("confidence", "normal"),
    }
