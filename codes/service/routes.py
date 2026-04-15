"""
API 路由
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from codes import pipeline
from codes.config import settings
from codes.service.models import (
    ConversationTurn,
    FinalizeRequest,
    FinalizeResponse,
    FollowupRequest,
    FollowupResponse,
    HealthResponse,
    NewTicketRequest,
    NewTicketResponse,
    SessionResponse,
    SimilarCase,
)
from codes.service.session import store
from codes.vector_store import load_index

router = APIRouter(prefix="/api")


@router.post("/ticket/new", response_model=NewTicketResponse)
async def new_ticket(req: NewTicketRequest) -> NewTicketResponse:
    sid = store.create()

    result = await pipeline.process_new_ticket(
        session_id=sid,
        question_ja=req.question,
    )

    store.update(
        sid,
        intent=result["intent"],
        state="DRAFT_PENDING",
        pipeline_state={
            "question_zh": result["question_zh"],
            "intent": result["intent"],
            "conversation_history": [],
        },
    )

    return NewTicketResponse(
        session_id=sid,
        question_zh=result["question_zh"],
        intent=result["intent"],
        draft_zh=result["draft_zh"],
        similar_cases=[SimilarCase(**_safe_case(c)) for c in result["similar_cases"]],
        confidence=result["confidence"],
        created_at=datetime.utcnow(),
    )


@router.post("/ticket/finalize", response_model=FinalizeResponse)
async def finalize_ticket(req: FinalizeRequest) -> FinalizeResponse:
    session = store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    if session["state"] != "DRAFT_PENDING":
        raise HTTPException(status_code=400, detail=f"当前状态 {session['state']} 不允许确认")

    result = await pipeline.process_finalize(
        session_id=req.session_id,
        draft_zh_confirmed=req.draft_zh,
        current_state=session["pipeline_state"],
    )

    # 更新会话
    store.update(
        req.session_id,
        state="AWAITING_PLAYER",
        pipeline_state={
            **session["pipeline_state"],
            "conversation_history": result["conversation_history"],
        },
    )
    store.append_turn(req.session_id, "player", session["pipeline_state"].get("question_zh", ""))
    store.append_turn(req.session_id, "staff", req.draft_zh)

    return FinalizeResponse(
        session_id=req.session_id,
        reply_ja=result["reply_ja"],
        finalized_at=datetime.utcnow(),
    )


@router.post("/ticket/followup", response_model=FollowupResponse)
async def followup_ticket(req: FollowupRequest) -> FollowupResponse:
    session = store.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    result = await pipeline.process_followup(
        session_id=req.session_id,
        followup_ja=req.message,
        current_state=session["pipeline_state"],
    )

    store.update(
        req.session_id,
        state="DRAFT_PENDING",
        pipeline_state={
            **session["pipeline_state"],
            "question_zh": result["followup_zh"],  # 追加消息作为新的 question_zh
            "conversation_history": session["pipeline_state"].get("conversation_history", []),
        },
    )

    return FollowupResponse(
        session_id=req.session_id,
        followup_zh=result["followup_zh"],
        followup_type=result["followup_type"],
        intent=session["intent"],
        draft_zh=result["draft_zh"],
        similar_cases=[SimilarCase(**_safe_case(c)) for c in result["similar_cases"]],
        confidence=result["confidence"],
    )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return SessionResponse(
        session_id=session_id,
        intent=session["intent"],
        state=session["state"],
        turns=[ConversationTurn(**t) for t in session["turns"]],
        created_at=session["created_at"],
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    index_loaded = False
    try:
        load_index()
        index_loaded = True
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        model=settings.kimi_model,
        index_loaded=index_loaded,
    )


# -------- 工具函数 --------

def _safe_case(case: dict) -> dict:
    """确保 SimilarCase 所需字段存在"""
    return {
        "mail_id": case.get("mail_id", ""),
        "question": case.get("question", ""),
        "answer": case.get("answer", ""),
        "inquiry_type": case.get("inquiry_type", ""),
        "fused_score": case.get("fused_score", 0.0),
    }
