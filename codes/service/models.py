"""
Pydantic v2 请求/响应模型
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# -------- 请求 --------

class NewTicketRequest(BaseModel):
    question: str = Field(..., description="玩家日语原文邮件内容")


class FinalizeRequest(BaseModel):
    session_id: str
    draft_zh: str = Field(..., description="客服确认/编辑后的中文草稿")


class FollowupRequest(BaseModel):
    session_id: str
    message: str = Field(..., description="玩家追加日语消息")


# -------- 响应 --------

class SimilarCase(BaseModel):
    mail_id: str | int
    question: str
    answer: str
    inquiry_type: str
    fused_score: float = 0.0


class NewTicketResponse(BaseModel):
    session_id: str
    question_zh: str
    intent: str
    draft_zh: str
    similar_cases: list[SimilarCase]
    confidence: Literal["high", "normal"]
    created_at: datetime


class FinalizeResponse(BaseModel):
    session_id: str
    reply_ja: str
    finalized_at: datetime


class FollowupResponse(BaseModel):
    session_id: str
    followup_zh: str
    followup_type: str
    intent: str
    draft_zh: str
    similar_cases: list[SimilarCase]
    confidence: Literal["high", "normal"]


class ConversationTurn(BaseModel):
    role: Literal["player", "staff"]
    content_zh: str
    timestamp: datetime


class SessionResponse(BaseModel):
    session_id: str
    intent: str
    state: str
    turns: list[ConversationTurn]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    model: str
    index_loaded: bool
