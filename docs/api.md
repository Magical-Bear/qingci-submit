# ぽちゃガチョ！ 客服系统 API 文档

**Base URL**: `http://localhost:8000`

---

## 对话流程

```
POST /api/ticket/new          ← 玩家日语来信，返回 session_id + 中文草稿
       ↓
POST /api/ticket/finalize     ← 客服确认/编辑草稿，返回日语回复
       ↓ (玩家有追加消息时)
POST /api/ticket/followup     ← 追加日语消息，返回新草稿
       ↓
POST /api/ticket/finalize     ← 再次确认...（循环）

GET  /api/session/{id}        ← 任意时刻查询完整对话历史
GET  /api/health              ← 服务健康检查
```

---

## 端点详情

### 1. POST /api/ticket/new

**说明**: 接收玩家日语邮件，翻译 + 分类 + RAG 检索 + 生成中文草稿。

**请求体**:
```json
{
  "question": "ゲームが起動しません。iPhoneを使用しています。"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 玩家日语邮件原文 |

**响应体** `200 OK`:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "question_zh": "游戏无法启动。我使用的是iPhone。",
  "intent": "不具合",
  "draft_zh": "感谢您的来信，给您带来不便深感抱歉。请尝试重启应用...",
  "similar_cases": [
    {
      "mail_id": "1234",
      "question": "アプリが起動しない",
      "answer": "アプリを再起動してください。",
      "inquiry_type": "不具合",
      "fused_score": 0.85
    }
  ],
  "confidence": "normal",
  "created_at": "2026-04-15T10:00:00Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | UUID，后续所有接口必须携带 |
| `question_zh` | string | 玩家问题中文译文（给客服看） |
| `intent` | string | 意图分类：`不具合` / `購入問題` / `意見建議` / `データ継承` / `未知` |
| `draft_zh` | string | AI 生成的中文回复草稿（客服可直接用或编辑） |
| `similar_cases` | array | RAG 检索到的相似历史工单，top-3 |
| `confidence` | string | `high`：高置信直接匹配（cosine > 0.92），`normal`：LLM 生成 |
| `created_at` | string | ISO 8601 时间戳 |

**错误**:
| 状态码 | 原因 |
|--------|------|
| `422` | 请求体缺少 `question` 字段 |

---

### 2. POST /api/ticket/finalize

**说明**: 客服确认（或编辑）中文草稿后，将其翻译为日语敬语体并作为最终回复。

**请求体**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "draft_zh": "非常感谢您的联系。关于您反映的问题，请先重启应用尝试解决..."
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | `new` 接口返回的 session_id |
| `draft_zh` | string | 是 | 客服确认/编辑后的中文草稿 |

**响应体** `200 OK`:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "reply_ja": "この度はご連絡いただきありがとうございます。ご報告いただいた問題につきまして...",
  "finalized_at": "2026-04-15T10:05:00Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `reply_ja` | string | 最终日语敬语回复（直接发给玩家） |
| `finalized_at` | string | ISO 8601 时间戳 |

会话状态从 `DRAFT_PENDING` 变为 `AWAITING_PLAYER`。

**错误**:
| 状态码 | 原因 |
|--------|------|
| `400` | 会话状态不是 `DRAFT_PENDING`（如重复确认） |
| `404` | session_id 不存在或已过期（TTL 24h） |
| `422` | 请求体字段缺失 |

---

### 3. POST /api/ticket/followup

**说明**: 玩家追加消息，在已有对话上下文中重新检索并生成新草稿。

**请求体**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "再起動しましたが、まだ起動しません。"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | session_id |
| `message` | string | 是 | 玩家追加的日语消息 |

**响应体** `200 OK`:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "followup_zh": "重启后依然无法启动。",
  "followup_type": "問題未解決",
  "intent": "不具合",
  "draft_zh": "非常抱歉问题仍未解决，建议您尝试重新安装应用...",
  "similar_cases": [...],
  "confidence": "normal"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `followup_zh` | string | 追加消息中文译文 |
| `followup_type` | string | `補充情報` / `問題未解決` / `新問題` / `確認` |
| `intent` | string | 沿用初始工单意图 |
| `draft_zh` | string | 新生成的中文草稿 |
| `similar_cases` | array | 基于初始问题+追加消息联合检索的相似案例 |
| `confidence` | string | 同 `new` 接口 |

完成后会话状态回到 `DRAFT_PENDING`，需再次调用 `finalize` 确认。

**错误**:
| 状态码 | 原因 |
|--------|------|
| `404` | session_id 不存在或已过期 |
| `422` | 请求体字段缺失 |

---

### 4. GET /api/session/{session_id}

**说明**: 获取完整会话信息和对话历史。

**路径参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | session UUID |

**响应体** `200 OK`:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "intent": "不具合",
  "state": "AWAITING_PLAYER",
  "turns": [
    {
      "role": "player",
      "content_zh": "游戏无法启动。",
      "timestamp": "2026-04-15T10:00:00Z"
    },
    {
      "role": "staff",
      "content_zh": "感谢您的联系，请重启应用。",
      "timestamp": "2026-04-15T10:05:00Z"
    }
  ],
  "created_at": "2026-04-15T10:00:00Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `state` | string | `DRAFT_PENDING`：等待客服确认 / `AWAITING_PLAYER`：等待玩家回复 |
| `turns` | array | 对话历史，按时间顺序，`role` 为 `player` 或 `staff` |

**错误**:
| 状态码 | 原因 |
|--------|------|
| `404` | session 不存在或已过期 |

---

### 5. GET /api/health

**说明**: 健康检查。

**响应体** `200 OK`:
```json
{
  "status": "ok",
  "model": "moonshot-v1-8k",
  "index_loaded": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `index_loaded` | bool | FAISS 索引是否已加载，`false` 时检索功能不可用 |

---

## 状态机

```
                  ┌─────────────────────────────┐
                  │          new ticket          │
                  └──────────────┬──────────────┘
                                 ▼
                         DRAFT_PENDING
                         (等待客服确认)
                                 │  finalize
                                 ▼
                        AWAITING_PLAYER
                         (等待玩家回复)
                                 │  followup
                                 ▼
                         DRAFT_PENDING  ──── finalize ──→ AWAITING_PLAYER
                              ↑__________________________|
```

---

## 错误格式

所有错误均返回标准 FastAPI 格式：

```json
{
  "detail": "会话不存在或已过期"
}
```

---

## 运行命令

```bash
# 1. 构建 FAISS 索引（首次必须执行）
uv run python -m codes.vector_store

# 2. 启动服务
uv run uvicorn codes.service.app:app --reload --port 8000

# 3. 跑接口测试（无需索引，全 mock）
uv run pytest tests/test_api.py -v

# 4. 跑评测（需要服务运行 + 索引）
uv run python -m codes.evaluation.evaluate
```
