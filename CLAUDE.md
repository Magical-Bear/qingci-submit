# ぽちゃガチョ！ 日本游戏智能客服系统

## 项目背景

游戏「ぽちゃガチョ！」在日本发行，中国客服团队为日本玩家提供服务。系统解决语言障碍，实现 RAG + Kimi API 的轻量 Web 应用。

## 对话流程

```
日语玩家来信
  → [自动解析结构化字段 + 翻译给客服看]
  → [AI 分类 + 检索相似案例 + 推荐中文处理方案]
  → [客服用中文确认/编辑回复]
  → [自动翻译成日语礼貌体发出]
```

## 技术栈

- **语言**: Python 3.11, 包管理 `uv`
- **Embedding**: 本地 bge-m3 (`weights/bge-m3/`, 1024 维)
- **向量库**: FAISS (IndexFlatIP, ~1761 条)
- **检索**: BM25 + 向量混合 (0.3/0.7)
- **LLM**: Kimi API (OpenAI 兼容，.env 配置)
- **Web**: FastAPI + uvicorn
- **评测**: Rouge-1 + BLEU + 语义相似度

## 项目结构

```
codes/
├── __init__.py
├── config.py                 # 路径、常量、环境变量
├── data_preprocessing.py     # 问题解析、回复清洗
├── embedding.py              # bge-m3 加载 & 编码
├── vector_store.py           # FAISS 索引构建/保存/加载
├── retriever.py              # BM25 + 向量混合检索
├── llm.py                    # Kimi API 客户端
├── prompts.py                # 所有 prompt 模板
├── pipeline.py               # 全流程编排
├── service/
│   ├── __init__.py
│   ├── app.py                # FastAPI 应用 + 生命周期管理
│   ├── models.py             # Pydantic 请求/响应模型
│   ├── session.py            # UUID 会话管理器
│   └── routes.py             # API 路由
└── evaluation/
    ├── __init__.py
    ├── metrics.py            # Rouge-1, BLEU, 语义相似度
    ├── evaluate.py           # 评测运行器
    └── test_eval.py          # pytest 入口
data/
└── index/                    # FAISS 索引 + BM25 pickle
dataset/
├── train.csv                 # 训练集 1761 条
├── eval.csv                  # 测试集 441 条
└── src_data.xlsx             # 原始数据
weights/
└── bge-m3/                   # 本地模型权重 (不要读取内容)
.env                          # KIMI_API_KEY, KIMI_BASE_URL, KIMI_MODEL
```

## 数据说明

- **格式**: 4 列 CSV (mail_id, question, answer, inquiry_type)
- **语言**: 日语邮件格式，question 含结构化表单字段
- **类型分布**: 不具合(57%), 购买问题(23%), 意见建议(14%), 数据继承(4%), 未知(2%)
- **多轮占比**: ~16% 的工单包含多轮对话，全部打包在单条 answer 字段中
- **清洗要点**: 去掉 `>` 引用、`※本メール...` 签名、日期时间戳分隔的历史记录

## 实施步骤

### Phase 1: 基础层
1. `codes/config.py` — 路径常量、.env 加载
2. `codes/data_preprocessing.py` — 问题解析(提取核心问题内容)、回复清洗(去引用/签名/多轮历史)
3. `codes/embedding.py` — bge-m3 加载编码 (sentence-transformers, 单例模式)

### Phase 2: 检索层
4. `codes/vector_store.py` — FAISS IndexFlatIP, 只索引 question_content
5. `codes/retriever.py` — BM25(fugashi 分词) + 向量混合检索, top-3

### Phase 3: LLM 层
6. `codes/llm.py` — Kimi API 客户端 (openai SDK, 重试逻辑)
7. `codes/prompts.py` — 4 个 prompt: 日→中翻译、意图分类、中文草稿生成(RAG)、中→日敬语翻译

### Phase 4: 流程 & 服务
8. `codes/pipeline.py` — 全流程编排 (高置信直接匹配优化)
9. `codes/service/` — FastAPI 服务 (详见 API 端点和多轮对话部分)

### Phase 5: 评测
10. `codes/evaluation/` — pytest 驱动, 逐条评分, 输出 response.csv

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ticket/new` | 新工单 → session_id + 中文草稿 |
| POST | `/api/ticket/finalize` | 客服确认中文草稿 → 日语回复 |
| POST | `/api/ticket/followup` | 玩家追加消息(多轮) |
| GET | `/api/session/{id}` | 获取完整对话历史 |
| GET | `/api/health` | 健康检查 |

## 多轮对话处理策略

### 数据层面的多轮特征

训练数据中 ~16% 是多轮对话，全部打包在单条 answer 字段中。4 种典型场景：

| 场景 | 占比(估) | 模式 | 示例 |
|------|----------|------|------|
| **信息收集** | ~40% | CS 要求补充信息 → 玩家提供 → CS 处理 | 要求截图、交易 ID、账号信息 |
| **排障升级** | ~30% | CS 给方案 → 玩家说没用 → CS 给更深方案 | 重启无效 → 重装 → 人工处理 |
| **纠正补充** | ~15% | 玩家追加/更正自己的描述 | 发错附件、补充说明 |
| **确认验证** | ~15% | CS 操作后让玩家确认 | 补发道具后确认到账 |

### 知识库构建时的多轮处理

对于训练集中的多轮工单，数据预处理时拆分为**多条独立 QA 对**入库：

```
原始数据 (一个 mail_id):
  question: 玩家初始问题
  answer: [最新CS回复] + [玩家追加消息] + [上一轮CS回复] + [原始问题引用]

拆分为:
  Round 1: question=原始问题, answer=第一轮CS回复
  Round 2: question=玩家追加消息, answer=最新CS回复
```

**拆分策略**:
- 用日期时间戳正则 `r'(\d{4}年\d{1,2}月\d{1,2}日.*?【.*?サポート】|(\d{4}/\d{1,2}/\d{1,2}\s+\d{2}:\d{2}:\d{2}))` 分割 answer 字段
- 每段识别角色 (CS 回复 vs 玩家追加消息)
- 每轮生成独立 QA 对，question 字段为该轮的用户输入，answer 为对应的 CS 回复
- 保留 `mail_id` + `round_number` 标识来源

**好处**: 检索时不仅能匹配初始问题，也能匹配追问模式（"重启了还是不行" → 找到类似的排障升级案例）

### 运行时的多轮会话管理

#### Session 数据结构

```python
class ConversationSession:
    id: str                          # UUID
    created_at: datetime
    ttl: timedelta = 24h
    intent: str                      # 初始意图分类
    metadata: dict                   # 设备、OS、玩家信息等
    
    # 对话历史 (双语)
    turns: list[ConversationTurn]
    
    # 当前状态
    state: SessionState              # DRAFT_PENDING / AWAITING_PLAYER / RESOLVED

class ConversationTurn:
    role: str                        # "player" | "staff"
    content_ja: str                  # 日语原文/译文
    content_zh: str                  # 中文原文/译文
    timestamp: datetime
    turn_type: str                   # "initial" | "followup" | "info_request" | "resolution"
    rag_context: list[dict] | None   # 本轮检索到的参考案例
```

#### 多轮流程

```
第 1 轮 (新工单):
  POST /api/ticket/new  {question: "日语问题"}
  → 解析 + 翻译 + 分类 + RAG检索 + 生成草稿
  → 返回 {session_id, question_zh, intent, draft_zh, similar_cases}
  → state = DRAFT_PENDING

客服确认:
  POST /api/ticket/finalize  {session_id, draft_zh}
  → 翻译成日语敬语体
  → state = AWAITING_PLAYER

第 N 轮 (玩家追加消息):
  POST /api/ticket/followup  {session_id, message: "新的日语消息"}
  → 翻译追加消息
  → 判断追加类型 (补充信息 / 问题未解决 / 新问题)
  → 携带历史上下文重新检索 + 生成新草稿
  → state = DRAFT_PENDING
```

#### 追加消息的上下文感知

处理 followup 时，不是孤立处理新消息，而是构建完整上下文：

1. **检索增强**: 将初始问题 + 追加消息拼接后检索，提高召回率
2. **对话历史注入 Prompt**: 将完整对话历史（双语）传入 Kimi API 的 messages 数组
3. **追加类型自动判断**: 通过 LLM 判断追加消息属于哪种场景，调整回复策略：
   - 补充信息 → 感谢提供 + 基于新信息给出方案
   - 问题未解决 → 道歉 + 升级方案
   - 新问题 → 新一轮处理
4. **意图漂移检测**: 如果追加消息的意图与初始意图不同（如从 bug 报告变成购买投诉），更新 session 意图并重新检索

#### Kimi API 多轮 messages 构建

```python
def build_multiturn_messages(session: ConversationSession, new_draft_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 注入对话历史作为 user/assistant 交替消息
    for turn in session.turns:
        if turn.role == "player":
            messages.append({
                "role": "user",
                "content": f"[玩家消息] {turn.content_zh}\n[原文] {turn.content_ja}"
            })
        else:
            messages.append({
                "role": "assistant",
                "content": f"[已发送回复] {turn.content_zh}"
            })
    
    # 最新的生成请求
    messages.append({"role": "user", "content": new_draft_prompt})
    return messages
```

### 高频重复问题的快速通道

对于高度重复的问题类型，系统提供快速匹配通道：

- **条件**: 向量检索 top-1 cosine > 0.92 且 inquiry_type 一致
- **行为**: 跳过 LLM 生成，直接将历史回复翻译成中文作为草稿
- **标记**: 在返回结果中标记 `confidence: "high"`，告知客服这是直接匹配
- **适用**: 主要覆盖 App Store 障害、常见 bug 已知修复方案等

## 关键设计决策

1. **索引问题而非回答** — 用户提问方式相似，问题到问题的匹配更可靠
2. **BM25 0.3 / 向量 0.7** — 向量捕获语义相似，BM25 捕获游戏术语精确匹配
3. **多轮工单拆分入库** — 训练集中的多轮对话拆成独立 QA 对，增加知识库覆盖率
4. **bge-m3 跨语言** — 同一模型处理日语和中文的 embedding，支持跨语言检索
5. **高置信直接匹配** — cosine > 0.92 跳过 LLM，降低延迟和成本

## 依赖安装

```bash
# Core ML
uv add sentence-transformers torch faiss-cpu transformers
# Retrieval
uv add rank-bm25 fugashi unidic-lite
# LLM API
uv add openai python-dotenv
# Web
uv add fastapi uvicorn pydantic-settings
# Evaluation
uv add rouge-score nltk sacrebleu
# Testing
uv add pytest httpx
```

## 运行命令

```bash
# 构建索引
uv run python -m codes.vector_store

# 启动服务
uv run uvicorn codes.service.app:app --reload

# 运行评测
uv run pytest codes/evaluation/test_eval.py -v
```

## 评测方案

- 逐条读取 eval.csv → 调用 FastAPI → 获取生成回复 → 保存到 response.csv
- 三项指标: Rouge-1 F1, BLEU (fugashi 分词), 语义相似度 (bge-m3 cosine)
- response.csv 列: mail_id, question, ground_truth, generated_ja, draft_zh, intent_pred, intent_actual, rouge1_f1, bleu, semantic_sim

## 注意事项

- 不要读取 `weights/` 文件夹的内容，只引用路径
- 不要读取 `chat_history/` 文件夹
- 可以使用 train.csv 前 100 行做测试数据
- 翻译出的日语必须符合日本商务礼仪 (敬語、謝罪表現、定型文)
