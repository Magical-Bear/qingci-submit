# ぽちゃガチョ！ 日本游戏智能客服系统

游戏「ぽちゃガチョ！」在日本发行，中国客服团队为日本玩家提供服务。系统通过 RAG + LangGraph 多 Agent Handoff + Kimi API 解决语言障碍，实现全流程自动化辅助。

## 对话流程

```
日语玩家来信
  → [百度翻译 JA→ZH ‖ Kimi 意图分类（asyncio 并行，节省 ~25% 延迟）]
  → [bge-m3 稀疏 + 向量混合检索相似案例]
  → [LangGraph Handoff：按意图路由到专用 Agent 生成中文草稿]
  → [客服用中文确认/编辑回复]
  → [Kimi 翻译成日语敬语体发出（保留 LLM 保证精度）]
```

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言/包管理 | Python 3.11, `uv` |
| Embedding | 本地 bge-m3 (`weights/bge-m3/`, 1024 维稠密 + 稀疏) |
| 向量库 | FAISS IndexFlatIP (~1761 条) |
| 检索 | bge-m3 稀疏 + 向量混合 (0.3/0.7) |
| LLM | Kimi API (OpenAI 兼容，`.env` 配置) |
| 翻译 | 百度翻译 API（JA→ZH 高速）+ Kimi（ZH→JA 敬语，保精度） |
| 流程编排 | LangGraph StateGraph + Handoff 意图路由 |
| Web | FastAPI + uvicorn |
| 评测 | Rouge-1 + BLEU + 语义相似度 |

## 四项核心优化

### 1. 检索升级：BM25 → bge-m3 原生稀疏向量

**原方案**：BM25（`rank_bm25` + `fugashi` MeCab 分词）+ 稠密向量混合检索

**新方案**：bge-m3 内置 `sparse_linear.pt`（`Linear(1024,1)` + ReLU）生成 SPLADE 风格稀疏向量，与稠密向量融合（0.3/0.7）

```
稀疏相似度：sim(q, d) = Σ q[t] × d[t]   （t ∈ 共享 token）
```

| 对比项 | BM25 | bge-m3 稀疏 |
|--------|------|-------------|
| 分词依赖 | fugashi + MeCab + unidic-lite | 无（共用 tokenizer） |
| 权重来源 | 词频统计（TF-IDF 变体） | 神经网络投影（语义感知） |
| 跨语言 | 需分语言分别处理 | 同一模型，天然支持 |
| 索引文件 | `bm25.pkl`（~676 KB） | `sparse_vectors.pkl`（~1.2 MB） |

**线程安全**：稠密和稀疏编码共享同一 Rust tokenizer，并发访问会触发 `RuntimeError: Already borrowed`。通过 `encode_both()` 将两者合并到单个 `asyncio.to_thread` 中顺序执行解决。

---

### 2. 延迟优化：翻译与分类并行 + 翻译分工

**原方案**（串行）：
```
JA→ZH 翻译（LLM）→ 意图分类（LLM）   ← 两次 LLM 顺序调用
```

**新方案**（并行 + 分工）：
```python
# node_translate_and_classify 中并行执行
question_zh, intent_raw = await asyncio.gather(
    baidu_translate.translate_ja_to_zh(question_ja),   # 百度 API，低延迟
    llm.chat(prompts.classify_intent(question_ja)),     # Kimi，直接分类日语原文
)
```

| 改进点 | 效果 |
|--------|------|
| JA→ZH 改用百度翻译 API | 响应 <200 ms，比 LLM 调用快 3–5× |
| 翻译与分类 asyncio 并行 | 消除串行等待，节省约 25% 首轮接口延迟 |
| ZH→JA 保留 Kimi | 敬语体、道歉表现、定型文精度不降级 |

---

### 3. LangGraph Handoff：意图路由多 Agent

**原方案**：单一 `generate_draft_zh` prompt，对所有意图使用相同系统提示

**新方案**：LangGraph `pipeline.py` 中按 `intent` 状态值路由到专用 Agent（Handoff 模式），每个 Agent 持有针对该场景深度优化的系统提示：

```python
# pipeline.py — 意图路由调度表
_DRAFT_PROMPT_MAP = {
    "故障问题":  prompts.generate_draft_zh_fault,
    "购买问题":  prompts.generate_draft_zh_purchase,
    "意见建议":  prompts.generate_draft_zh_feedback,
    "数据继承":  prompts.generate_draft_zh_data_migration,
    "未知":      prompts.generate_draft_zh_unknown,
}

# node_generate_draft 内路由
prompt_fn = _DRAFT_PROMPT_MAP.get(intent, prompts.generate_draft_zh_unknown)
```

| 意图 | Agent | 专注策略 |
|------|-------|---------|
| 故障问题 | `generate_draft_zh_fault` | 道歉 + 排查步骤（重启/清缓存/重装）+ 设备信息收集 + 修复时间预期 |
| 购买问题 | `generate_draft_zh_purchase` | 核查订单 + 补偿/退款政策 + 明确时间节点承诺 |
| 意见建议 | `generate_draft_zh_feedback` | 热情致谢 + 概括建议要点 + 转达产品团队（不承诺实现） |
| 数据继承 | `generate_draft_zh_data_migration` | 数据安全保障 + 迁移码步骤 + 账号绑定说明 |
| 未知 | `generate_draft_zh_unknown` | 通用专业回复 + 礼貌追问详情 |

---

### 4. API 限流韧性：指数退避重试

**原方案**：Kimi API 遇到 429 直接报错，高并发时请求失败

**新方案**：`llm.py` 通过 `tenacity` 对 `RateLimitError` 实现指数退避，最多重试 3 次：

```python
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
async def chat(messages, max_tokens=2048): ...
```

| 改进点 | 效果 |
|--------|------|
| 捕获 429 RateLimitError | 不再直接抛出 500 错误 |
| 指数退避（2s → 4s → 8s…，上限 30s） | 避免同时重试加剧限流 |
| 最多重试 3 次 | 暂时性限流自动恢复，长期过载快速失败 |



## 环境配置

`.env` 文件需包含：

```
KIMI_API_KEY=...
KIMI_BASE_URL=...
KIMI_MODEL=kimi-k2.5
BAIDU_TRANSLATE_APP_ID=...
BAIDU_TRANSLATE_API_KEY=...
```

## 依赖安装

```bash
# Core ML
uv add sentence-transformers torch faiss-cpu transformers
# LLM API
uv add openai python-dotenv
# Web
uv add fastapi uvicorn pydantic-settings
# LangGraph
uv add langchain langchain-openai langgraph
# Evaluation
uv add rouge-score nltk sacrebleu
# Testing
uv add pytest httpx
```

## 运行命令

```bash
# 构建索引（首次或 --force 强制重建）
uv run python -m codes.vector_store
uv run python -m codes.vector_store --force

# 启动服务
uv run uvicorn codes.service.app:app --reload

# 运行评测
uv run pytest codes/evaluation/test_eval.py -v
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ticket/new` | 新工单 → session_id + 中文草稿 |
| POST | `/api/ticket/finalize` | 客服确认中文草稿 → 日语回复 |
| POST | `/api/ticket/followup` | 玩家追加消息（多轮） |
| GET | `/api/session/{id}` | 获取完整对话历史 |
| GET | `/api/health` | 健康检查 |

### 示例请求

```bash
# 新工单
curl -X POST http://localhost:8000/api/ticket/new \
  -H "Content-Type: application/json" \
  -d '{"question": "アプリが起動しません。iPhone 14、iOS 17です。"}'

# 客服确认草稿
curl -X POST http://localhost:8000/api/ticket/finalize \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<uuid>", "draft_zh": "您好，感谢您的反馈..."}'

# 玩家追加消息
curl -X POST http://localhost:8000/api/ticket/followup \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<uuid>", "message": "再起動しても直りません。"}'
```

## 高置信快速通道

对高度重复的问题跳过 LLM 生成，直接翻译历史回复：

- **条件**: 向量检索 top-1 cosine > 0.92 且 `inquiry_type` 一致
- **行为**: 百度翻译历史 answer → 中文草稿（无 LLM 调用）
- **标记**: 响应中 `confidence: "high"`

## 多轮对话

- 会话由 UUID `session_id` 标识，TTL 24 小时
- 每轮对话历史以 `{role, content_zh}` 追加到 session，注入到后续 LLM prompt
- followup 消息自动分类为：补充信息 / 问题未解决 / 新问题 / 确认

## 关键设计决策

1. **索引问题而非回答** — 玩家提问方式相似，问-问匹配比问-答匹配更可靠
2. **bge-m3 稀疏替代 BM25** — 同一模型同时提供稠密语义向量和稀疏词法向量，无需额外分词依赖
3. **encode_both 串行编码** — 避免 Rust tokenizer 并发借用冲突
4. **意图路由多 Agent** — 不同工单类型差异显著，专用提示词比通用提示词效果更好
5. **多轮工单拆分入库** — 训练集中的多轮对话拆成独立 QA 对，提升检索覆盖率
6. **bge-m3 跨语言** — 同一模型处理日语和中文 embedding，支持跨语言检索

## 评测方案

逐条读取 `eval.csv` → 调用 FastAPI → 获取生成回复 → 计算三项指标

| 指标 | 说明 |
|------|------|
| Rouge-1 F1 | 词级别召回，衡量关键词覆盖 |
| BLEU（fugashi 分词） | n-gram 精确度，衡量局部短语匹配 |
| 语义相似度 | bge-m3 cosine，衡量语义层面的对齐 |

### 运行评测（run_eval）

> 前提：服务已启动（`uv run uvicorn codes.service.app:app --reload`）

```bash
uv run python codes/evaluation/run_eval.py
```

可配置参数（直接编辑 `run_eval.py` 顶部常量）：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `extract_nums` | `10` | 从 `eval.csv` 随机抽取的评测条数 |
| `submit_nums` | `4` | 送给大模型做质量总结的最高/最低分样本数 |
| `RANDOM_SEED` | `42` | 随机种子，保证复现 |
| `CONCURRENCY` | `3` | 并发请求数 |

报告输出至 `report/evals/eval_<timestamp>.html` 和 `.json`，HTML 包含：
- **整体指标卡片**：Rouge-1、BLEU、语义相似度、各接口平均延迟（`/new` / `/followup` / `/finalize` 分别统计）
- **大模型质量评估**：Kimi 对高分/低分样本的中文分析
- **详细结果表格**：每轮对话可展开查看原文、中文草稿、生成回复与标准回复对比

### 接口测试（test）

单元层面通过 pytest 验证指标计算：

```bash
uv run pytest codes/evaluation/ -v
```

端到端接口测试（服务需运行中）：

```bash
# 健康检查
curl http://localhost:8000/api/health

# 新工单（返回 session_id + 中文草稿）
curl -X POST http://localhost:8000/api/ticket/new \
  -H "Content-Type: application/json" \
  -d '{"question": "アプリが起動しません。iPhone 14、iOS 17です。"}'

# 客服确认草稿（返回日语正式回复）
curl -X POST http://localhost:8000/api/ticket/finalize \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<uuid>", "draft_zh": "您好，感谢您的反馈，我们正在调查..."}'

# 玩家追加消息（多轮）
curl -X POST http://localhost:8000/api/ticket/followup \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<uuid>", "message": "再起動しても直りません。"}'

# 查看完整会话历史
curl http://localhost:8000/api/session/<uuid>
```

## 项目思维链

详见 [logical_reasoning.md](logical_reasoning.md)

