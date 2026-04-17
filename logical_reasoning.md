# ぽちゃガチョ！游戏智能客服系统：技术方案与优化策略

## 摘要

针对日本游戏「ぽちゃガチョ！」的客服场景，本方案设计了一套基于 RAG（Retrieval-Augmented Generation）与 Multi-Agent 架构的智能客服辅助系统。系统核心解决三大挑战：**中日语言障碍**、**多轮对话上下文管理**、**垂直领域专业回复生成**。通过引入 bge-m3 统一嵌入模型实现稠密-稀疏混合检索、LangGraph Handoff 模式实现意图路由、以及分层翻译策略（百度 API + 大模型），在保持回复质量的同时显著降低响应延迟。

---

## 1. 问题定义与场景分析

### 1.1 业务背景

游戏客服场景具有以下特征：
- **语言壁垒**：日本玩家使用日语提交工单，中国客服团队需要准确理解并专业回复
- **多轮交互**：约 16% 的工单涉及多轮对话（信息收集、排障升级、纠正补充等）
- **时效敏感**：玩家对响应速度有较高期待，尤其是故障类问题
- **礼仪要求**：日语回复需符合商务敬语规范（敬語、謝罪表現、定型文）

### 1.2 核心挑战

| 挑战维度 | 具体问题 | 影响 |
|---------|---------|------|
| 检索精度 | 游戏术语、故障描述的语义匹配 | 召回相关案例的准确性 |
| 响应延迟 | 翻译 + 分类 + 检索 + 生成的串联耗时 | 用户体验 |
| 回复质量 | 垂直领域专业性、多轮上下文连贯性 | 客服满意度 |
| 系统扩展 | 新增问题类型的适配成本 | 维护效率 |

---

## 2. 系统架构设计

### 2.1 整体流程

```
日语玩家来信
  → [并行处理层] 百度翻译 JA→ZH ‖ Kimi 意图分类
  → [检索层] bge-m3 稠密 + 稀疏向量混合检索
  → [生成层] LangGraph Handoff → 专用 Agent 生成中文草稿
  → [人工确认层] 客服审阅/编辑中文回复
  → [输出层] Kimi 翻译成日语敬语体（统一前缀规范）
```

### 2.2 模块划分

| 模块 | 职责 | 关键技术 |
|------|------|---------|
| Translation Service | 中日双向翻译 | 百度翻译 API（JA→ZH）、Kimi LLM（ZH→JA） |
| Intent Classifier | 工单意图识别 | Kimi + 5 分类体系 |
| Retriever | 相似案例检索 | bge-m3 + FAISS |
| Agent Router | 意图路由调度 | LangGraph StateGraph |
| Draft Generator | 中文回复生成 | 意图专用 Prompt |
| Session Manager | 会话状态维护 | UUID + TTL + 状态机 |

---

## 3. 关键技术优化

### 3.1 检索层：bge-m3 统一稠密-稀疏表示

#### 3.1.1 方案演进

**传统方案（BM25 + 稠密向量）**：
- BM25 依赖 fugashi + MeCab 分词，需维护日语分词词典
- 稠密向量与稀疏向量来自不同模型，语义空间不对齐
- 跨语言检索需额外处理

**优化方案（bge-m3 原生稀疏向量）**：

bge-m3 模型内置 `sparse_linear.pt` 模块，通过 `Linear(1024, 1) + ReLU` 将稠密表示投影为 SPLADE 风格的稀疏向量：

```
稀疏相似度计算：sim(q, d) = Σ q[t] × d[t]  （t ∈ 共享 token）
```

#### 3.1.2 技术优势

| 特性 | BM25 | bge-m3 稀疏 |
|------|------|-------------|
| 分词依赖 | 需 fugashi + MeCab | 共用 tokenizer，无额外依赖 |
| 权重机制 | TF-IDF 统计 | 神经网络投影，语义感知 |
| 跨语言支持 | 需分语言处理 | 同一模型，天然支持 |
| 存储开销 | ~676 KB (bm25.pkl) | ~1.2 MB (sparse_vectors.pkl) |
| 语义一致性 | 低（统计方法） | 高（与稠密向量同源） |

#### 3.1.3 混合检索公式

最终相似度得分融合稠密与稀疏向量：

```
score = 0.7 × dense_cosine + 0.3 × sparse_dot_product
```

权重分配依据：稠密向量捕获语义相似性，稀疏向量强化术语精确匹配，7:3 比例在实践中取得最佳平衡。

#### 3.1.4 线程安全处理

bge-m3 的 Rust tokenizer 在并发访问时会触发 `RuntimeError: Already borrowed`。解决方案是将稠密与稀疏编码合并到单个 `asyncio.to_thread` 调用中顺序执行：

```python
def encode_both(texts):
    dense = model.encode(texts, convert_to_numpy=True)
    sparse = model.encode(texts, return_sparse=True)
    return dense, sparse
```

### 3.2 延迟优化：翻译与分类并行化

#### 3.2.1 流程重构

**原始串行流程**：
```
JA→ZH 翻译（LLM）→ 意图分类（LLM）  [两次 LLM 顺序调用]
```

**优化并行流程**：
```python
question_zh, intent_raw = await asyncio.gather(
    baidu_translate.translate_ja_to_zh(question_ja),  # 百度 API，< 200ms
    llm.chat(prompts.classify_intent(question_ja)),   # Kimi，直接分类日语原文
)
```

#### 3.2.2 分层翻译策略

| 方向 | 方案 | 理由 |
|------|------|------|
| JA → ZH | 百度翻译 API | 响应快（< 200ms），日常用语翻译质量可接受 |
| ZH → JA | Kimi LLM + 提示工程 | 敬语体、謝罪表現、定型文需要高精度 |

该分工在速度和精度之间取得平衡：理解阶段追求速度，输出阶段追求质量。

### 3.3 生成层：LangGraph Multi-Agent Handoff

#### 3.3.1 设计动机

单一通用 Prompt 难以覆盖所有工单类型的专业需求。故障问题需要排查步骤，购买问题需要政策说明，意见建议需要转达话术。通过 Handoff 模式，按意图路由到专用 Agent。

#### 3.3.2 意图分类体系

| 意图类别 | 占比（估） | 专用 Agent | 核心策略 |
|---------|-----------|-----------|---------|
| 故障问题 | 57% | `generate_draft_zh_fault` | 道歉 + 排查步骤 + 设备信息收集 |
| 购买问题 | 23% | `generate_draft_zh_purchase` | 核查订单 + 补偿政策 + 时间节点 |
| 意见建议 | 14% | `generate_draft_zh_feedback` | 致谢 + 概括要点 + 转达产品团队 |
| 数据继承 | 4% | `generate_draft_zh_data_migration` | 安全说明 + 迁移码步骤 |
| 未知 | 2% | `generate_draft_zh_unknown` | 通用回复 + 追问详情 |

#### 3.3.3 路由实现

```python
_DRAFT_PROMPT_MAP = {
    "故障问题": prompts.generate_draft_zh_fault,
    "购买问题": prompts.generate_draft_zh_purchase,
    "意见建议": prompts.generate_draft_zh_feedback,
    "数据继承": prompts.generate_draft_zh_data_migration,
    "未知": prompts.generate_draft_zh_unknown,
}

# node_generate_draft 内动态路由
prompt_fn = _DRAFT_PROMPT_MAP.get(intent, prompts.generate_draft_zh_unknown)
```

### 3.4 翻译质量优化：统一前缀规范

针对日语敬语翻译，测试发现增加统一前缀可显著提升输出稳定性：

```
【统一回复前缀】
この度は「ぽちゃガチョ！」をご利用いただき、
誠にありがとうございます。
カスタマーサポート窓口よりご連絡いたします。

[正文内容]

今後とも「ぽちゃガチョ！」をよろしくお願いいたします。
```

该前缀确保每封回复都以标准商务格式开头，符合日本玩家对游戏客服的期待。

### 3.5 领域适配：游戏术语表

针对游戏专属名词，构建中日互译术语表以提升翻译一致性：

| 中文 | 日文 | 备注 |
|------|------|------|
| 钻石 | ダイヤ | 付费货币 |
| 扭蛋 | ガチャ | 抽卡系统 |
| 体力 | スタミナ | 行动点数 |
| 限定活动 | 限定イベント | 时间限定内容 |
| 数据继承 | データ引き継ぎ | 账号迁移 |
| 客服 | カスタマーサポート | 官方支持 |

术语表在翻译阶段作为上下文注入，确保专业名词准确对应。

---

## 4. 多轮对话管理

### 4.1 会话状态机

```
new ticket → DRAFT_PENDING ──finalize──→ AWAITING_PLAYER
                  ↑                           │ followup
                  └───────────────────────────┘
```

### 4.2 上下文注入策略

每轮对话历史以双语形式存储，在生成新草稿时注入 Kimi 的 messages 数组：

```python
def build_multiturn_messages(session, new_draft_prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

    messages.append({"role": "user", "content": new_draft_prompt})
    return messages
```

### 4.3 Followup 类型自动判断

玩家追加消息自动分类为：
- **补充信息**：提供额外截图、交易 ID 等
- **问题未解决**：已尝试方案无效
- **新问题**：偏离原始意图的新诉求
- **确认**：对客服操作的确认回复

分类结果影响后续检索策略（是否重新检索）和回复策略（道歉升级 vs 感谢确认）。

---

## 5. 高置信快速通道

对于高度重复的问题，系统提供快速匹配通道：

**触发条件**：
- 向量检索 top-1 cosine > 0.92
- inquiry_type 与历史案例一致

**处理方式**：
- 跳过 LLM 生成
- 直接翻译历史 answer 作为中文草稿
- 标记 `confidence: "high"`

**适用场景**：App Store 障害公告、已知 bug 的固定修复方案等。

---

## 6. 评估体系

### 6.1 自动指标

| 指标 | 计算方法 | 评估维度 |
|------|---------|---------|
| Rouge-1 F1 | 词级别召回 | 关键词覆盖度 |
| BLEU | fugashi 分词后的 n-gram 精确度 | 局部短语匹配 |
| 语义相似度 | bge-m3 cosine | 语义层面一致性 |

### 6.2 LLM Judge

对评测结果中的高分/低分样本，提交给 Kimi 生成质量分析报告：

```python
# 选取最高分和最低分样本各 submit_nums 条
high_score_samples = df.nlargest(submit_nums, 'overall_score')
low_score_samples = df.nsmallest(submit_nums, 'overall_score')

# 请求大模型分析优劣原因
analysis = await llm_judge.analyze_samples(
    high_samples=high_score_samples,
    low_samples=low_score_samples
)
```

LLM Judge 从专业性、完整性、礼貌度等维度给出定性评估，补充自动指标的不足。

### 6.3 数据隔离

训练集（1761 条）与测试集（441 条）在数据预处理阶段即完成切分，防止数据泄露。评测时调用 FastAPI 接口，模拟真实生产链路。

---

## 7. 工程实现

### 7.1 API 设计

系统封装为标准 RESTful API：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/ticket/new` | POST | 新工单处理 |
| `/api/ticket/finalize` | POST | 客服确认草稿 |
| `/api/ticket/followup` | POST | 玩家追加消息 |
| `/api/session/{id}` | GET | 查询对话历史 |
| `/api/health` | GET | 健康检查 |

### 7.2 限流与容错

Kimi API 通过 `tenacity` 实现指数退避重试：

```python
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
)
async def chat(messages, max_tokens=2048): ...
```

### 7.3 运行环境

- **语言**: Python 3.11
- **包管理**: uv
- **Web 框架**: FastAPI + uvicorn
- **流程编排**: LangGraph

---

## 8. 关键设计决策总结

1. **索引问题而非回答**：玩家提问方式相似，问-问匹配比问-答匹配更可靠
2. **bge-m3 统一表示**：稠密 + 稀疏向量同源生成，避免 BM25 的分词依赖
3. **encode_both 串行编码**：规避 Rust tokenizer 并发冲突
4. **分层翻译策略**：速度敏感环节用 API，质量敏感环节用 LLM
5. **意图路由多 Agent**：垂直领域专用提示词优于通用提示词
6. **多轮工单拆分入库**：训练集中的多轮对话拆成独立 QA 对，提升检索覆盖率
7. **统一回复前缀**：保证日语输出格式规范

---

## 9. 后续优化方向

- **在线学习**：收集客服编辑反馈，持续优化检索排序
- **意图细分**：在当前 5 分类基础上进一步细分（如区分 iOS/Android 故障）
- **多模态支持**：处理玩家上传的截图附件
- **A/B 测试框架**：系统化的 Prompt 对比评测
- **DeepEval**: 基于大模型的自动化评测

