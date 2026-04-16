"""
所有 Prompt 模板
4 个节点：日→中翻译、意图分类、中文草稿生成(RAG)、中→日敬语翻译
多智能体：按意图类型路由到专用草稿生成 Agent
"""

SYSTEM_CS = """\
あなたは日本のゲーム「ぽちゃガチョ！」のカスタマーサポートシステムです。
中国語サポートチームが日本人プレイヤーにサービスを提供しています。
"""


# ---------- 1. 日→中翻译 ----------

def translate_ja_to_zh(text_ja: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "你是专业的日中翻译，专注于游戏客服领域。"
                "请将以下日语邮件翻译成中文，保持原文结构，不添加任何解释。"
            ),
        },
        {"role": "user", "content": text_ja},
    ]


# ---------- 2. 意图分类 ----------

INTENT_LABELS = ["不具合", "購入問題", "意見建議", "データ継承", "未知"]

def classify_intent(text_ja: str) -> list[dict]:
    labels = "・".join(INTENT_LABELS)
    return [
        {
            "role": "system",
            "content": (
                f"あなたはゲームカスタマーサポートの問い合わせ分類の専門家です。"
                f"プレイヤーの問い合わせを次のいずれかに分類してください：{labels}。"
                "分類結果のみを出力し、説明は不要です。"
            ),
        },
        {"role": "user", "content": f"プレイヤーの問い合わせ：\n{text_ja}"},
    ]


# ---------- 3. 中文草稿生成(RAG) ----------

def generate_draft_zh(
    question_zh: str,
    intent: str,
    similar_cases: list[dict],
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """
    生成中文回复草稿
    conversation_history: [{"role": "player"|"staff", "content_zh": str}, ...]
    """
    # 构建参考案例文本
    cases_text = ""
    for i, case in enumerate(similar_cases, 1):
        cases_text += (
            f"\n【参考案例 {i}】(相似度: {case.get('fused_score', 0):.2f})\n"
            f"问题: {case['question']}\n"
            f"回复: {case['answer']}\n"
        )

    system_prompt = (
        "你是专业的游戏客服助手（ぽちゃガチョ！），帮助中国客服团队起草中文回复草稿。\n"
        "要求：\n"
        "1. 参考提供的历史案例，但不要照抄\n"
        "2. 回复友好、专业，符合客服语气\n"
        "3. 只输出回复草稿正文，不要加标题或说明\n"
        f"4. 本次工单意图类型：{intent}"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # 注入多轮历史（如有）
    if conversation_history:
        for turn in conversation_history:
            role = "user" if turn["role"] == "player" else "assistant"
            messages.append({"role": role, "content": turn["content_zh"]})

    messages.append({
        "role": "user",
        "content": (
            f"玩家问题（中文）：\n{question_zh}\n\n"
            f"历史参考案例：{cases_text}\n\n"
            "请生成中文回复草稿："
        ),
    })

    return messages


# ---------- 4. 中→日敬语翻译 ----------

def translate_zh_to_ja_polite(text_zh: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "あなたはプロの翻訳者です。中国語のカスタマーサポート返信を、"
                "丁寧で礼儀正しい日本語（敬語・謝罪表現・定型文を含む）に翻訳してください。"
                "翻訳のみ出力し、説明は不要です。"
            ),
        },
        {"role": "user", "content": text_zh},
    ]


# ---------- 5. 追加消息类型判断 ----------

FOLLOWUP_TYPES = ["補充情報", "問題未解決", "新問題", "確認"]

def classify_followup(followup_ja: str, history_summary: str) -> list[dict]:
    labels = "・".join(FOLLOWUP_TYPES)
    return [
        {
            "role": "system",
            "content": (
                f"あなたはカスタマーサポート対話分析の専門家です。"
                f"プレイヤーの追加メッセージのタイプを判定してください：{labels}。"
                "判定結果のみを出力し、説明は不要です。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"対話の背景：{history_summary}\n\n"
                f"プレイヤーの追加メッセージ：{followup_ja}"
            ),
        },
    ]


# ---------- 6. 多智能体：意图专用草稿生成 ----------

def _build_cases_text(similar_cases: list[dict]) -> str:
    """构建参考案例文本块（供各 Agent 复用）"""
    parts = []
    for i, case in enumerate(similar_cases, 1):
        parts.append(
            f"\n【参考案例 {i}】(相似度: {case.get('fused_score', 0):.2f})\n"
            f"问题: {case['question']}\n"
            f"回复: {case['answer']}\n"
        )
    return "".join(parts)


def _build_messages(
    system_prompt: str,
    question_zh: str,
    cases_text: str,
    conversation_history: list[dict] | None,
) -> list[dict]:
    """组装 messages 列表，注入多轮历史（供各 Agent 复用）"""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        for turn in conversation_history:
            role = "user" if turn["role"] == "player" else "assistant"
            messages.append({"role": role, "content": turn["content_zh"]})
    messages.append({
        "role": "user",
        "content": (
            f"玩家问题（中文）：\n{question_zh}\n\n"
            f"历史参考案例：{cases_text}\n\n"
            "请生成中文回复草稿："
        ),
    })
    return messages


def generate_draft_zh_fault(
    question_zh: str,
    intent: str,
    similar_cases: list[dict],
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """故障问题 Agent：聚焦排查步骤、设备信息收集、技术团队跟进"""
    system_prompt = (
        "你是专业的游戏客服助手（ぽちゃガチョ！），负责处理【故障问题】工单。\n"
        "处理策略：\n"
        "1. 首先向玩家道歉并感谢反馈\n"
        "2. 根据参考案例，提供具体的故障排查步骤（重启应用、清缓存、重装等）\n"
        "3. 如参考案例未涵盖，礼貌询问设备信息（OS版本、设备型号、应用版本号）\n"
        "4. 说明技术团队正在调查，给出合理预期时间\n"
        "5. 回复友好专业，控制在150字以内\n"
        "6. 只输出回复草稿正文，不要标题或说明\n"
        f"当前工单意图类型：{intent}"
    )
    return _build_messages(
        system_prompt, question_zh,
        _build_cases_text(similar_cases), conversation_history,
    )


def generate_draft_zh_purchase(
    question_zh: str,
    intent: str,
    similar_cases: list[dict],
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """购买问题 Agent：聚焦购买核查、补偿政策、退款流程"""
    system_prompt = (
        "你是专业的游戏客服助手（ぽちゃガチョ！），负责处理【购买问题】工单。\n"
        "处理策略：\n"
        "1. 道歉并表示理解玩家的不便\n"
        "2. 告知将核查购买记录，如未提供请礼貌要求订单号或截图\n"
        "3. 参考案例中的补偿/处理方式，给出明确承诺和时间节点\n"
        "4. 说明退款或补发道具的流程和预计完成时间\n"
        "5. 回复专业且有保证感，控制在150字以内\n"
        "6. 只输出回复草稿正文，不要标题或说明\n"
        f"当前工单意图类型：{intent}"
    )
    return _build_messages(
        system_prompt, question_zh,
        _build_cases_text(similar_cases), conversation_history,
    )


def generate_draft_zh_feedback(
    question_zh: str,
    intent: str,
    similar_cases: list[dict],
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """意见建议 Agent：聚焦热情致谢、概括建议要点、转达产品团队"""
    system_prompt = (
        "你是专业的游戏客服助手（ぽちゃガチョ！），负责处理【意见建议】工单。\n"
        "处理策略：\n"
        "1. 热情感谢玩家的宝贵反馈，表示重视\n"
        "2. 简要概括玩家建议的核心要点，体现已理解\n"
        "3. 明确告知将转达给产品/开发团队参考\n"
        "4. 不承诺具体功能实现或上线时间节点\n"
        "5. 鼓励继续提供反馈，回复温暖友好，控制在120字以内\n"
        "6. 只输出回复草稿正文，不要标题或说明\n"
        f"当前工单意图类型：{intent}"
    )
    return _build_messages(
        system_prompt, question_zh,
        _build_cases_text(similar_cases), conversation_history,
    )


def generate_draft_zh_data_migration(
    question_zh: str,
    intent: str,
    similar_cases: list[dict],
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """数据继承 Agent：聚焦数据安全保障、迁移步骤说明、账号信息收集"""
    system_prompt = (
        "你是专业的游戏客服助手（ぽちゃガチョ！），负责处理【数据继承/账号迁移】工单。\n"
        "处理策略：\n"
        "1. 安抚玩家，明确强调数据安全，不会丢失\n"
        "2. 提供清晰的数据继承步骤（参考历史案例）：获取继承码 → 新设备输入\n"
        "3. 说明支持的账号绑定方式（邮箱、Google、Apple ID等）\n"
        "4. 提醒操作前先绑定账号或备份继承码\n"
        "5. 如问题仍未解决，请玩家提供账号ID和操作步骤截图\n"
        "6. 回复有条理、有安全感，控制在180字以内\n"
        "7. 只输出回复草稿正文，不要标题或说明\n"
        f"当前工单意图类型：{intent}"
    )
    return _build_messages(
        system_prompt, question_zh,
        _build_cases_text(similar_cases), conversation_history,
    )


def generate_draft_zh_unknown(
    question_zh: str,
    intent: str,
    similar_cases: list[dict],
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """未知意图 Agent：通用专业回复，礼貌请求补充问题详情"""
    system_prompt = (
        "你是专业的游戏客服助手（ぽちゃガチョ！），帮助中国客服团队起草中文回复草稿。\n"
        "要求：\n"
        "1. 参考提供的历史案例，但不要照抄\n"
        "2. 回复友好、专业，符合客服语气\n"
        "3. 如意图不明，礼貌请求玩家补充问题详情\n"
        "4. 只输出回复草稿正文，不要加标题或说明\n"
        f"5. 本次工单意图类型：{intent}"
    )
    return _build_messages(
        system_prompt, question_zh,
        _build_cases_text(similar_cases), conversation_history,
    )
