"""
所有 Prompt 模板
4 个节点：日→中翻译、意图分类、中文草稿生成(RAG)、中→日敬语翻译
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
