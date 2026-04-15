"""
数据预处理
- 问题解析: 提取核心问题内容（去掉表单字段头）
- 回复清洗: 去 > 引用、※本メール签名、时间戳分隔的历史记录
- 多轮拆分: 按时间戳分割 answer，生成独立 QA 对
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from codes.config import DATASET_DIR

# ============================================================
# 正则
# ============================================================

# 表单字段前缀（question 里的结构化头部）
_FORM_FIELDS = re.compile(
    r"^(お問い合わせ\n|アカウントID\s*:.*\n|プレイヤー名\s*:.*\n|"
    r"アプリバージョン\s*:.*\n|ご利用のOSバージョン\s*:.*\n|"
    r"ご利用端末名\s*:.*\n|お問い合わせ内容の種類\s*:.*\n|"
    r"ご利用環境\s*:.*\n|問題が発生した日時\s*:.*\n|"
    r"メールアドレス\s*:.*\n|添付ファイル\s*:.*\n?)+",
    re.MULTILINE,
)

# 引用行（> 开头）
_QUOTE_LINE = re.compile(r"^>.*$", re.MULTILINE)

# 签名行（※本メール）
_SIGNATURE = re.compile(r"※本メール.*$", re.DOTALL)

# 日期时间戳分隔符（多轮对话的时间戳行）
_TIMESTAMP = re.compile(
    r"(\d{4}年\d{1,2}月\d{1,2}日.{0,30}?【.*?サポート】|"
    r"\d{4}/\d{1,2}/\d{1,2}\s+\d{2}:\d{2}:\d{2})"
)

# CS 回复特征开头
_CS_HEADER = re.compile(r"いつもご利用いただきありがとうございます")


# ============================================================
# 问题解析
# ============================================================

def parse_question(raw: str) -> str:
    """提取核心问题内容，去掉表单字段头"""
    # 找 お問い合わせ内容 ：之后的正文
    match = re.search(r"お問い合わせ内容\s*[：:]\s*([\s\S]+?)(?:添付ファイル|$)", raw)
    if match:
        content = match.group(1).strip()
        # 去掉附件行
        content = re.sub(r"添付ファイル.*$", "", content, flags=re.DOTALL).strip()
        return content if content else raw.strip()
    # fallback: 去掉结构化字段，保留剩余文本
    cleaned = _FORM_FIELDS.sub("", raw).strip()
    return cleaned if cleaned else raw.strip()


# ============================================================
# 回复清洗
# ============================================================

def clean_answer(raw: str) -> str:
    """清洗 CS 回复：去签名、去引用行、取最新一轮回复"""
    # 先按时间戳拆分，取第一段（最新 CS 回复）
    parts = _TIMESTAMP.split(raw)
    latest = parts[0].strip() if parts else raw

    # 去签名
    latest = _SIGNATURE.sub("", latest).strip()

    # 去引用行
    latest = _QUOTE_LINE.sub("", latest)

    # 合并多余空行
    latest = re.sub(r"\n{3,}", "\n\n", latest).strip()

    return latest


# ============================================================
# 多轮拆分
# ============================================================

@dataclass
class QAPair:
    mail_id: str | int
    round_number: int
    question: str
    answer: str
    inquiry_type: str
    is_multiturn: bool = False


def split_multiturn(row: pd.Series) -> list[QAPair]:
    """
    将多轮 answer 拆成独立 QA 对
    返回至少一条 QAPair（单轮也走这里统一处理）
    """
    raw_answer = str(row["answer"])
    mail_id = row["mail_id"]
    inquiry_type = str(row.get("inquiry_type", ""))
    question_raw = str(row["question"])

    # 按时间戳分割
    segments = _TIMESTAMP.split(raw_answer)

    # segments 结构: [文本, 时间戳, 文本, 时间戳, ...]
    # 提取纯文本段（偶数索引）
    text_segments = [segments[i].strip() for i in range(0, len(segments), 2) if segments[i].strip()]

    if len(text_segments) <= 1:
        # 单轮
        return [QAPair(
            mail_id=mail_id,
            round_number=0,
            question=parse_question(question_raw),
            answer=clean_answer(raw_answer),
            inquiry_type=inquiry_type,
            is_multiturn=False,
        )]

    # 多轮: text_segments[0] = 最新CS回复, [1] = 玩家追加消息 或 上一轮CS回复, ...
    pairs: list[QAPair] = []

    # Round 0: 原始问题 + 倒数第一段CS回复（最老的那轮）
    oldest_cs = text_segments[-1] if _CS_HEADER.search(text_segments[-1]) else text_segments[0]
    pairs.append(QAPair(
        mail_id=mail_id,
        round_number=0,
        question=parse_question(question_raw),
        answer=clean_answer(oldest_cs),
        inquiry_type=inquiry_type,
        is_multiturn=True,
    ))

    # 中间追加轮次（玩家消息 + 对应CS回复）
    # 从后往前配对
    i = len(text_segments) - 2
    round_num = 1
    while i >= 1:
        player_msg = text_segments[i].strip()
        cs_reply = text_segments[i - 1].strip() if i - 1 >= 0 else ""

        # 跳过 CS 自己的引用段
        if _CS_HEADER.search(player_msg):
            i -= 1
            continue

        if player_msg and cs_reply:
            pairs.append(QAPair(
                mail_id=mail_id,
                round_number=round_num,
                question=player_msg,
                answer=clean_answer(cs_reply),
                inquiry_type=inquiry_type,
                is_multiturn=True,
            ))
            round_num += 1
        i -= 2

    return pairs


# ============================================================
# 全量预处理（返回展开后的 DataFrame）
# ============================================================

def preprocess_dataset(csv_path: str | None = None) -> pd.DataFrame:
    path = csv_path or str(DATASET_DIR / "train.csv")
    df = pd.read_csv(path)

    all_pairs: list[dict] = []
    for _, row in df.iterrows():
        for pair in split_multiturn(row):
            all_pairs.append({
                "mail_id": pair.mail_id,
                "round_number": pair.round_number,
                "question": pair.question,
                "answer": pair.answer,
                "inquiry_type": pair.inquiry_type,
                "is_multiturn": pair.is_multiturn,
            })

    result = pd.DataFrame(all_pairs)
    print(f"原始: {len(df)} 条 → 展开: {len(result)} 条 QA 对")
    return result


if __name__ == "__main__":
    df = preprocess_dataset()
    print(df.head(3).to_string())
