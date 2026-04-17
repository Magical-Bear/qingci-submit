#!/usr/bin/env python3
"""
评估脚本：从 eval.csv 随机抽取样本，评测多轮对话质量
生成 JSON 报告和 HTML 表格，并包含大模型质量评估
"""

import asyncio
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from codes.config import BASE_DIR, DATASET_DIR, settings
from codes.evaluation.metrics import compute_all
from codes.llm import chat

# ==================== 用户可配置常量 ====================
extract_nums = 20  # 从 eval.csv 抽取评测的条数
submit_nums = 4 # 给大模型评估的最好/最差翻译片段数量

# ==================== 其他常量 ====================
RANDOM_SEED = 42
CONCURRENCY = 2
BASE_URL = f"http://{settings.host}:{settings.port}"
REPORT_DIR = BASE_DIR / "report" / "evals"

# 多轮检测正则
TIMESTAMP_REGEX = re.compile(
    r'\d{4}年\d{1,2}月\d{1,2}日[（(][月火水木金土日][）)]\s+\d{1,2}:\d{2}'
)
CS_OPENER_REGEX = re.compile(r'^(いつもご利用|ご連絡|平素より|お世話になっております)')


def parse_thread_turns(question_ja: str, answer: str) -> list[dict]:
    """
    解析多轮对话的 answer 字段，返回最多 2 轮对话
    每轮包含：cs_reply (客服回复), player_message (玩家追加消息，可选)
    返回按时间顺序排列（老 → 新）
    """
    lines = answer.split('\n')

    segments = []
    current_segment_lines = []
    current_type = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if line_stripped.startswith('>'):
            line_type = 'quoted'
        elif TIMESTAMP_REGEX.search(line_stripped):
            line_type = 'timestamp'
        else:
            line_type = 'plain'

        if line_type != current_type and current_segment_lines:
            segments.append({
                'type': current_type,
                'content': '\n'.join(current_segment_lines).strip()
            })
            current_segment_lines = []

        current_type = line_type
        current_segment_lines.append(line_stripped)

    if current_segment_lines:
        segments.append({
            'type': current_type,
            'content': '\n'.join(current_segment_lines).strip()
        })

    plain_segments = [s for s in segments if s['type'] == 'plain']

    if len(plain_segments) == 1:
        content = plain_segments[0]['content']
        is_cs = bool(CS_OPENER_REGEX.search(content))
        if is_cs:
            return [{'cs_reply': content, 'player_message': None}]
        return []

    turns = []
    i = 0
    while i < len(plain_segments) - 1:
        current = plain_segments[i]['content']
        next_seg = plain_segments[i + 1]['content']

        current_is_cs = bool(CS_OPENER_REGEX.search(current))
        next_is_cs = bool(CS_OPENER_REGEX.search(next_seg))

        if current_is_cs and not next_is_cs:
            turns.append({'cs_reply': current, 'player_message': next_seg})
            i += 2
        elif current_is_cs and next_is_cs:
            turns.append({'cs_reply': current, 'player_message': None})
            i += 1
        else:
            i += 1

    if i < len(plain_segments):
        last = plain_segments[i]['content']
        if CS_OPENER_REGEX.search(last):
            turns.append({'cs_reply': last, 'player_message': None})

    if not turns and plain_segments:
        last = plain_segments[-1]['content']
        if CS_OPENER_REGEX.search(last):
            turns = [{'cs_reply': last, 'player_message': None}]

    if len(turns) > 2:
        turns = turns[-2:]

    turns.reverse()

    return turns


def color_score(score: float) -> str:
    """根据分数返回颜色类名"""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    return "low"


async def evaluate_one_row(
    session: aiohttp.ClientSession,
    row: pd.Series,
    sem: asyncio.Semaphore
) -> dict | None:
    """
    评估单行数据，支持多轮对话
    返回包含所有轮次评估结果的字典
    """
    mail_id = row['mail_id']
    question_ja = row['question']
    answer_ja = row['answer']
    inquiry_type = row.get('inquiry_type', 'unknown')

    turns = parse_thread_turns(question_ja, answer_ja)
    if not turns:
        print(f"[WARN] {mail_id}: 无法解析对话轮次")
        return None

    async with sem:
        try:
            round_results = []
            session_id = None

            for round_idx, turn in enumerate(turns):
                round_num = round_idx + 1
                cs_ground_truth = turn['cs_reply']
                player_followup = turn.get('player_message')

                endpoint_latencies: dict[str, float] = {}

                if round_idx == 0:
                    t0 = time.time()
                    async with session.post(
                        f"{BASE_URL}/api/ticket/new",
                        json={"question": question_ja}
                    ) as resp:
                        if resp.status != 200:
                            print(f"[ERROR] {mail_id} R{round_num}: /new failed {resp.status}")
                            return None
                        data = await resp.json()
                        session_id = data.get('session_id')
                        question_zh = data.get('question_zh', '')
                        draft_zh = data.get('draft_zh', '')
                    endpoint_latencies['new'] = time.time() - t0
                else:
                    if not session_id or not player_followup:
                        continue

                    t0 = time.time()
                    async with session.post(
                        f"{BASE_URL}/api/ticket/followup",
                        json={
                            "session_id": session_id,
                            "message": player_followup
                        }
                    ) as resp:
                        if resp.status != 200:
                            print(f"[ERROR] {mail_id} R{round_num}: /followup failed {resp.status}")
                            return None
                        data = await resp.json()
                        question_zh = data.get('followup_zh', '')
                        draft_zh = data.get('draft_zh', '')
                    endpoint_latencies['followup'] = time.time() - t0

                t0 = time.time()
                async with session.post(
                    f"{BASE_URL}/api/ticket/finalize",
                    json={"session_id": session_id, "draft_zh": draft_zh}
                ) as resp:
                    if resp.status != 200:
                        print(f"[ERROR] {mail_id} R{round_num}: /finalize failed {resp.status}")
                        return None
                    finalize_data = await resp.json()
                    reply_ja = finalize_data.get('reply_ja', '')
                endpoint_latencies['finalize'] = time.time() - t0

                latency = sum(endpoint_latencies.values())

                metrics = compute_all(reply_ja, cs_ground_truth)

                round_results.append({
                    'round': round_num,
                    'question_ja': question_ja if round_idx == 0 else player_followup,
                    'question_zh': question_zh,
                    'draft_zh': draft_zh,
                    'reply_ja_generated': reply_ja,
                    'reply_ja_ground_truth': cs_ground_truth,
                    'latency_ms': round(latency * 1000, 2),
                    'endpoint_latencies_ms': {k: round(v * 1000, 2) for k, v in endpoint_latencies.items()},
                    'rouge1_f1': round(metrics['rouge1_f1'], 4),
                    'bleu': round(metrics['bleu'], 4),
                    'semantic_sim': round(metrics['semantic_sim'], 4)
                })

            if not round_results:
                return None

            # Per-endpoint latency aggregation
            ep_buckets: dict[str, list[float]] = defaultdict(list)
            for r in round_results:
                for ep, ms in r['endpoint_latencies_ms'].items():
                    ep_buckets[ep].append(ms)
            avg_ep_latencies = {ep: round(sum(v) / len(v), 2) for ep, v in ep_buckets.items()}

            avg_metrics = {
                'avg_rouge1_f1': round(sum(r['rouge1_f1'] for r in round_results) / len(round_results), 4),
                'avg_bleu': round(sum(r['bleu'] for r in round_results) / len(round_results), 4),
                'avg_semantic_sim': round(sum(r['semantic_sim'] for r in round_results) / len(round_results), 4),
                'avg_latency_ms': round(sum(r['latency_ms'] for r in round_results) / len(round_results), 2),
                'avg_endpoint_latencies_ms': avg_ep_latencies,
            }

            return {
                'mail_id': mail_id,
                'inquiry_type': inquiry_type,
                'num_rounds': len(round_results),
                'rounds': round_results,
                **avg_metrics
            }

        except Exception as e:
            print(f"[ERROR] {mail_id}: {e}")
            return None


async def run_llm_assessment(
    results: list[dict],
    overall_avg: dict
) -> str:
    """
    使用 Kimi API 对评估结果进行质量总结
    发送得分最高和最低的 submit_nums 条样本
    """
    sorted_by_score = sorted(
        results,
        key=lambda x: x['avg_semantic_sim'],
        reverse=True
    )

    top_samples = sorted_by_score[:submit_nums]
    bottom_samples = sorted_by_score[-submit_nums:]

    prompt_lines = [
        "你是一位专业的日语客服翻译质量评估专家。",
        "",
        f"本次评估共 {len(results)} 条样本，平均指标如下：",
        f"- Rouge-1 F1: {overall_avg['avg_rouge1_f1']:.4f}",
        f"- BLEU: {overall_avg['avg_bleu']:.4f}",
        f"- 语义相似度: {overall_avg['avg_semantic_sim']:.4f}",
        f"- 平均总响应延迟: {overall_avg['avg_latency_ms']:.2f}ms",
    ]
    for ep, ms in overall_avg.get('avg_endpoint_latencies_ms', {}).items():
        prompt_lines.append(f"  - /{ep} 平均延迟: {ms:.2f}ms")
    prompt_lines.extend([
        "",
        "=" * 50,
        f"【得分最高的 {submit_nums} 条样本】",
        "=" * 50,
    ])

    for i, sample in enumerate(top_samples, 1):
        prompt_lines.extend([
            f"\n--- 样本 {i} (mail_id: {sample['mail_id']}) ---",
            f"问题类型: {sample['inquiry_type']}",
            f"对话轮数: {sample['num_rounds']}",
            f"平均语义相似度: {sample['avg_semantic_sim']:.4f}",
        ])
        for r in sample['rounds']:
            prompt_lines.extend([
                f"\n第 {r['round']} 轮:",
                f"  用户问题(日): {r['question_ja'][:100]}..." if len(r['question_ja']) > 100 else f"  用户问题(日): {r['question_ja']}",
                f"  中文草稿: {r['draft_zh'][:100]}..." if len(r['draft_zh']) > 100 else f"  中文草稿: {r['draft_zh']}",
                f"  系统日语回复: {r['reply_ja_generated'][:100]}..." if len(r['reply_ja_generated']) > 100 else f"  系统日语回复: {r['reply_ja_generated']}",
                f"  标准回复: {r['reply_ja_ground_truth'][:100]}..." if len(r['reply_ja_ground_truth']) > 100 else f"  标准回复: {r['reply_ja_ground_truth']}",
            ])

    prompt_lines.extend([
        "",
        "=" * 50,
        f"【得分最低的 {submit_nums} 条样本】",
        "=" * 50,
    ])

    for i, sample in enumerate(bottom_samples, 1):
        prompt_lines.extend([
            f"\n--- 样本 {i} (mail_id: {sample['mail_id']}) ---",
            f"问题类型: {sample['inquiry_type']}",
            f"对话轮数: {sample['num_rounds']}",
            f"平均语义相似度: {sample['avg_semantic_sim']:.4f}",
        ])
        for r in sample['rounds']:
            prompt_lines.extend([
                f"\n第 {r['round']} 轮:",
                f"  用户问题(日): {r['question_ja'][:100]}..." if len(r['question_ja']) > 100 else f"  用户问题(日): {r['question_ja']}",
                f"  中文草稿: {r['draft_zh'][:100]}..." if len(r['draft_zh']) > 100 else f"  中文草稿: {r['draft_zh']}",
                f"  系统日语回复: {r['reply_ja_generated'][:100]}..." if len(r['reply_ja_generated']) > 100 else f"  系统日语回复: {r['reply_ja_generated']}",
                f"  标准回复: {r['reply_ja_ground_truth'][:100]}..." if len(r['reply_ja_ground_truth']) > 100 else f"  标准回复: {r['reply_ja_ground_truth']}",
            ])

    prompt_lines.extend([
        "",
        "=" * 50,
        "请从以下几个方面对翻译质量进行中文总结：",
        "1. 整体质量评价（高得分样本的共同优点）",
        "2. 主要问题分析（低得分样本的共性问题）",
        "3. 改进建议（针对客服日语回复的准确性和礼貌性）",
        "4. 多轮对话处理效果评价",
        "",
        "请给出专业的评估意见。"
    ])

    prompt = '\n'.join(prompt_lines)

    try:
        summary = await chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )
        return summary
    except Exception as e:
        print(f"[ERROR] LLM assessment failed: {e}")
        return f"评估失败: {e}"


def _md_to_html(text: str) -> str:
    """Convert simple markdown (headers, bold, bullet lists) to HTML."""
    import html as html_lib
    lines = text.split('\n')
    out: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith('### '):
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append(f'<h4>{html_lib.escape(stripped[4:])}</h4>')
        elif stripped.startswith('## '):
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append(f'<h3>{html_lib.escape(stripped[3:])}</h3>')
        elif stripped.startswith('# '):
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append(f'<h2>{html_lib.escape(stripped[2:])}</h2>')
        elif stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                out.append('<ul>')
                in_list = True
            item = html_lib.escape(stripped[2:])
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
            out.append(f'<li>{item}</li>')
        elif stripped == '':
            if in_list:
                out.append('</ul>')
                in_list = False
        else:
            if in_list:
                out.append('</ul>')
                in_list = False
            para = html_lib.escape(stripped)
            para = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', para)
            out.append(f'<p>{para}</p>')
    if in_list:
        out.append('</ul>')
    return '\n'.join(out)


def generate_html(
    results: list[dict],
    overall_avg: dict,
    llm_summary: str,
    ts_str: str,
    metadata: dict
) -> str:
    """生成 HTML 报告"""

    ep_latencies = overall_avg.get('avg_endpoint_latencies_ms', {})
    ep_cards = ''.join(
        f"<div class='card'><h3>/{ep} 延迟</h3>"
        f"<div class='value'>{ms:.0f}<span class='unit'>ms</span></div></div>"
        for ep, ms in ep_latencies.items()
    )
    llm_html = _md_to_html(llm_summary)

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='zh-CN'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"    <title>客服系统评估报告 - {ts_str}</title>",
        "    <style>",
        "        * { box-sizing: border-box; margin: 0; padding: 0; }",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; padding: 20px; color: #333; }",
        "        .container { max-width: 1400px; margin: 0 auto; }",
        "        h1 { text-align: center; margin-bottom: 10px; color: #2c3e50; }",
        "        .meta { text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }",
        "        .section-title { margin: 0 0 15px; color: #2c3e50; font-size: 18px; border-left: 4px solid #3498db; padding-left: 10px; }",
        "        .summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 15px; margin-bottom: 30px; }",
        "        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }",
        "        .card h3 { font-size: 13px; color: #666; margin-bottom: 10px; }",
        "        .card .value { font-size: 26px; font-weight: bold; color: #2c3e50; }",
        "        .card .unit { font-size: 12px; color: #999; }",
        "        .llm-summary { background: white; padding: 24px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; line-height: 1.8; }",
        "        .llm-summary h2 { margin-bottom: 15px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
        "        .llm-summary h3 { margin: 16px 0 8px; color: #34495e; font-size: 15px; }",
        "        .llm-summary h4 { margin: 12px 0 6px; color: #555; font-size: 14px; }",
        "        .llm-summary p { margin: 6px 0; color: #444; font-size: 14px; }",
        "        .llm-summary ul { margin: 6px 0 6px 20px; }",
        "        .llm-summary li { color: #444; font-size: 14px; margin: 3px 0; }",
        "        table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }",
        "        th { background: #34495e; color: white; padding: 12px 8px; text-align: left; font-weight: 600; font-size: 13px; }",
        "        td { padding: 10px 8px; border-bottom: 1px solid #eee; vertical-align: top; font-size: 13px; }",
        "        tr:hover { background: #f8f9fa; }",
        "        .round-header { background: #ecf0f1; font-weight: bold; }",
        "        .round-sub { background: #fafafa; }",
        "        .mail-id { font-family: monospace; color: #3498db; font-size: 12px; }",
        "        .type-tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; background: #e74c3c; color: white; }",
        "        .ja-text { color: #2c3e50; max-width: 300px; overflow: hidden; text-overflow: ellipsis; }",
        "        .zh-text { color: #27ae60; max-width: 250px; overflow: hidden; text-overflow: ellipsis; }",
        "        .reply-text { color: #8e44ad; max-width: 300px; overflow: hidden; text-overflow: ellipsis; }",
        "        .score { font-weight: bold; font-family: monospace; }",
        "        .score.high { color: #27ae60; }",
        "        .score.medium { color: #f39c12; }",
        "        .score.low { color: #e74c3c; }",
        "        .latency { color: #7f8c8d; font-size: 12px; }",
        "        .round-badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; background: #95a5a6; color: white; margin-right: 5px; }",
        "        .expand-btn { cursor: pointer; color: #3498db; font-size: 11px; }",
        "        .full-text { display: none; margin-top: 5px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }",
        "        .full-text.show { display: block; }",
        "    </style>",
        "    <script>",
        "        function toggleText(id) {",
        "            const el = document.getElementById(id);",
        "            el.classList.toggle('show');",
        "        }",
        "    </script>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        f"        <h1>ぽちゃガチョ！客服系统评估报告</h1>",
        f"        <div class='meta'>生成时间: {metadata['timestamp']} | 样本数: {metadata['sample_count']} | 随机种子: {metadata['random_seed']}</div>",
        "",
        "        <h2 class='section-title'>整体指标</h2>",
        "        <div class='summary-cards'>",
        f"            <div class='card'><h3>Rouge-1 F1</h3><div class='value'>{overall_avg['avg_rouge1_f1']:.4f}</div></div>",
        f"            <div class='card'><h3>BLEU</h3><div class='value'>{overall_avg['avg_bleu']:.4f}</div></div>",
        f"            <div class='card'><h3>语义相似度</h3><div class='value'>{overall_avg['avg_semantic_sim']:.4f}</div></div>",
        f"            <div class='card'><h3>总平均延迟</h3><div class='value'>{overall_avg['avg_latency_ms']:.0f}<span class='unit'>ms</span></div></div>",
        f"            {ep_cards}",
        "        </div>",
        "",
        "        <div class='llm-summary'>",
        "            <h2>大模型质量评估</h2>",
        f"            {llm_html}",
        "        </div>",
        "",
        "        <h2 class='section-title'>详细评估结果</h2>",
        "        <table>",
        "            <thead>",
        "                <tr>",
        "                    <th>Mail ID</th>",
        "                    <th>类型</th>",
        "                    <th>用户问题 (JA)</th>",
        "                    <th>中文翻译</th>",
        "                    <th>系统回复 / 标准回复</th>",
        "                    <th>延迟</th>",
        "                    <th>Rouge-1</th>",
        "                    <th>BLEU</th>",
        "                    <th>语义相似</th>",
        "                </tr>",
        "            </thead>",
        "            <tbody>",
    ]

    for result in results:
        mail_id = result['mail_id']
        inquiry_type = result['inquiry_type']
        num_rounds = result['num_rounds']

        for round_idx, r in enumerate(result['rounds']):
            is_first = round_idx == 0
            row_class = "round-header" if is_first else "round-sub"

            ja_short = r['question_ja'][:50] + '...' if len(r['question_ja']) > 50 else r['question_ja']
            zh_short = r['draft_zh'][:40] + '...' if len(r['draft_zh']) > 40 else r['draft_zh']
            reply_short = r['reply_ja_generated'][:50] + '...' if len(r['reply_ja_generated']) > 50 else r['reply_ja_generated']

            ja_id = f"ja_{mail_id}_{r['round']}"
            zh_id = f"zh_{mail_id}_{r['round']}"
            reply_id = f"reply_{mail_id}_{r['round']}"

            html_parts.append(f"                <tr class='{row_class}'>")

            if is_first:
                html_parts.extend([
                    f"                    <td rowspan='{num_rounds}' class='mail-id'>{mail_id}</td>",
                    f"                    <td rowspan='{num_rounds}'><span class='type-tag'>{inquiry_type}</span></td>",
                ])

            html_parts.extend([
                f"                    <td>",
                f"                        <span class='round-badge'>R{r['round']}</span>",
                f"                        <div class='ja-text' title='{r['question_ja'].replace(chr(39), '&apos;')}'>{ja_short}</div>",
                f"                        <span class='expand-btn' onclick=\"toggleText('{ja_id}')\">[展开]</span>",
                f"                        <div id='{ja_id}' class='full-text'>{r['question_ja']}</div>",
                f"                    </td>",
                f"                    <td>",
                f"                        <div class='zh-text'>{zh_short}</div>",
                f"                        <span class='expand-btn' onclick=\"toggleText('{zh_id}')\">[展开]</span>",
                f"                        <div id='{zh_id}' class='full-text'>{r['draft_zh']}</div>",
                f"                    </td>",
                f"                    <td>",
                f"                        <div class='reply-text'><b>生成:</b> {reply_short}</div>",
                f"                        <span class='expand-btn' onclick=\"toggleText('{reply_id}')\">[展开]</span>",
                f"                        <div id='{reply_id}' class='full-text'><b>生成:</b> {r['reply_ja_generated']}\n\n<b>标准:</b> {r['reply_ja_ground_truth']}</div>",
                f"                    </td>",
                f"                    <td class='latency'>{r['latency_ms']:.0f}ms</td>",
                f"                    <td><span class='score {color_score(r['rouge1_f1'])}'>{r['rouge1_f1']:.4f}</span></td>",
                f"                    <td><span class='score {color_score(r['bleu'])}'>{r['bleu']:.4f}</span></td>",
                f"                    <td><span class='score {color_score(r['semantic_sim'])}'>{r['semantic_sim']:.4f}</span></td>",
                f"                </tr>",
            ])

    html_parts.extend([
        "            </tbody>",
        "        </table>",
        "    </div>",
        "</body>",
        "</html>"
    ])

    return '\n'.join(html_parts)


async def main():
    """主函数"""
    print("=" * 60)
    print("ぽちゃガチョ！客服系统评估脚本")
    print("=" * 60)
    print(f"配置: extract_nums={extract_nums}, submit_nums={submit_nums}")
    print(f"并发数: {CONCURRENCY}, 随机种子: {RANDOM_SEED}")
    print("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    eval_csv = DATASET_DIR / "eval.csv"
    if not eval_csv.exists():
        print(f"[ERROR] 找不到评估文件: {eval_csv}")
        return

    df = pd.read_csv(eval_csv)
    print(f"加载 eval.csv: {len(df)} 条")

    sample_size = min(extract_nums, len(df))
    random.seed(RANDOM_SEED)
    sampled_indices = random.sample(range(len(df)), sample_size)
    sampled_df = df.iloc[sampled_indices].reset_index(drop=True)

    print(f"随机抽取 {sample_size} 条进行评测")

    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            evaluate_one_row(session, row, sem)
            for _, row in sampled_df.iterrows()
        ]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            if result:
                results.append(result)
                print(f"  ✓ [{i+1}/{sample_size}] {result['mail_id']} - {result['num_rounds']}轮 "
                      f"Rouge={result['avg_rouge1_f1']:.4f} BLEU={result['avg_bleu']:.4f} "
                      f"Sem={result['avg_semantic_sim']:.4f}")
            else:
                print(f"  ✗ [{i+1}/{sample_size}] 失败")

    if not results:
        print("[ERROR] 没有成功评估任何样本")
        return

    overall_avg = {
        'avg_rouge1_f1': sum(r['avg_rouge1_f1'] for r in results) / len(results),
        'avg_bleu': sum(r['avg_bleu'] for r in results) / len(results),
        'avg_semantic_sim': sum(r['avg_semantic_sim'] for r in results) / len(results),
        'avg_latency_ms': sum(r['avg_latency_ms'] for r in results) / len(results),
    }

    # Aggregate per-endpoint latencies across all results
    ep_all: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for ep, ms in r.get('avg_endpoint_latencies_ms', {}).items():
            ep_all[ep].append(ms)
    overall_avg['avg_endpoint_latencies_ms'] = {
        ep: round(sum(v) / len(v), 2) for ep, v in ep_all.items()
    }

    print("\n" + "=" * 60)
    print("评估完成，整体指标:")
    print(f"  Rouge-1 F1: {overall_avg['avg_rouge1_f1']:.4f}")
    print(f"  BLEU:       {overall_avg['avg_bleu']:.4f}")
    print(f"  语义相似度: {overall_avg['avg_semantic_sim']:.4f}")
    print(f"  平均总延迟: {overall_avg['avg_latency_ms']:.2f}ms")
    print("  接口平均延迟:")
    for ep, ms in overall_avg['avg_endpoint_latencies_ms'].items():
        print(f"    /{ep}: {ms:.2f}ms")
    print("=" * 60)

    print("\n正在进行大模型质量评估...")
    llm_summary = await run_llm_assessment(results, overall_avg)
    print(llm_summary)
    print("大模型评估完成")

    now = datetime.now()
    ts_str = now.strftime("%Y%m%d_%H%M%S")

    json_data = {
        'metadata': {
            'timestamp': now.isoformat(),
            'sample_count': len(results),
            'random_seed': RANDOM_SEED,
            'extract_nums': extract_nums,
            'submit_nums': submit_nums
        },
        'overall_average': overall_avg,
        'llm_summary': llm_summary,
        'results': results
    }

    json_path = REPORT_DIR / f"eval_{ts_str}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告已保存: {json_path}")

    html = generate_html(
        results,
        overall_avg,
        llm_summary,
        ts_str,
        json_data['metadata']
    )
    html_path = REPORT_DIR / f"eval_{ts_str}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML 报告已保存: {html_path}")

    print("\n评估完成!")


if __name__ == "__main__":
    asyncio.run(main())
