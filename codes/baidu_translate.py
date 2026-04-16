"""
百度翻译 API 客户端 (aiohttp 异步，仅用于日语→中文)
文档: https://api.fanyi.baidu.com/doc/21
"""
from __future__ import annotations

import hashlib
import random

import aiohttp

from codes.config import settings

_URL = "http://api.fanyi.baidu.com/api/trans/vip/translate"


def _sign(appid: str, text: str, salt: int, appkey: str) -> str:
    raw = appid + text + str(salt) + appkey
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


async def translate_ja_to_zh(text: str) -> str:
    """将日语文本翻译成中文，返回翻译结果字符串。"""
    appid = settings.baidu_translate_app_id
    appkey = settings.baidu_translate_api_key
    salt = random.randint(32768, 65536)
    params = {
        "appid": appid,
        "q": text,
        "from": "jp",
        "to": "zh",
        "salt": str(salt),
        "sign": _sign(appid, text, salt, appkey),
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            _URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            result = await resp.json(content_type=None)

    if "trans_result" not in result:
        raise RuntimeError(f"百度翻译失败: {result}")

    return "".join(item["dst"] for item in result["trans_result"])
