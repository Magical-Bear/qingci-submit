"""
Kimi API 客户端 (OpenAI 兼容)
带重试、异步
"""
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from codes.config import settings


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.kimi_api_key,
        base_url=settings.kimi_base_url,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def chat(
    messages: list[dict],
    max_tokens: int = 2048,
) -> str:
    """调用 Kimi API，返回 content 字符串。注意: kimi-k2.5 只支持 temperature=1"""
    client = get_client()
    resp = await client.chat.completions.create(
        model=settings.kimi_model,
        messages=messages,
        temperature=1,   # kimi-k2.5 仅支持 temperature=1
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
