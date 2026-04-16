# 接口性能测试说明

## 测试文件

[tests/test_api_performance.py](../tests/test_api_performance.py)

### 主要作用

对 ぽちゃガチョ！ 客服系统的所有 FastAPI 接口进行性能基准测试。每个接口运行 **5 次**，统计以下指标：

- **均值**：平均响应时间
- **最小 / 最大**：响应时间区间
- **标准差**：衡量响应时间离散程度
- **P50（中位数）**：排除极端值后的典型响应时间
- **变异系数（CV）**：标准差 / 均值，评定稳定性等级（稳定 / 一般 / 波动较大）

### 覆盖接口

| 测试类 | 接口 | 前置准备 |
|--------|------|----------|
| `TestHealthPerformance` | `GET /api/health` | 无 |
| `TestNewTicketPerformance` | `POST /api/ticket/new` | 无（5 个不同日语问题） |
| `TestFinalizePerformance` | `POST /api/ticket/finalize` | 每次先调用 `new` 创建 session |
| `TestFollowupPerformance` | `POST /api/ticket/followup` | 每次先调用 `new` → `finalize` |
| `TestGetSessionPerformance` | `GET /api/session/{id}` | 创建一个 session 后重复查询 5 次 |
| `TestErrorResponsePerformance` | `GET /api/session/{id}` (404) | 错误路径基准，无需准备 |

### 运行方式

> 运行前须确保服务已在 `http://localhost:8000` 启动。

```bash
# 启动服务（另开终端）
uv run uvicorn codes.service.app:app --reload --port 8000

# 运行性能测试
uv run pytest tests/test_api_performance.py -v
```

---

## 报告位置

测试结束后报告自动生成于：

```
report/apis/<yy-mm-dd-hh-mm>.md
```

查看最新报告，请取 `report/apis/` 目录下**按文件名排序最后一个** `.md` 文件：

```bash
ls report/apis/ | sort | tail -1
```

---

## 注意事项：新增测试代码规范

> **严禁覆盖 `tests/test_api_performance.py` 和 `docs/tests.md`。**
>
> - 新增测试代码时，请直接在 `tests/test_api_performance.py` **末尾追加**新的测试类或函数，不得删除或修改已有内容。
> - 新增测试的文档说明，请在本文件（`docs/tests.md`）**末尾追加**对应章节，不得删除或修改已有章节。

追加测试代码示例：

```python
# 在 test_api_performance.py 末尾追加
@pytest.mark.asyncio
class TestNewEndpointPerformance:
    """新接口性能测试 — 追加于文件末尾，不覆盖已有内容"""

    async def test_new_endpoint_5_runs(self, perf_results: dict) -> None:
        ...
```

新增的测试类只需接受 `perf_results` fixture，测试结果会自动汇入同一份报告。
