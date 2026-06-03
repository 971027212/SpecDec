"""runtime 执行层出口。

runtime 负责统一编排生成流程；method-specific 的差异应通过策略对象注入，
不应写成 runtime 内部的 if/else 分支。
"""

from specplatform.runtime.async_pipeline import AsyncPipelineRuntimeEngine
from specplatform.runtime.distributed_pipeline import DistributedBatchPipelineRuntimeEngine
from specplatform.runtime.engine import RuntimeEngine, RuntimeRequestResult, RuntimeRunResult
from specplatform.runtime.session import GenerationSession

__all__ = [
    "GenerationSession",
    "AsyncPipelineRuntimeEngine",
    "DistributedBatchPipelineRuntimeEngine",
    "RuntimeEngine",
    "RuntimeRequestResult",
    "RuntimeRunResult",
]
