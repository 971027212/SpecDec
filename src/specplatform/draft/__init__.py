"""draft runner 出口。

draft 层只负责 draft model 的执行能力，不负责验证 proposal 或决定接受哪些 token。
"""

from specplatform.draft.runner import FakeDraftRunner

__all__ = ["FakeDraftRunner"]
