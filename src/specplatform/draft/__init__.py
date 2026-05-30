"""draft runner 出口。

draft 层只负责 draft model 的执行能力，不负责验证 proposal 或决定接受哪些 token。
真实 GreedyDraftRunner 会在 Step 2 引入。
"""

__all__: list[str] = []
