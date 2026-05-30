from __future__ import annotations

"""3090 侧最小跨机 speculative decoding smoke。

默认加载 /home/chajiahao/data/hf_models/Qwen3-1.7B 作为 draft model，
通过 HTTP 调 A100 的 /verify_linear target verifier。
"""

import argparse

from specplatform.core import DraftBudget, RuntimeContext
from specplatform.draft import GreedyDraftRunner
from specplatform.methods import GreedyPrefixAcceptancePolicy, LinearCandidateStrategy
from specplatform.model import TransformersCausalLMRunner
from specplatform.runtime import GenerationSession, RuntimeEngine
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.verification import HttpLinearVerifierClient


def build_parser() -> argparse.ArgumentParser:
    """解析 smoke 参数。"""
    parser = argparse.ArgumentParser(description="Run one minimal 3090 -> A100 speculative decoding smoke.")
    parser.add_argument("--draft-model-path", default="/home/chajiahao/data/hf_models/Qwen3-1.7B")
    parser.add_argument("--target-url", default="http://a100-specdec:8010")
    parser.add_argument("--prompt", default="介绍一下 speculative decoding")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default=None)
    return parser


def main() -> None:
    """启动单请求 speculative decoding，并打印 token 与文本结果。"""
    args = build_parser().parse_args()
    draft_model = TransformersCausalLMRunner.from_pretrained(
        args.draft_model_path,
        runner_id="3090-qwen3-1.7b",
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    prompt_ids = draft_model.encode(args.prompt)
    eos_token_ids = _tokenizer_eos_ids(draft_model.tokenizer)
    session = GenerationSession(
        request_id="smoke-1",
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        max_len=draft_model.max_len,
        eos_token_ids=eos_token_ids,
    )
    engine = RuntimeEngine(
        candidate_strategy=LinearCandidateStrategy(),
        acceptance_policy=GreedyPrefixAcceptancePolicy(),
        scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=args.draft_tokens)),
        verifier=HttpLinearVerifierClient(base_url=args.target_url),
    )
    result = engine.run(
        run_id="smoke-run",
        sessions=[session],
        draft_runners={"draft-worker-0": GreedyDraftRunner(model=draft_model, runner_id="draft-worker-0")},
        context=RuntimeContext(
            run_config={
                "method": "linear",
                "eos_token_ids": eos_token_ids,
            },
            backend_info={
                "target_placement": "a100",
                "target_backend": "http_linear",
                "target_host": args.target_url,
            },
        ),
    )
    output_text = draft_model.decode(session.generated_ids)
    print("request_id:", result.request_results[0].request_id)
    print("generated_ids:", session.generated_ids)
    print("generated_len:", len(session.generated_ids))
    print("is_finished:", session.is_finished)
    print("text:", output_text)


def _tokenizer_eos_ids(tokenizer: object) -> list[int]:
    """从 tokenizer 中提取一个或多个 EOS token id。"""
    eos_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_ids.append(int(eos_token_id))
    additional = getattr(tokenizer, "additional_special_tokens_ids", None) or []
    for token_id in additional:
        if int(token_id) not in eos_ids:
            eos_ids.append(int(token_id))
    return eos_ids


if __name__ == "__main__":
    main()
