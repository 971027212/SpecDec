from __future__ import annotations

"""Hugging Face Transformers causal LM runner。

model 层只适配 encode/decode/next_token_logits，不写 generation loop。
draft runner 和 verifier 会复用这个接口，但不会依赖 Transformers 的具体类型。
"""

from dataclasses import dataclass
from typing import Any

from specplatform.model.base import CausalLMRunner, ModelForwardInput, ModelForwardOutput


@dataclass
class TransformersCausalLMRunner(CausalLMRunner):
    """把真实 Transformers causal LM 包装成 CausalLMRunner。"""

    runner_id: str
    tokenizer: Any
    model: Any
    max_len: int
    device: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        runner_id: str,
        device: str | None = "cuda",
        torch_dtype: str | None = "auto",
        device_map: str | None = None,
        trust_remote_code: bool = True,
    ) -> "TransformersCausalLMRunner":
        """加载本地 Hugging Face 权重。

        这里延迟 import torch/transformers，保证普通单元测试不需要安装大模型依赖。
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        dtype = _resolve_torch_dtype(torch, torch_dtype)
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device_map is None and device is not None:
            model = model.to(device)
        model.eval()

        max_len = int(
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "seq_length", None)
            or 32768
        )
        input_device = _first_parameter_device(model, fallback=device)
        return cls(
            runner_id=runner_id,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            device=input_device,
        )

    def encode(self, text: str) -> list[int]:
        """把文本编码成 token ids；不在这里启动生成循环。"""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 解码成文本，用于 smoke 输出检查。"""
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """执行一次模型 forward，返回每个位置的 logits。"""
        import torch

        if not request.input_ids:
            raise ValueError("TransformersCausalLMRunner.forward requires input_ids.")
        input_ids = torch.tensor([request.input_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
        logits = output.logits[0].detach().float().cpu().tolist()
        return ModelForwardOutput(logits=logits, metadata={"runner_id": self.runner_id})

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        """取最后一个 prefix 位置的 logits，供 greedy_next_token/verifier 使用。"""
        import torch

        if not prefix_ids:
            raise ValueError("TransformersCausalLMRunner.next_token_logits requires a non-empty prefix.")
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
        # 只搬运最后一个位置的 logits；verification/greedy 不需要整段序列 logits。
        return output.logits[0, -1].detach().float().cpu().tolist()


def _resolve_torch_dtype(torch: Any, torch_dtype: str | None) -> Any | None:
    """把命令行 dtype 字符串转换成 torch dtype。"""
    if torch_dtype in (None, "auto"):
        return None if torch_dtype is None else "auto"
    if torch_dtype == "bf16":
        return torch.bfloat16
    if torch_dtype == "fp16":
        return torch.float16
    if torch_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


def _first_parameter_device(model: Any, *, fallback: str | None) -> str | None:
    """推断模型输入应该放到哪个 device。"""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback
