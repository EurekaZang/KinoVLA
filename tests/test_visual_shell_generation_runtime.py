from __future__ import annotations

from scripts.generate_embodiedgen_visual_shell_pano import (
    _audit_pipeline_device,
    _audit_prompt_tokens,
)


class _Parameter:
    def __init__(self, device: str):
        self.device = device


class _Module:
    def __init__(self, device: str):
        self._device = device

    def parameters(self):
        yield _Parameter(self._device)


class _Pipeline:
    def __init__(self, pipeline_device: str, component_devices: tuple[str, str, str]):
        self.device = pipeline_device
        self.text_encoder = _Module(component_devices[0])
        self.unet = _Module(component_devices[1])
        self.vae = _Module(component_devices[2])


class _Tokenizer:
    model_max_length = 6

    def __call__(self, prompt: str, **_: object):
        return {"input_ids": list(range(len(prompt.split()) + 2))}


def test_single_cuda_device_passes() -> None:
    result = _audit_pipeline_device(
        _Pipeline("cuda:0", ("cuda:0", "cuda:0", "cuda:0")), expected_prefix="cuda"
    )
    assert result["passed"] is True


def test_cpu_cuda_split_fails_before_generation() -> None:
    result = _audit_pipeline_device(
        _Pipeline("cpu", ("cuda:0", "cuda:0", "cpu")), expected_prefix="cuda"
    )
    assert result["passed"] is False
    assert result["checks"]["single_device_policy"] is False


def test_prompt_token_audit_accepts_context_fitting_prompt() -> None:
    result = _audit_prompt_tokens(_Tokenizer(), "one two three four")
    assert result["token_count_including_special_tokens"] == 6
    assert result["passed"] is True


def test_prompt_token_audit_rejects_silent_truncation() -> None:
    result = _audit_prompt_tokens(_Tokenizer(), "one two three four five")
    assert result["token_count_including_special_tokens"] == 7
    assert result["checks"]["within_model_context"] is False
    assert result["passed"] is False
