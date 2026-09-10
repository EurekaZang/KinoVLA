"""KinoVLA: Qwen3-VL-4B + LoRA + Kino-Projector — the Recovery Planner (spec §10 PHASE 4 / §11).

The deployed planner (spec §1 layer 03) is a vision-language model that reads the body camera and
its proprioception and emits a ``<Thought>…</Thought><Action>{json}</Action>`` recovery. This
wrapper assembles it:

- **backbone**: ``Qwen3-VL-4B-Instruct`` (the latest Qwen VLM; supersedes the
  spec's Qwen2-VL), LoRA-adapted on the language tower (Kino-SFT, spec §11 Stage 1);
- **latent route (B5)**: the :class:`~kino_vla.vla.projector.KinoProjector` lifts the 500 ms
  proprioception into K soft tokens spliced at the ``<kino_tokens>`` placeholder. The splice is a
  forward hook on the token-embedding layer, so the model's native vision merge (incl. deepstack)
  and 3-D mrope are untouched — only the K Kino-token rows are overwritten with the projected
  embeddings (autograd flows back into the projector);
- **text route (B4)**: no projector; the proprioception is already in the prompt text (the §3
  ablation arm). Selected by ``route`` at construction.

Loss is next-token CE on the completion tokens only (the prompt — system, vision, Kino-tokens — is
masked) plus a small privileged-θ MSE on the projector (spec §4 grounding). This is the real
GPU/model dependency (§0 hard rule): unit-smoke-tested against the real weights in
``tests/test_vla_model_smoke.py`` (sim/GPU-gated), not the CPU surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from kino_vla.utils.config import Config
from kino_vla.vla.projector import KinoProjector
from kino_vla.vla.prompt import KINO_TAG

KINO_TOKEN = "<|kino|>"  # dedicated placeholder token; its embedding is always overwritten


@dataclass
class VlaInputs:
    """Tokenized model inputs for one example (built by :meth:`KinoVLA.build_inputs`)."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    mm_token_type_ids: torch.Tensor | None  # Qwen3-VL M-RoPE needs this (from the processor)
    proprio_window: torch.Tensor | None  # (T, F) raw, for the projector (latent route)
    target_theta: torch.Tensor | None


class KinoVLA(nn.Module):
    """Qwen3-VL-4B + LoRA recovery planner with optional Kino-Token latent injection."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        route: str = "latent",
        projector: KinoProjector | None = None,
        kino_token_id: int | None = None,
        theta_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.route = route
        self.projector = projector
        self.kino_token_id = kino_token_id
        self.theta_loss_weight = float(theta_loss_weight)
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None  # (kino_mask, soft_tokens)
        if route == "latent":
            if projector is None or kino_token_id is None:
                raise ValueError("latent route needs a projector and a kino_token_id")
            self._register_kino_hook()

    # ----------------------------------------------------------------- construction
    @classmethod
    def from_pretrained(
        cls,
        cfg: Config,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        adapter_dir: str | Path | None = None,
    ) -> KinoVLA:
        """Load the backbone + processor, attach LoRA + the Kino-Projector (spec §11 Stage 1)."""
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        m = cfg.model
        route = str(cfg.get("route", "latent"))
        # KINOVLA_MODEL_ID lets a box with a locally-staged checkpoint override the canonical HF id
        # (kept portable in the config) without editing committed YAML — e.g. when the HF download
        # was staged to a local dir. Falls back to the config's model_id.
        import os as _os

        model_id = _os.environ.get("KINOVLA_MODEL_ID") or str(m.model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype)

        kino_token_id = None
        projector = None
        if route == "latent":
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [KINO_TOKEN]})
            model.resize_token_embeddings(len(processor.tokenizer))
            kino_token_id = processor.tokenizer.convert_tokens_to_ids(KINO_TOKEN)

        # LoRA on the language tower (Kino-SFT). Vision + projector carry the modality bridge.
        lora = m.lora
        peft_cfg = LoraConfig(
            r=int(lora.r),
            lora_alpha=int(lora.alpha),
            lora_dropout=float(lora.dropout),
            target_modules=list(lora.target_modules),
            task_type="CAUSAL_LM",
        )
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        else:
            model = get_peft_model(model, peft_cfg)

        if route == "latent":
            vlm_dim = int(model.config.text_config.hidden_size)
            pj = cfg.projector
            projector = KinoProjector(
                vlm_dim,
                conv_channels=tuple(int(c) for c in pj.conv_channels),
                conv_kernels=tuple(int(k) for k in pj.conv_kernels),
                latent_dim=int(pj.latent_dim),
                n_latents=int(pj.n_latents),
                n_heads=int(pj.n_heads),
                mlp_hidden=int(pj.mlp_hidden),
                proj_hidden=int(pj.get("proj_hidden", 512)),
            )
            if adapter_dir is not None:
                pj_path = Path(adapter_dir) / "kino_projector.pt"
                if pj_path.exists():
                    projector.load_state_dict(torch.load(pj_path, map_location="cpu"))
            projector = projector.to(device=device, dtype=torch.float32)

        model = model.to(device)
        return cls(
            model,
            processor,
            route=route,
            projector=projector,
            kino_token_id=kino_token_id,
            theta_loss_weight=float(cfg.train.get("theta_loss_weight", 0.1)),
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ------------------------------------------------------------ kino splice hook
    def _register_kino_hook(self) -> None:
        """Forward hook on the token-embedding layer: overwrite Kino-token rows with soft tokens.

        Fires on every ``embed_tokens`` call; a no-op when no Kino token is present (so generation
        continuation steps and the text route are untouched). Returns a fresh tensor so autograd
        flows from the projected soft tokens through the LM to the loss.
        """
        embed = self.model.get_input_embeddings()

        def hook(_module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
            if self._pending is None:
                return output
            kino_mask, soft = self._pending
            ids = inputs[0]
            mask = ids == self.kino_token_id
            if not bool(mask.any()):
                return output
            out = output.clone()
            out[mask] = soft.reshape(-1, soft.shape[-1]).to(out.dtype)
            return out

        embed.register_forward_hook(hook)

    def _set_pending_soft(self, input_ids: torch.Tensor, window: torch.Tensor | None) -> None:
        """Project the proprio window → soft tokens and stash them for the embedding hook."""
        if self.route != "latent" or window is None:
            self._pending = None
            return
        w = window.to(device=self.device, dtype=torch.float32)
        if w.dim() == 2:
            w = w.unsqueeze(0)
        soft, _theta = self.projector(w)  # (B, K, D)
        kino_mask = input_ids == self.kino_token_id
        self._pending = (kino_mask, soft)

    # --------------------------------------------------------------- input building
    def _expand_messages(self, messages: list[dict], images: list[Image.Image]) -> list[dict]:
        """Inline the PIL images into the message image placeholders for the chat template."""
        out: list[dict] = []
        img_iter = iter(images)
        for msg in messages:
            content = []
            for c in msg["content"]:
                if c.get("type") == "image":
                    content.append({"type": "image", "image": next(img_iter)})
                else:
                    content.append(c)
            out.append({"role": msg["role"], "content": content})
        return out

    def _kino_text(self, text: str) -> str:
        """Replace the single ``<kino_tokens>`` marker with K copies of the kino placeholder."""
        if self.route != "latent":
            return text.replace(KINO_TAG, "")
        return text.replace(KINO_TAG, KINO_TOKEN * self.projector.n_soft_tokens)

    def build_inputs(
        self,
        messages: list[dict],
        images: list[np.ndarray],
        *,
        target_text: str | None = None,
        proprio_window: np.ndarray | None = None,
        target_theta: list[float] | None = None,
        loss_span: str = "completion",
    ) -> VlaInputs:
        """Build tokenized inputs (+ masked labels when ``target_text`` is given).

        ``loss_span`` controls which completion tokens are supervised / scored:
        ``"completion"`` (default) the whole ``<Thought>…</Action>``; ``"action"`` only the
        ``<Action>{…}</Action>`` decision span (the Thought is masked into the context). The
        ``"action"`` span is the fix for on-policy DPO: free-form Chosen/Rejected
        generations differ wholesale in their Thought prose, so a whole-completion preference
        optimizes spurious narrative features; masking to the Action makes the DPO contrast the
        *decision* (attribution + primitive), the §11 target.
        """
        pil = [Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)) for im in images]
        msgs = self._expand_messages(messages, pil)
        prompt_text = self._kino_text(
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
        proc_imgs = pil if pil else None

        if target_text is None:  # inference
            enc = self.processor(text=[prompt_text], images=proc_imgs, return_tensors="pt")
            labels = None
        else:  # training: full = prompt + completion + <|im_end|>
            full_text = prompt_text + target_text + "<|im_end|>\n"
            enc = self.processor(text=[full_text], images=proc_imgs, return_tensors="pt")
            # Mask everything up to the supervised span. For loss_span="action" the Thought joins
            # the masked prefix, so only the <Action> decision tokens carry the loss/logprob.
            mask_prefix = prompt_text
            if loss_span == "action" and "<Action>" in target_text:
                mask_prefix = prompt_text + target_text.split("<Action>", 1)[0]
            elif loss_span not in ("completion", "action"):
                raise ValueError(f"loss_span must be 'completion' or 'action', got {loss_span!r}")
            prefix_enc = self.processor(text=[mask_prefix], images=proc_imgs, return_tensors="pt")
            mask_len = int(prefix_enc["input_ids"].shape[1])
            labels = enc["input_ids"].clone()
            labels[:, :mask_len] = -100
            labels[enc["attention_mask"] == 0] = -100

        return VlaInputs(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
            pixel_values=enc.get("pixel_values"),
            image_grid_thw=enc.get("image_grid_thw"),
            mm_token_type_ids=enc.get("mm_token_type_ids"),
            proprio_window=(
                torch.tensor(np.asarray(proprio_window), dtype=torch.float32)
                if proprio_window is not None
                else None
            ),
            target_theta=(
                torch.tensor(target_theta, dtype=torch.float32)
                if target_theta is not None
                else None
            ),
        )

    def _to_device(self, x: VlaInputs) -> VlaInputs:
        dev = self.device
        return VlaInputs(
            input_ids=x.input_ids.to(dev),
            attention_mask=x.attention_mask.to(dev),
            labels=x.labels.to(dev) if x.labels is not None else None,
            pixel_values=x.pixel_values.to(dev) if x.pixel_values is not None else None,
            image_grid_thw=x.image_grid_thw.to(dev) if x.image_grid_thw is not None else None,
            mm_token_type_ids=(
                x.mm_token_type_ids.to(dev) if x.mm_token_type_ids is not None else None
            ),
            proprio_window=x.proprio_window,
            target_theta=x.target_theta.to(dev) if x.target_theta is not None else None,
        )

    # ----------------------------------------------------------------------- forward
    def compute_loss(self, x: VlaInputs) -> dict[str, torch.Tensor]:
        """Next-token CE on the completion + the privileged-θ grounding loss (spec §4/§11)."""
        x = self._to_device(x)
        theta_loss = torch.zeros((), device=self.device)
        if self.route == "latent":
            w = x.proprio_window.to(self.device, dtype=torch.float32).unsqueeze(0)
            soft, theta_pred = self.projector(w)
            self._pending = (x.input_ids == self.kino_token_id, soft)
            if x.target_theta is not None:
                theta_loss = nn.functional.mse_loss(theta_pred.squeeze(0), x.target_theta)
        out = self.model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            pixel_values=x.pixel_values,
            image_grid_thw=x.image_grid_thw,
            mm_token_type_ids=x.mm_token_type_ids,
            labels=x.labels,
        )
        self._pending = None
        lm_loss = out.loss
        total = lm_loss + self.theta_loss_weight * theta_loss
        return {"loss": total, "lm_loss": lm_loss.detach(), "theta_loss": theta_loss.detach()}

    def _logits_and_labels(self, x: VlaInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, labels) with the Kino soft tokens spliced in."""
        x = self._to_device(x)
        if self.route == "latent":
            w = x.proprio_window.to(self.device, dtype=torch.float32).unsqueeze(0)
            soft, _ = self.projector(w)
            self._pending = (x.input_ids == self.kino_token_id, soft)
        out = self.model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            pixel_values=x.pixel_values,
            image_grid_thw=x.image_grid_thw,
            mm_token_type_ids=x.mm_token_type_ids,
        )
        self._pending = None
        return out.logits, x.labels

    def completion_logprob(self, x: VlaInputs, *, reference: bool = False) -> torch.Tensor:
        """Sum of token log-probs over the completion (labels != -100).

        ``reference=True`` evaluates the LoRA-OFF base model under ``no_grad`` (the adapter-toggle
        DPO reference — no second model loaded); the policy path keeps grad.
        """
        import contextlib

        adapter_ctx = self.model.disable_adapter() if reference else contextlib.nullcontext()
        grad_ctx = torch.no_grad() if reference else contextlib.nullcontext()
        with adapter_ctx, grad_ctx:
            logits, labels = self._logits_and_labels(x)
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
            mask = labels != -100
            logp = torch.log_softmax(logits.float(), dim=-1)
            tok = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            return (tok * mask).sum()

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict],
        images: list[np.ndarray],
        *,
        proprio_window: np.ndarray | None = None,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
    ) -> str:
        """Generate one recovery reflection (the closed-loop / eval inference path)."""
        x = self._to_device(self.build_inputs(messages, images, proprio_window=proprio_window))
        self._set_pending_soft(x.input_ids, x.proprio_window)
        do_sample = temperature > 0.0
        gen = self.model.generate(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            pixel_values=x.pixel_values,
            image_grid_thw=x.image_grid_thw,
            mm_token_type_ids=x.mm_token_type_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        self._pending = None
        new = gen[0, x.input_ids.shape[1] :]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True)

    # ------------------------------------------------------------------- checkpoint
    def save_adapter(self, path: str | Path) -> None:
        """Persist the LoRA adapter + the Kino-Projector (the only trained weights)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        if self.projector is not None:
            torch.save(self.projector.state_dict(), path / "kino_projector.pt")

    def trainable_parameters(self) -> list[nn.Parameter]:
        """The parameters Kino-SFT updates: the LoRA adapter + the projector."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.projector is not None:
            params += list(self.projector.parameters())
        return params
