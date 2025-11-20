from atomgpt.inverse_models.llama import *  # noqa: F401,F403
import os

from atomgpt.inverse_models._utils import __version__  # noqa: F401
from atomgpt.inverse_models._utils2 import Version, _get_dtype  # noqa: F401

try:
    # New HF GPT-OSS modeling API
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssModel,
        GptOssForCausalLM,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "AtomGPT: transformers installation does not appear to include "
        "the `gpt_oss` model. Please upgrade transformers:\n"
        '  pip install --upgrade "transformers"\n'
        "and ensure you are on a release that supports GPT-OSS."
    ) from exc

# --- AtomGPT: fix GPT-OSS position_ids shape for rotary embeddings ---
# Some fast-generation paths may end up passing a 1D tensor for `position_ids`
# (shape [seq_len]), but GPT-OSS's rotary embeddings expect [batch, seq_len].
# This wrapper upgrades 1D position_ids → [1, seq_len] to avoid IndexError.

if not hasattr(GptOssModel, "_atomgpt_position_ids_patched"):
    _original_gpt_oss_forward = GptOssModel.forward

    def _atomgpt_gpt_oss_forward(self, *args, **kwargs):
        pos = kwargs.get("position_ids", None)
        try:
            if pos is not None and hasattr(pos, "dim") and pos.dim() == 1:
                # [seq_len] -> [1, seq_len]
                kwargs["position_ids"] = pos.unsqueeze(0)
        except Exception:
            # Best-effort: never let our fix be the thing that breaks.
            pass
        return _original_gpt_oss_forward(self, *args, **kwargs)

    GptOssModel.forward = _atomgpt_gpt_oss_forward
    GptOssModel._atomgpt_position_ids_patched = True

    print(
        "AtomGPT: Patched GptOssModel.forward to fix 1D position_ids for GPT-OSS rotary embeddings."
    )


# Convenience list of all 4 Unsloth GPT-OSS models that are supported via
# FastLanguageModel.from_pretrained(..., model_name=...).
#
# You can use these as drop-in `model_name` values:
#
#   from atomgpt.inverse_models.gpt_oss import UNSLOTH_GPT_OSS_MODELS
#   model_name = UNSLOTH_GPT_OSS_MODELS[0]
#
UNSLOTH_GPT_OSS_MODELS = [
    # BitsAndBytes 4bit Unsloth quantizations
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    # MXFP4 “original” weights that Unsloth wraps
    "unsloth/gpt-oss-20b",
    "unsloth/gpt-oss-120b",
]


def _log_once(msg: str) -> None:
    """Tiny helper to avoid spamming logs if imported multiple times."""
    if getattr(_log_once, "_seen", None) is None:
        _log_once._seen = set()
    if msg in _log_once._seen:
        return
    _log_once._seen.add(msg)
    print(msg)


class FastGptOssModel(FastLlamaModel):
    """
    Fast GPT-OSS integration for AtomGPT.

    This mirrors the overall structure of `FastMistralModel` but takes a more
    conservative approach:

    * We **do not** override GPT-OSS attention / MoE internals. Those are
      handled by the upstream `transformers` implementation and whatever
      `unsloth_compile_transformers` is already doing in your loader.
    * We **do**:
        - patch PEFT `PeftModelForCausalLM.forward` to the same fast path
          that LLaMA / Mistral use.
        - (for now) leave `GptOssForCausalLM.prepare_inputs_for_generation`
          untouched, because the LLaMA-style patch breaks GPT-OSS attention
          shapes during sampling.
    * Everything else is delegated to `FastLlamaModel.from_pretrained` with
      `model_patcher=FastGptOssModel`, to keep the hierarchy uniform.
    """

    @staticmethod
    def pre_patch():
        """
        Apply GPT-OSS-specific patches.

        We deliberately do **not** touch GPT-OSS attention / decoder layer
        implementations here, to avoid shape / MoE wiring mistakes. Instead we
        reuse only the architecture-agnostic bits from `llama.py`.
        """
        # Reuse the PEFT fast forward path (architecture-agnostic: it only
        # assumes a CausalLM head with `.lm_head`).
        global PeftModelForCausalLM  # imported from llama.py via *
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward

        # IMPORTANT:
        # Do NOT call `fix_prepare_inputs_for_generation(GptOssForCausalLM)`
        # here. That patch is tailored to LLaMA/Mistral KV-cache semantics and
        # causes attention shape mismatches for GPT-OSS (e.g. value_states
        # ending up with seq_len = 1 instead of the full context length).
        #
        # We'll rely on the official transformers implementation of
        # `prepare_inputs_for_generation` for GPT-OSS instead.
        # fix_prepare_inputs_for_generation(GptOssForCausalLM)

        _log_once(
            "AtomGPT: Patched GPT-OSS (PEFT fast forward only; "
            "using native prepare_inputs_for_generation)."
        )
        return


    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length: int | None = None,
        dtype=None,
        load_in_4bit: bool = True,
        token=None,
        device_map: str | dict = "sequential",
        rope_scaling=None,  # GPT-OSS does not use classic RoPE scaling, kept for API symmetry
        fix_tokenizer: bool = True,
        model_patcher=None,
        tokenizer_name: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Thin wrapper around `FastLlamaModel.from_pretrained`.

        The important part is that we pass `model_patcher=FastGptOssModel`,
        which causes:

          * `FastGptOssModel.pre_patch()` to run before loading.
          * All the Unsloth / AtomGPT compile + quantization machinery to be
            reused exactly as for LLaMA / Mistral.

        Usage (drop-in with your loader):

            from atomgpt.inverse_models.loader import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
        """
        # Defer to the LLaMA machinery – it will:
        #   * call FastGptOssModel.pre_patch()
        #   * run unsloth_compile_transformers
        #   * handle bitsandbytes / 4bit / 8bit / PEFT, etc.
        return FastLlamaModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
            device_map=device_map,
            rope_scaling=rope_scaling,
            fix_tokenizer=fix_tokenizer,
            model_patcher=FastGptOssModel if model_patcher is None else model_patcher,
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


__all__ = [
    "FastGptOssModel",
    "UNSLOTH_GPT_OSS_MODELS",
]

