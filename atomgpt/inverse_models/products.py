# products.py

from dataclasses import dataclass
from typing import Protocol, Any, Callable
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase


class DatasetFormattingFunction(Protocol):
    def get_formatting_prompts_func() -> Callable:
        pass
