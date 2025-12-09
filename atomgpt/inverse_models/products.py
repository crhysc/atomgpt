# products.py

from dataclasses import dataclass
from typing import Protocol, Any
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase


class ChatTemplate(Protocol):
    def format() -> str:
        pass
