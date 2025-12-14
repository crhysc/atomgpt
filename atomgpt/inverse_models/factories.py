# factories.py

from abc import ABC, abstractmethod
from atomgpt.inverse_models.products import LoadedModel
from typing import Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from atomgpt.inverse_models.inverse_models import TrainingPropConfig
from peft import PeftModel
from atomgpt.inverse_models.loader import FastLanguageModel as AtomGPTFastLanguageModel
from unsloth import FastLanguageModel as UnslothFastLanguageModel
from typing import Dict
from atomgpt.inverse_models.dataset_utils import alpaca_formatting_prompts_func
from atomgpt.inverse_models.dataset_utils import harmony_formatting_prompts_func
from functools import partial
from typing import List


class LanguageModelFactory(ABC):
    @abstractmethod
    def load_for_training(self, config: TrainingPropConfig) -> LoadedModel:
        pass

    @abstractmethod
    def load_for_inference(self, checkpoint_path: str, config: TrainingPropConfig) -> LoadedModel:
        pass

    @abstractmethod
    def get_formatting_prompts_func(self, config, model, tokenizer) -> Callable:
        pass


class AtomGPTFactory(LanguageModelFactory):
    def load_for_training(self, config: TrainingPropConfig) -> LoadedModel:
        model, tokenizer = AtomGPTFastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit
        )
        if not isinstance(model, PeftModel):
            # import sys
            print("Not yet a peft model, converting into peft model")
            # sys.exit()
            model = AtomGPTFastLanguageModel.get_peft_model(
                model,
                r=config.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=config.lora_alpha,
                lora_dropout=0,  # Supports any, but = 0 is optimized
                bias="none",  # Supports any, but = "none" is optimized
                use_gradient_checkpointing=True,
                random_state=3407,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None,  # And LoftQ
            )
            print("Peft model created")
        EOS_TOKEN = tokenizer.eos_token
        return LoadedModel(model=model, tokenizer=tokenizer)

    def load_for_inference(self, checkpoint_path: str, config: TrainingPropConfig) -> LoadedModel:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
        )
        AtomGPTFastLanguageModel.for_inference(model)
        return LoadedModel(model=model, tokenizer=tokenizer)

    def get_formatting_prompts_func(self, config, model, tokenizer) -> Callable:
        eos = tokenizer.eos_token or "</s>"
        return partial(alpaca_formatting_prompts_func, alpaca_prompt=config.alpaca_prompt, eos_token=eos)


class GPTOSSFactory(LanguageModelFactory):
    def load_for_training(self, config: TrainingPropConfig) -> LoadedModel:
        model, tokenizer = UnslothFastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
            full_finetuning = False,
        )
        if not isinstance(model, PeftModel):
            print("Not yet a peft model, converting into peft model")
            model = UnslothFastLanguageModel.get_peft_model(
                model,
                r=config.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=config.lora_alpha,
                lora_dropout=0,  # Supports any, but = 0 is optimized
                bias="none",  # Supports any, but = "none" is optimized
                use_gradient_checkpointing=True,
                random_state=3407,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None,  # And LoftQ
            )
            print("Peft model created")
        return LoadedModel(model=model, tokenizer=tokenizer)

    def load_for_inference(self, checkpoint_path: str, config: TrainingPropConfig) -> LoadedModel:
        model, tokenizer = UnslothFastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
        )
        UnslothFastLanguageModel.for_inference(model)
        return LoadedModel(model=model, tokenizer=tokenizer)
    
    def get_formatting_prompts_func(self, config, model, tokenizer) -> Callable:
        return partial(harmony_formatting_prompts_func, tokenizer=tokenizer)

FACTORY_REGISTRY: Dict[str, type[LanguageModelFactory]] = {
    "gemma": AtomGPTFactory,
    "qwen": AtomGPTFactory,
    "Meta": AtomGPTFactory,
    "Llama": AtomGPTFactory,
    "llama": AtomGPTFactory,
    "Mistral": AtomGPTFactory,
    "mistral": AtomGPTFactory,
    "gpt-oss": GPTOSSFactory,
}

def get_lm_factory(config: TrainingPropConfig) -> LanguageModelFactory:
    model_name = config.model_name
    if "gpt-oss" in model_name:
        return GPTOSSFactory()
    else:
        return AtomGPTFactory()
