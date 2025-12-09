# factories.py

from abc import ABC, abstractmethod
from .products import LoadedModel, ChatTemplate
from .inverse_models import TrainingPropConfig
from peft import PeftModel
from .loader import FastLanguageModel as AtomGPTFastLanguageModel
from unsloth import FastLanguageModel as UnslothFastLanguageModel
from typing import Dict


class LanguageModelFactory(ABC):
    @abstractmethod
    def load_for_training(self, config: TrainingPropConfig) -> LoadedModel:
        pass

    @abstractmethod
    def load_for_inference(self, checkpoint_path: str, config: TrainingPropConfig) -> LoadedModel:
        pass

    @abstractmethod
    def create_chat_template(self, config: TrainingPropConfig) -> ChatTemplate:
        pass


class AlpacaTemplate:
    def format(self, instruction: str, user_input: str, output: str | None = None) -> str:
        if output is None:
            output = ""
        return f"### Instruction:\n{instruction}\n### Input:\n{user_input}\n### Output:\n{output}"


class HarmonyTemplate:
    def format(self, instruction: str, user_input: str, output: str | None = None) -> str:
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
            model = FastLanguageModel.get_peft_model(
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
        FastLanguageModel.for_inference(model)
        return LoadedModel(model=model, tokenizer=tokenizer))

    def create_chat_template(self, config: TrainingPropConfig) -> ChatTemplate:
        return AlpacaTemplate()


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
            model = FastLanguageModel.get_peft_model(
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
                use_gradient_checkpointing=unsloth,
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
        FastLanguageModel.for_inference(model)
        return LoadedModel(model=model, tokenizer=tokenizer)
    
    def create_chat_template(self, config: TrainingPropConfig) -> ChatTemplate:
        return HarmonyTemplate()


FACTORY_REGISTRY: Dict[str, type[LanguageModelFactory]] = {
    "gemma": AtomGPTFactory,
    "qwen": AtomGPTFactory,
    "Meta": AtomGPTFactory,
    "Llama": AtomGPTFactory,
    "llama": AtomGPTFactory,
    "Mistral": AtomGPTFactory,
    "mistral": AtomGPTFactory,
    "gpt-oss": GPTOssFactory,
}

def get_lm_factory(config: TrainingPropConfig) -> LanguageModelFactory:
    model_name = config.model_name
    factory_cls = FACTORY_REGISTRY.get(model_name.split("/", 1)[1].split("-", 1)[0])
    if factory_cls is None:
        raise ValueError(f"Unsupported model name: {model_name}. No model factory found.")
    return factory_cls()
