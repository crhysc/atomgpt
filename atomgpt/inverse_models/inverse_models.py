from typing import Optional
from typing import Dict
from typing import Literal
from atomgpt.inverse_models.factories import get_lm_factory

from atomgpt.inverse_models.callbacks import (
    PrintGPUUsageCallback,
    ExampleTrainerCallback,
)
from transformers import (
    TrainingArguments,
)
import torch
from atomgpt.inverse_models.utils import (
    gen_atoms,
    text2atoms,
    get_crystal_string_t,
    get_figlet,
)
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
from datasets import load_dataset
from functools import partial
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
from tqdm import tqdm
import pprint
from jarvis.io.vasp.inputs import Poscar
import csv
import os
import numpy as np
from pydantic_settings import BaseSettings
import sys
import json
import argparse
import time
from jarvis.core.composition import Composition
import traceback
from atomgpt.inverse_models.dataset_utils import make_alpaca_json

# from atomgpt.inverse_models.custom_trainer import CustomSFTTrainer

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer."
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)


# Adapted from https://github.com/unslothai/unsloth
class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""

    id_prop_path: Optional[str] = "atomgpt/examples/inverse_model/id_prop.csv"
    prefix: str = "atomgpt_run"
    model_name: str = "knc6/atomgpt_mistral_tc_supercon"
    batch_size: int = 2
    num_epochs: int = 2
    logging_steps: int = 1
    dataset_num_proc: int = 2
    seed_val: int = 3407
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train: Optional[int] = None
    num_test: Optional[int] = None
    test_ratio: Optional[float] = 0.1
    val_ratio: Optional[float] = 0.1
    model_save_path: str = "atomgpt_lora_model"
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    loss_type: str = "default"
    optim: str = "adamw_8bit"
    id_tag: str = "id"
    lr_scheduler_type: str = "linear"
    separator: str = ","
    prop: str = "Tc_supercon"
    output_dir: str = "outputs"
    csv_out: str = "AI-AtomGen-prop-dft_3d-test-rmse.csv"
    chem_info: Literal["none", "formula", "element_list", "element_dict"] = (
        "formula"
    )
    file_format: Literal["poscar", "xyz", "pdb"] = "poscar"
    save_strategy: Literal["epoch", "steps", "no"] = "steps"
    save_steps: int = 2
    callback_samples: int = 2
    max_seq_length: int = (
        2048  # Choose any! We auto support RoPE Scaling internally!
    )
    dtype: Optional[str] = None
    # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit: bool = True
    # True  # Use 4bit quantization to reduce memory usage. Can be False.
    instruction: str = "Below is a description of a superconductor material."
    alpaca_prompt: str = (
        "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
    )
    output_prompt: str = (
        " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )
    # num_val: Optional[int] = 2
    hp_cfg_path: Optional[str] = "hp_search_config.json"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 3
    warmup_ratio: float = 0.0
    logging_steps: int = 10


def load_model(path="", config=None):
    if config is None:
        config_file = os.path.join(path, "config.json")
        config = loadjson(config_file)
        config = TrainingPropConfig(**config)
        pprint.pprint(config.dict())
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer, config

def _validate_atoms(atoms):
    if atoms is None:
        return False, "atoms_is_none"
    try:
        lat = np.asarray(getattr(atoms, "lattice_mat", None), dtype=float)
        if lat.shape != (3, 3):
            return False, f"bad_lattice_shape:{getattr(atoms,'lattice_mat',None)}"
        if not np.isfinite(lat).all():
            return False, "nonfinite_lattice"
        n = getattr(atoms, "num_atoms", None)
        if n is None or n <= 0:
            return False, f"num_atoms_invalid:{n}"
        _ = Poscar(atoms).to_string()
        return True, ""
    except Exception as e:
        return False, f"poscar_fail:{type(e).__name__}:{e}"

def _poscar_one_line(at):
    return Poscar(at).to_string().replace("\n", "\\n")

def _misses_path(csv_out, config):
    fname = getattr(config, "miss_csv", None)
    if fname is None or not str(fname).strip():
        root, ext = os.path.splitext(csv_out)
        fname = root + ".misses.csv"
    os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
    return fname

def evaluate(
    test_set=[],
    model="",
    tokenizer="",
    csv_out="out.csv",
    config="",
):
    print("Testing\n", len(test_set))
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    miss_csv_out = _misses_path(csv_out, config)

    with open(csv_out, "w", newline="") as f_ok, open(miss_csv_out, "w", newline="") as f_miss:
        ok_writer = csv.writer(f_ok)
        miss_writer = csv.writer(f_miss)
        ok_writer.writerow(["id", "target", "prediction"])
        miss_writer.writerow(["id", "stage", "error", "detail", "raw_text_preview"])

        for i in tqdm(test_set, total=len(test_set)):
            sample_id = i.get("id", "")
            target_mat = None
            target_err = None
            try:
                target_mat = text2atoms("\n" + i["output"])
                if os.environ.get("PRINT_STRUCTURES"):
                    print(f"Target Structure ({sample_id}):")
                    print(target_mat)

                ok, detail = _validate_atoms(target_mat)
                if not ok:
                    target_err = detail
            except Exception as e:
                target_err = f"text2atoms:{type(e).__name__}:{e}"
                if os.environ.get("PRINT_STRUCTURES"):
                    print(f"Target Structure ({sample_id}) FAILED: {target_err}")

            if target_err:
                miss_writer.writerow([sample_id, "target", "invalid_target", target_err, (i.get("output","")[:240])])
                continue

            gen_mat = None
            gen_err = None
            raw_response = ""
            try:
                gen_mat, raw_response = gen_atoms(
                    prompt=i["input"],
                    tokenizer=tokenizer,
                    model=model,
                    alpaca_prompt=config.alpaca_prompt,
                    instruction=config.instruction,
                )
                if os.environ.get("PRINT_STRUCTURES"):
                    print(f"Predicted Structure ({sample_id}):")
                    print(gen_mat)

                ok, detail = _validate_atoms(gen_mat)
                if not ok:
                    gen_err = detail
            except Exception as e:
                gen_err = f"gen_atoms:{type(e).__name__}:{e}"
                if os.environ.get("PRINT_STRUCTURES"):
                    print(f"Predicted Structure ({sample_id}) FAILED: {gen_err}")
                    print(traceback.format_exc())
                    print(f"Raw LLM Output ({sample_id}):")
                    print(raw_response)

            if gen_err:
                miss_writer.writerow([sample_id, "prediction", "invalid_prediction", gen_err, ""])
                continue

            try:
                ok_writer.writerow([
                    sample_id,
                    _poscar_one_line(target_mat),
                    _poscar_one_line(gen_mat),
                ])
            except Exception as e:
                miss_writer.writerow([sample_id, "write", "write_failed", f"{type(e).__name__}:{e}", ""])



def batch_evaluate(
    test_set=[],
    prompts=[],
    model="",
    tokenizer="",
    csv_out="out.csv",
    config="",
    batch_size=None,
):
    gen_atoms = []
    f = open(csv_out, "w")
    if not prompts:
        target_exists = True
        prompts = [i["input"] for i in test_set]
        ids = [i["id"] for i in test_set]
    else:
        target_exists = False
        ids = ["id-" + str(i) for i in range(len(prompts))]
    print("Testing\n", len(prompts))
    if batch_size is None:
        batch_size = len(prompts)
    outputs_decoded = []
    for batch_start in tqdm(range(0, len(prompts), batch_size)):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        # print("batch_prompts",batch_prompts)
        # Tokenize and prepare inputs
        inputs = tokenizer(
            [
                config.alpaca_prompt.format(config.instruction, msg, "")
                for msg in batch_prompts
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
        ).to("cuda")

        # Generate outputs using the model
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_seq_length,
            use_cache=True,
        )

        # Decode outputs
        outputs_decoded_temp = tokenizer.batch_decode(outputs)
        # print('outputs_decoded_temp',outputs_decoded_temp)
        for output in outputs_decoded_temp:
            outputs_decoded.append(
                output.replace("<unk>", "")
                .split("### Output:")[1]
                .strip("</s>")
            )

    # print("outputs_decoded", outputs_decoded)
    f.write("id,target,prediction\n")

    for ii, i in tqdm(enumerate(outputs_decoded), total=len(outputs_decoded)):
        try:
            # print("outputs_decoded[ii]",i)
            atoms = text2atoms(i)
            gen_mat = Poscar(atoms).to_string().replace("\n", "\\n")
            gen_atoms.append(atoms.to_dict())
            if target_exists:
                target_mat = (
                    Poscar(text2atoms("\n" + i["output"]))
                    .to_string()
                    .replace("\n", "\\n")
                )
            else:
                target_mat = ""
            # print("target_mat", target_mat)
            # print("genmat", gen_mat)
            line = ids[ii] + "," + target_mat + "," + gen_mat + "\n"
            f.write(line)
            # print()
        except Exception as exp:
            print("Error", exp)
            pass
    f.close()
    return gen_atoms


def main(config_file=None):
    if config_file is None:
        args = parser.parse_args(sys.argv[1:])
        config_file = args.config_name
    if not torch.cuda.is_available():
        raise ValueError("Currently model training is possible with GPU only.")
    figlet = get_figlet()
    print(figlet)
    t1 = time.time()
    print("config_file", config_file)
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    pprint.pprint(config.dict())
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    f = open(os.path.join(config.model_save_path, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    id_prop_path = config.id_prop_path
    run_path = os.path.dirname(id_prop_path)
    num_train = config.num_train
    num_test = config.num_test
    callback_samples = config.callback_samples
    with open(id_prop_path, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]
    if not num_train:
        num_test = int(len(dt) * config.test_ratio)
        num_train = len(dt) - num_test
    dat = []
    ids = []
    for i in tqdm(dt, total=len(dt)):
        info = {}
        info["id"] = i[0]
        ids.append(i[0])
        tmp = [j for j in i[1:]]
        # tmp = [float(j) for j in i[1:]]
        # print("tmp", tmp)
        if len(tmp) == 1:
            tmp = str(float(tmp[0]))
        else:
            tmp = config.separator.join(map(str, tmp))

        # if ";" in i[1]:
        #    tmp = "\n".join([str(round(float(j), 2)) for j in i[1].split(";")])
        # else:
        #    tmp = str(round(float(i[1]), 3))
        info[config.prop] = (
            tmp  # float(i[1])  # [float(j) for j in i[1:]]  # float(i[1]
        )
        pth = os.path.join(run_path, info["id"])
        if config.file_format == "poscar":
            atoms = Atoms.from_poscar(pth)
        elif config.file_format == "xyz":
            atoms = Atoms.from_xyz(pth)
        elif config.file_format == "cif":
            atoms = Atoms.from_cif(pth)
        elif config.file_format == "pdb":
            # not tested well
            atoms = Atoms.from_pdb(pth)
        info["atoms"] = atoms.to_dict()
        dat.append(info)

    train_ids = ids[0:num_train]
    print("num_train", num_train)
    print("num_test", num_test)
    test_ids = ids[num_train : num_train + num_test]

    alpaca_prop_train_filename = os.path.join(
        config.output_dir, "alpaca_prop_train.json"
    )
    if not os.path.exists(alpaca_prop_train_filename):
        m_train = make_alpaca_json(
            dataset=dat,
            jids=train_ids,
            config=config,
            # prop=config.property_name,
            # instruction=config.instruction,
            # chem_info=config.chem_info,
            # output_prompt=config.output_prompt,
        )
        dumpjson(data=m_train, filename=alpaca_prop_train_filename)
    else:
        print(alpaca_prop_train_filename, " exists")
        m_train = loadjson(alpaca_prop_train_filename)
    print("Sample:\n", m_train[0])

    alpaca_prop_test_filename = os.path.join(
        config.output_dir, "alpaca_prop_test.json"
    )
    if not os.path.exists(alpaca_prop_test_filename):

        m_test = make_alpaca_json(
            dataset=dat,
            jids=test_ids,
            config=config,
            # prop="prop",
            include_jid=True,
            # instruction=config.instruction,
            # chem_info=config.chem_info,
            # output_prompt=config.output_prompt,
        )
        dumpjson(data=m_test, filename=alpaca_prop_test_filename)
    else:
        print(alpaca_prop_test_filename, "exists")
        m_test = loadjson(alpaca_prop_test_filename)

    factory = get_lm_factory(config)
    loaded: LoadedModel = factory.load_for_training(config)
    model, tokenizer = loaded.model, loaded.tokenizer
    formatting_prompts_func = factory.get_formatting_prompts_func(config, model, tokenizer)

    train_dataset = load_dataset(
        "json",
        data_files=alpaca_prop_train_filename,
        split="train",
        # "json", data_files="alpaca_prop_train.json", split="train"
    )
    eval_dataset = load_dataset(
        "json",
        data_files=alpaca_prop_test_filename,
        split="train",
        # "json", data_files="alpaca_prop_train.json", split="train"
    )

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
        )

    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=config.dataset_num_proc
    )
    eval_dataset = eval_dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=config.dataset_num_proc
    )
    # Compute the actual max sequence length in raw text
    lengths = [
        len(tokenizer(example["text"], truncation=False)["input_ids"])
        for example in eval_dataset
    ]
    max_seq_length = max(lengths)
    print(f"🧠 Suggested max_seq_length based on dataset: {max_seq_length}")

    tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=config.dataset_num_proc)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, num_proc=config.dataset_num_proc)
    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "output"]
    )
    tokenized_eval.set_format(
        type="torch", columns=["input_ids", "attention_mask", "output"]
    )

    """
    trainer = SFTTrainer(
        # trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        # train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        # loss_type=config.loss_type,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=5,
            overwrite_output_dir=True,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            # max_steps = 60,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config.logging_steps,
            optim=config.optim,
            weight_decay=0.01,
            lr_scheduler_type=config.lr_scheduler_type,  # "linear",
            seed=config.seed_val,
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            report_to="none",
        ),
    )
    """

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            overwrite_output_dir=True,
            warmup_ratio=config.warmup_ratio,
            # max_steps=60,
            logging_steps=config.logging_steps,
            output_dir=config.output_dir,
            optim=config.optim,
            seed=config.seed_val,
            num_train_epochs=config.num_epochs,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            disable_tqdm=False,
            log_level="info", 
            ),
    )
    if callback_samples > 0:
        callback = ExampleTrainerCallback(
            some_tokenized_dataset=tokenized_eval,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            callback_samples=callback_samples,
        )
        trainer.add_callback(callback)
    
    gpu_usage = PrintGPUUsageCallback()
    trainer.add_callback(gpu_usage)
    trainer_stats = trainer.train()
    trainer.save_model(config.model_save_path)

    model = trainer.model
    FastLanguageModel.for_inference(model)
    evaluate(
        test_set=m_test,
        model=model,
        tokenizer=tokenizer,
        csv_out=config.csv_out,
        config=config,
    )
    t2 = time.time()
    print("Time taken:", t2 - t1)


if __name__ == "__main__":
    # output_dir = make_id_prop()
    # output_dir="."
    args = parser.parse_args(sys.argv[1:])
    main(config_file=args.config_name)
    #    config_file="config.json"
    # )
    # x=load_model(path="/wrk/knc6/Software/atomgpt_opt/atomgpt/lora_model_m/")
