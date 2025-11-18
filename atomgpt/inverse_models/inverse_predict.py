from atomgpt.inverse_models.loader import FastLanguageModel
from atomgpt.inverse_models.inverse_models import TrainingPropConfig
from jarvis.db.jsonutils import loadjson, dumpjson
import os
import pprint
from atomgpt.inverse_models.utils import gen_atoms, main_spectra, load_exp_file
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
import time
from jarvis.core.atoms import ase_to_atoms

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer"
    + " Inverse Model Predictor."
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Name of the output directory",
)
parser.add_argument(
    "--pred_csv",
    default="pred_list_inverse.csv",
    help="CSV file for prediction list.",
)
parser.add_argument(
    "--intvl",
    default="0.3",
    help="XRD 2 theta bin",
)
parser.add_argument(
    "--relax",
    default="True",
    help="Relax cell or not",
)
parser.add_argument(
    "--background_subs",
    default="False",
    help="Perform background subtraction",
)
parser.add_argument(
    "--model_name",
    default=None,
    help="Name or path of model if not using config.json",
)
parser.add_argument(
    "--dat_path",
    default=None,
    help="Spectra .dat path with X and Y ",
)
parser.add_argument(
    "--formula",
    default=None,
    help="Chemical formula ",
)
parser.add_argument(
    "--prop_val",
    default=None,
    help="Property values ",
)
parser.add_argument(
    "--config_path",
    default=None,
    help="Chemical formula ",
)


def relax_atoms(
    atoms=None,
    fmax=0.05,
    nsteps=150,
    constant_volume=False,
):
    from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

    calculator = AlignnAtomwiseCalculator(path=default_path(), device="cpu")
    t1 = time.time()
    # if calculator is None:
    #  return atoms
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator

    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=constant_volume)
    # TODO: Make it work with any other optimizer
    dyn = FIRE(ase_atoms)
    dyn.run(fmax=fmax, steps=nsteps)
    en = ase_atoms.atoms.get_potential_energy()
    final_atoms = ase_to_atoms(ase_atoms.atoms)
    t2 = time.time()
    return final_atoms


def predict(
    output_dir=None,
    config_path=None,
    pred_csv="pred_list_inverse.csv",
    fname="out_inv.json",
    device="cuda",
    intvl=0.3,
    tol=0.1,
    relax=False,
    model_name=None,
    dat_path=None,
    background_subs=False,
    filename="Q4_K_M.gguf",
    formula=None,
    prop_val=None,
    dtype=None,
    max_seq_length=1058,
    load_in_4bit=None,
    verbose=True,
):
    print("config_path", config_path)
    if output_dir is not None:
        config_name = os.path.join(output_dir, "config.json")
        parent = Path(output_dir).parent
        if not os.path.exists(config_name):
            config_name = os.path.join(parent, "config.json")
        adapter = os.path.join(output_dir, "adapter_config.json")
        if os.path.exists(adapter):
            model_name = output_dir
    if config_path is not None:
        config_name = config_path
        if verbose:
            print("config used", config_name)
        temp_config = loadjson(config_name)
        if verbose:
            print("config used", temp_config)
        temp_config = TrainingPropConfig(**temp_config).dict()
        max_seq_length = temp_config["max_seq_length"]
        dtype = temp_config["dtype"]
    # temp_config = TrainingPropConfig().dict()
    temp_config = TrainingPropConfig(**temp_config).dict()
    if verbose:
        pprint.pprint(temp_config)
    if model_name is None:
        model_name = temp_config["model_name"]
    if load_in_4bit is None:
        load_in_4bit = temp_config["load_in_4bit"]

    if verbose:
        print("Model used:", model_name)
        print("config used:", config_path)
        print("formula:", formula)

    model = None
    tokenizer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, gguf_file=filename
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, gguf_file=filename
        )

    atoms_arr = []
    lines = []
    if formula is None:
        with open(pred_csv, "r") as f:
            lines = f.read().splitlines()
    else:
        if dat_path is not None:
            lines = [dat_path]
        else:
            lines = [formula]

    mem = []

    for idx, i in enumerate(lines):
        prompt = i
        if ".dat" in i or dat_path is not None:
            if dat_path is None:
                parent = Path(pred_csv).parent
                fname_csv = os.path.join(parent, i)
            else:
                fname_csv = dat_path
            _formula, x, y = load_exp_file(
                filename=fname_csv,
                intvl=intvl,
                tol=tol,
                formula=formula,
                background_subs=background_subs,
            )
            y_new_str = y
            try:
                if ".dat" in i:
                    formula = str(_formula.split("/")[-1].split(".dat")[0])
            except Exception:
                pass
            prompt = (
                "The chemical formula is "
                + formula
                + " The "
                + temp_config["prop"]
                + " is "
                + y_new_str
                + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
            )
        else:
            if formula is not None:
                prompt = (
                    "The chemical formula is "
                    + formula
                    + " The "
                    + temp_config["prop"]
                    + " is "
                    + str(prop_val)
                    + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
                )

        if verbose:
            print(f"[{idx}] prompt:", prompt.replace("\n", ","))

        info = {"prompt": prompt}
        gen_mat = None

        # --- NEW: robust error handling around generation / structure use ---
        try:
            gen_mat = gen_atoms(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                alpaca_prompt=temp_config["alpaca_prompt"],
                instruction=temp_config["instruction"],
                device=device,
            )

            if verbose:
                print(f"[{idx}] gen atoms:", gen_mat)
                # spacegroup() can fail for broken structures, so guard it
                try:
                    print(f"[{idx}] gen atoms spacegroup:", gen_mat.spacegroup())
                except Exception as e_sg:
                    print(
                        f"[WARN] Failed to compute spacegroup for sample {idx}: {e_sg}"
                    )

            if relax:
                try:
                    gen_mat = relax_atoms(atoms=gen_mat)
                    if verbose:
                        print(
                            f"[{idx}] gen atoms relax:",
                            gen_mat,
                            gen_mat.spacegroup(),
                        )
                except Exception as e_relax:
                    print(
                        f"[WARN] Relaxation failed for sample {idx}, "
                        "continuing with unrelaxed structure."
                    )
                    print(traceback.format_exc())

            # this is another common crash point if gen_mat is invalid
            atoms_dict = gen_mat.to_dict()
            atoms_arr.append(atoms_dict)
            info["atoms"] = atoms_dict

        except Exception as e:
            print(
                f"[ERROR] Failed to generate a valid structure for sample {idx} "
                f"(input: {i}): {e}"
            )
            # optional: print full traceback for debugging
            print(traceback.format_exc())
            info["error"] = str(e)
            # do NOT re-raise; just skip this structure and move on
            mem.append(info)
            continue

        mem.append(info)

    dumpjson(data=mem, filename=fname)
    return model, tokenizer, temp_config


if __name__ == "__main__":
    # output_dir = make_id_prop()
    # output_dir="."
    args = parser.parse_args(sys.argv[1:])
    print("args.config_path", args.config_path)
    predict(
        output_dir=args.output_dir,
        pred_csv=args.pred_csv,
        intvl=float(args.intvl),
        model_name=args.model_name,
        dat_path=args.dat_path,
        formula=args.formula,
        config_path=args.config_path,
        prop_val=args.prop_val,
        background_subs=args.background_subs,
        # config_name=args.config_name,
    )
