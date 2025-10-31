#!/usr/bin/env python3
# make_raman_alpaca.py
#
# Read Raman JSON, keep only modes whose activity is non-zero AFTER rounding
# to the requested precision (so "0.000000" modes are dropped), format values,
# and write Alpaca-style train/test JSONs compatible with your finetuning script.

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import dumpjson


def get_crystal_string_t(atoms: Atoms) -> str:
    lengths = atoms.lattice.abc
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords
    crystal_str = (
        " ".join("{0:.2f}".format(x) for x in lengths)
        + "\n"
        + " ".join(str(int(x)) for x in angles)
        + "\n"
        + "\n".join(
            f"{t} " + " ".join("{0:.3f}".format(x) for x in c)
            for t, c in zip(atom_ids, frac_coords)
        )
    )
    return crystal_str


def format_fixed_decimals(val: float, decimals: int = 6) -> str:
    """Format a number with fixed decimal places (handles scientific-notation inputs)."""
    try:
        v = float(val)
    except Exception:
        return "0"
    if not np.isfinite(v):
        return "0"
    return f"{v:.{decimals}f}"


def make_raman_record(
    entry: dict,
    freq_decimals: int = 2,
    activity_decimals: int = 6,
) -> dict | None:
    atoms_dict = entry.get("atoms")
    if not atoms_dict:
        return None

    try:
        atoms = Atoms.from_dict(atoms_dict)
    except Exception:
        return None

    try:
        formula = atoms.composition.reduced_formula
    except Exception:
        formula = "Unknown"

    # Coerce to float; handles numbers or strings like "7.88E-7"
    freqs = np.array(entry.get("freq_cm", []), dtype=float)
    acts = np.array(entry.get("raman_activity", []), dtype=float)

    if freqs.size == 0 or acts.size == 0 or freqs.size != acts.size:
        return None

    # Drop non-finite, then exclude anything that *appears* as 0.00... after rounding
    acts = np.where(np.isfinite(acts), acts, 0.0)
    acts_rounded = np.round(acts, decimals=activity_decimals)
    keep_mask = acts_rounded != 0.0  # also drops "-0.0"

    if not np.any(keep_mask):
        return None

    freqs_kept = freqs[keep_mask]
    acts_kept = acts[keep_mask]
    acts_rounded_kept = acts_rounded[keep_mask]

    # Sort by frequency
    order = np.argsort(freqs_kept)
    freqs_kept = freqs_kept[order]
    acts_kept = acts_kept[order]
    acts_rounded_kept = acts_rounded_kept[order]

    # Format output strings
    fmt_f = f"{{0:.{freq_decimals}f}}"
    pairs = [
        f"{fmt_f.format(float(freq))} ({format_fixed_decimals(float(act_r), activity_decimals)})"
        for freq, act_r in zip(freqs_kept, acts_rounded_kept)
    ]
    raman_text = ", ".join(pairs)

    rec = {
        "instruction": "Below is a description of a material.",
        "input": (
            f"The chemical formula is: {formula}.\n"
            f"The Raman spectrum shows active modes in cm^-1 with normalized intensities () at: {raman_text}.\n"
            f"Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
        ),
        "output": get_crystal_string_t(atoms),
        "id": entry.get("id", "na"),
        "raman_text": raman_text,
    }
    return rec


def main():
    p = argparse.ArgumentParser(
        description="Build Alpaca train/test JSONs from a Raman spectroscopy dataset."
    )
    p.add_argument("--raman-json", type=Path, required=True,
                   help="Path to Raman JSON file (list of entries).")
    p.add_argument("--test-ratio", type=float, default=0.1,
                   help="Fraction for test split (default: 0.10).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the split (default: 42).")
    p.add_argument("--train-out", type=Path, default=Path("alpaca_prop_train.json"),
                   help="Output path for train JSON.")
    p.add_argument("--test-out", type=Path, default=Path("alpaca_prop_test.json"),
                   help="Output path for test JSON.")
    p.add_argument("--freq-decimals", type=int, default=2,
                   help="Decimals for frequencies in cm^-1 (default: 2).")
    p.add_argument("--activity-decimals", type=int, default=6,
                   help="Decimals for Raman activities (default: 6).")
    args = p.parse_args()

    with args.raman_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for entry in tqdm(raw, total=len(raw), desc="Processing Raman entries"):
        rec = make_raman_record(
            entry,
            freq_decimals=args.freq_decimals,
            activity_decimals=args.activity_decimals,
        )
        if rec is not None:
            records.append(rec)

    if not records:
        raise SystemExit("No valid records with nonzero Raman activity (after rounding) were found.")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_total = len(records)
    n_test = max(1, int(round(args.test_ratio * n_total)))
    test = records[:n_test]
    train = records[n_test:]

    dumpjson(data=train, filename=str(args.train_out))
    dumpjson(data=test, filename=str(args.test_out))

    print(f"Wrote {len(train)} train records → {args.train_out}")
    print(f"Wrote {len(test)}  test records → {args.test_out}")

    # Quick compatibility check for the evaluator (needs id/input/output)
    ex = test[0]
    for k in ("id", "instruction", "input", "output"):
        if k not in ex:
            print(f"WARNING: key '{k}' missing from test example!")


if __name__ == "__main__":
    main()

