#!/usr/bin/env python3
# make_raman_alpaca.py
#
# Read Raman JSON, optionally Niggli-reduce cells, keep only modes whose activity
# is non-zero AFTER rounding to the requested precision, optionally normalize
# frequencies to [0,1] (1.0 = max kept freq), format values, and write
# Alpaca-style train/test JSONs.

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import dumpjson


def get_crystal_string_t(atoms: Atoms) -> str:
    # Lattice
    lengths = np.array(atoms.lattice.abc, dtype=float).ravel()
    angles  = np.array(atoms.lattice.angles, dtype=float).ravel()

    # Per-site species and fractional coordinates; force shape (N, 3)
    atom_ids = [str(x) for x in list(atoms.elements)]
    frac = np.asarray(atoms.frac_coords, dtype=float)
    if frac.ndim == 1:
        if frac.size == 3:
            frac = frac.reshape(1, 3)
        else:
            raise ValueError(f"Unexpected fractional coord shape: {frac.shape}")
    elif frac.ndim == 2 and frac.shape[1] != 3:
        raise ValueError(f"Expected frac coords with 3 columns, got {frac.shape}")

    # If species length doesn't match coords, broadcast a single species tag
    if len(atom_ids) != len(frac):
        if len(atom_ids) == 1 and len(frac) > 1:
            atom_ids = atom_ids * len(frac)
        else:
            raise ValueError(
                f"Elements length ({len(atom_ids)}) != coords length ({len(frac)})"
            )

    lengths_str = " ".join(f"{x:.2f}" for x in lengths.tolist())
    angles_str  = " ".join(f"{x:.2f}" for x in angles.tolist())
    coords_str  = "\n".join(
        f"{t} " + " ".join(f"{c:.3f}" for c in row.tolist())
        for t, row in zip(atom_ids, frac)
    )
    return f"{lengths_str}\n{angles_str}\n{coords_str}"


def niggli_reduce_atoms(atoms: Atoms) -> Atoms:
    """
    Try to Niggli-reduce using pymatgen (preferred).
    Falls back to returning the original atoms if reduction fails or pymatgen is absent.
    """
    try:
        from pymatgen.core import Structure, Lattice  # lazy import
        species = list(atoms.elements)  # per-site symbols
        frac = np.array(atoms.frac_coords, dtype=float)
        lat = np.array(atoms.lattice.matrix, dtype=float)
        pmg = Structure(Lattice(lat), species, frac, coords_are_cartesian=False)

        # Niggli reduction on the full structure (updates lattice + fractional coords)
        reduced, _ = pmg.get_reduced_structure(reduction_algo="niggli")
        return Atoms(
            lattice_mat=np.array(reduced.lattice.matrix),
            coords=np.array(reduced.frac_coords),
            elements=[str(s) for s in reduced.species],
            cartesian=False,
        )
    except Exception:
        # Best-effort fallback: return original if anything goes wrong
        return atoms


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
    normalize_freq: bool = False,
    niggli: bool = False,
    include_max_freq: bool = False,
) -> dict | None:
    atoms_dict = entry.get("atoms")
    if not atoms_dict:
        return None

    try:
        atoms = Atoms.from_dict(atoms_dict)
    except Exception:
        return None

    # Optional Niggli reduction BEFORE anything else
    if niggli:
        atoms = niggli_reduce_atoms(atoms)

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
    acts_rounded_kept = acts_rounded[keep_mask]

    # Optional normalize frequencies to [0,1], with 1.0 = max kept frequency
    max_f = float(np.max(freqs_kept)) if freqs_kept.size else 0.0
    if normalize_freq:
        if max_f > 0.0:
            freqs_display = freqs_kept / max_f  # zero maps to 0.0, max -> 1.0
        else:
            freqs_display = np.zeros_like(freqs_kept)
        freq_unit_caption = "normalized frequency 0–1"
    else:
        freqs_display = freqs_kept
        freq_unit_caption = "cm^-1"

    # Sort by *display* frequency so ordering matches what we print
    order = np.argsort(freqs_display)
    freqs_display = freqs_display[order]
    freqs_kept = freqs_kept[order]  # keep original too, in case needed later
    acts_rounded_kept = acts_rounded_kept[order]

    # Format output strings
    fmt_f = f"{{0:.{freq_decimals}f}}"
    pairs = [
        f"{fmt_f.format(float(fd))} ({format_fixed_decimals(float(act_r), activity_decimals)})"
        for fd, act_r in zip(freqs_display, acts_rounded_kept)
    ]
    raman_text = ", ".join(pairs)

    # Build prompt text
    extra_norm_line = ""
    if normalize_freq and include_max_freq and max_f > 0.0:
        extra_norm_line = (
            f"\nNormalization reference: 1.00 corresponds to "
            f"{fmt_f.format(max_f)} cm^-1."
        )

    input_header = (
        f"The chemical formula is: {formula}.\n"
        f"The Raman spectrum shows active modes in {freq_unit_caption} "
        f"with normalized intensities () at: {raman_text}.{extra_norm_line}\n"
        f"Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )

    rec = {
        "instruction": "Below is a description of a material.",
        "input": input_header,
        "output": get_crystal_string_t(atoms),
        "id": entry.get("id", "na"),
        "raman_text": raman_text,
    }
    if normalize_freq and include_max_freq:
        rec["max_freq_cm"] = float(max_f)
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
                   help="Decimals for frequencies (cm^-1 or normalized), default: 2.")
    p.add_argument("--activity-decimals", type=int, default=6,
                   help="Decimals for Raman activities (default: 6).")
    p.add_argument("--normalize-freq", action="store_true",
                   help="Normalize frequencies to [0,1]; 1.0 = max kept frequency after intensity rounding.")
    p.add_argument("--include-max-freq", action="store_true",
                   help="When used with --normalize-freq, include the unnormalized max frequency (that maps to 1.0) in the prompt.")
    p.add_argument("--niggli-reduce", action="store_true",
                   help="Apply Niggli reduction to each cell before partitioning into train/test.")
    args = p.parse_args()

    with args.raman_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for entry in tqdm(raw, total=len(raw), desc="Processing Raman entries"):
        rec = make_raman_record(
            entry,
            freq_decimals=args.freq_decimals,
            activity_decimals=args.activity_decimals,
            normalize_freq=args.normalize_freq,
            niggli=args.niggli_reduce,
            include_max_freq=args.include_max_freq,
        )
        if rec is not None:
            records.append(rec)

    if not records:
        raise SystemExit("No valid records with nonzero Raman activity (after rounding) were found.")

    # Shuffle & split AFTER optional Niggli reduction (as requested)
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

