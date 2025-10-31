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

    # NEW: drop NaN/inf, then drop anything that *rounds* to 0.00... at the chosen precision
    acts = np.where(np.isfinite(acts), acts, 0.0)
    rounded = np.round(acts, decimals=activity_decimals)
    mask = rounded != 0.0  # this also excludes "-0.00"

    if not np.any(mask):
        return None

    freqs_nz = freqs[mask]
    acts_nz = acts[mask]

    order = np.argsort(freqs_nz)
    freqs_nz = freqs_nz[order]
    acts_nz = acts_nz[order]

    fmt_f = f"{{0:.{freq_decimals}f}}"
    pairs = [
        f"{fmt_f.format(float(freq))} ({format_fixed_decimals(float(act), activity_decicals)})"
        for freq, act in zip(freqs_nz, acts_nz)
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

