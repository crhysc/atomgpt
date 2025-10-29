import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import csv
import pandas as pd

def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "\n"
        + " ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )
    return crystal_str

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raman_json_path")
    args = parser.parse_args()
    df = pd.DataFrame()
    with open(args.raman_json_path, mode="r", encoding="utf-8") as file:
        data = json.load(file)
    atoms = []
    for obj in data:
        atoms = Atoms(
            lattice_mat=data[obj]['atoms']['lattice_mat'],
            coords=data[obj]['atoms']['coords'],
            elements=data[obj]['atoms']['elements']
            )
        jid = data[obj]['id']
        intensities = data[obj]['raman_activity']
        frequencies = data[obj]['freq_cm']
        





if __name__ == '__main__':
    main()
