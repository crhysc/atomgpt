# This is an updated version of diffractgpt using find_peaks algorithm
import numpy as np
from scipy.signal import find_peaks
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from tqdm import tqdm
from jarvis.db.jsonutils import dumpjson
def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc  # structure.lattice.parameters[:3]
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

    # crystal_str = atoms_describer(atoms) + "\n*\n" + crystal_str
    return crystal_str


def gaussian_recast(x_original=[], y_original=[], x_new=[], sigma=.1):
    y_new = np.zeros_like(x_new, dtype=np.float64)
    for x0, amp in zip(x_original, y_original):
        y_new += amp * np.exp(-0.5 * ((x_new - x0) / sigma) ** 2)
    x_new=np.array(x_new)
    y_new=np.array(y_new)
    return x_new, y_new

def make_diffractgpt_prompt(atoms, jid='na',thetas=[0, 90], num_peaks=20):
    """Reads 2θ–intensity data, extracts top N peaks, and builds a prompt for DiffractGPT."""
    two_theta, d, intensity = XRD(thetas=thetas).simulate(atoms=atoms)
    intensity = np.array(intensity)
    intensity /= intensity.max()

    two_theta = np.array(two_theta)
    x_new = np.arange(0, 90, .1)
    two_theta,intensity = gaussian_recast(x_original=two_theta,y_original=intensity,x_new=x_new)
    #print("two_theta",two_theta)
    #print("intensity",intensity)
    intensity /= intensity.max()
    #print(intensity,max(intensity),len(intensity))

    # Find all peaks (with minimal height threshold to ignore noise)
    peaks, props = find_peaks(intensity, height=0.01, distance=1,prominence=0.05)
    #print("peaks",peaks)
    # Get top N peaks by intensity
    top_indices = np.argsort(props['peak_heights'])[::-1][:num_peaks]
    top_peaks = peaks[top_indices]
    top_peaks_sorted = top_peaks[np.argsort(two_theta[top_peaks])]

    # Create list of (2θ, intensity)
    peak_list = [(round(two_theta[p], 2), round(intensity[p], 2)) for p in top_peaks_sorted]

    # Build DiffractGPT prompt
    peak_text = ", ".join([f"{t}°({i})" for t, i in peak_list])
    print(jid,peak_text)
    num_peaks = len(peaks)
    formula=atoms.composition.reduced_formula
    input = (
        f"The chemical formula is: {formula}.\n"
        f"The XRD pattern shows main peaks at: {peak_text}.\n"
        f"Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )
    output= get_crystal_string_t(atoms)
    info={}
    info["instruction"]= "Below is a description of a material."
    info["input"]=input
    info['id']=jid
    info['peak_text']=peak_text
    info["output"]=output
    return info

# Example usage
if __name__ == "__main__":
    #atoms = Atoms.from_poscar('POSCAR')
    jids=["JVASP-32","JVASP-15014","JVASP-1002","JVASP-107","JVASP-17957","JVASP-1151"]
    f=open('id_prop.csv','w')
    dat=data('dft_3d') #[0:num_samples]
    test=[]
    train=[]
    for i in tqdm(dat,total=len(dat)):
      if i['jid'] in jids:
        atoms=Atoms.from_dict(i['atoms'])
        prompt = make_diffractgpt_prompt(atoms, jid=i['jid'],num_peaks=20)
        filename='POSCAR-'+i['jid']
        atoms.write_poscar(filename)
        line=filename+','+prompt['peak_text']+'\n'
        f.write(line)



        #print(i['jid'],prompt)
        train.append(prompt)
        if i['jid']=="JVASP-32":
            atoms=Atoms.from_dict(i['atoms'])
            prompt = make_diffractgpt_prompt(atoms, jid=i['jid'],num_peaks=20)
            #print(i['jid'],prompt)
            test.append(prompt)
            train.append(prompt)
        if i['jid']=="JVASP-15014":
            atoms=Atoms.from_dict(i['atoms'])
            prompt = make_diffractgpt_prompt(atoms, jid=i['jid'],num_peaks=20)
            #print(i['jid'],prompt)
            test.append(prompt)
            train.append(prompt)
        if i['jid']=="JVASP-1002":
            atoms=Atoms.from_dict(i['atoms'])
            prompt = make_diffractgpt_prompt(atoms, jid=i['jid'],num_peaks=20)
            #print(i['jid'],prompt)
            test.append(prompt)
            train.append(prompt)
        if i['jid']=="JVASP-107":
            atoms=Atoms.from_dict(i['atoms'])
            prompt = make_diffractgpt_prompt(atoms, jid=i['jid'],num_peaks=20)
            #print(i['jid'],prompt)
            test.append(prompt)
            train.append(prompt)
        if i['jid']=="JVASP-17957":
            atoms=Atoms.from_dict(i['atoms'])
            prompt = make_diffractgpt_prompt(atoms, jid=i['jid'],num_peaks=20)
            #print(i['jid'],prompt)
            test.append(prompt)
            train.append(prompt)
    f.close()
    dumpjson(data=train,filename='alpaca_prop_train.json')
    dumpjson(data=test,filename='alpaca_prop_test.json')
