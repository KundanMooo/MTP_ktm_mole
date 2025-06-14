import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# SMARTS patterns for functional groups
groups = {
    'Alcohol': '[OH]',
    'Carboxylic Acid': '[CX3](=O)[OX2H1]',
    'Amine (Primary)': '[NX3;H2]',
    'Amine (Secondary)': '[NX3;H1]',
    'Amine (Tertiary)': '[NX3;H0]',
    'Aldehyde': '[CX3H1](=O)[#6]',
    'Ketone': '[CX3](=O)[#6]',
    'Ester': '[#6][CX3](=O)[OX2H0][#6]',
    'Amide': '[CX3](=[OX1])[NX3H2]',
    'Nitro': '[NX3](=[OX1])([OX1])',
    'Alkene': '[CX2]=[CX2]',
    'Alkyne': '[CX1]#[CX1]',
    'Aromatic Ring': 'c1ccccc1',
    'Halogen': '[F,Cl,Br,I]',
    'Ether': '[OD2]([#6])[#6]',
    'Sulfide': '[SD2]([#6])[#6]',
    'Sulfoxide': '[SD3](=[OX1])([#6])[#6]',
    'Sulfone': '[SD4](=[OX1])(=[OX1])([#6])[#6]',
    'Thiol': '[SH]',
    'Disulfide': '[SD2]([#6])[SD2]([#6])',
    'Phosphine': '[PX3]([#6])([#6])([#6])',
    'Phosphate': '[PX4](=[OX1])([OX2H])([OX2H])([OX2H])',
    'Phosphonate': '[PX4](=[OX1])([OX2H])([OX2H])([#6])',
    'Nitrile': '[NX1]#[CX2]',
    'Isocyanate': '[NX2]=[CX2]=[OX1]',
    'Thiocyanate': '[SX2][CX2]#[NX1]',
    'Isothiocyanate': '[NX2]=[CX2]=[SX1]',
    'Imine': '[CX3]([#6])=[NX2][#6]',
    'Enamine': '[CX3]([#6])=[CX3][NX3]',
    'Enol': '[CX3]([#6])=[CX3][OX2H]',
    'Hydrazine': '[NX3][NX3]',
    'Hydrazone': '[CX3]([#6])=[NX2][NX3]',
    'Azide': '[NX3]=[NX2+]=[NX1-]',
    'Diazo': '[NX2+]#[NX1-]',
    'Epoxide': '[OX2]1[CX4][CX4]1',
    'Lactone': '[OX2]1[CX4](=[OX1])[CX4][CX4]1',
    'Lactam': '[NX3]1[CX4](=[OX1])[CX4][CX4]1',
    'Anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
    'Peroxide': '[OX2][OX2]',
    'Hydroperoxide': '[OX2H][OX2]',
    'Enone': '[CX3]([#6])=[CX3][CX3](=[OX1])',
    'Quinone': 'O=[CX3]1[CX4]=[CX4][CX3](=[OX1])[CX4]=[CX4]1',
    'Phenol': 'c1ccccc1[OH]',
    'Thiophene': 's1cccc1',
    'Furan': 'o1cccc1',
    'Pyrrole': '[nH]1cccc1',
    'Imidazole': 'n1cnc1',
    'Pyridine': 'n1ccccc1',
    'Pyrimidine': 'n1cnccc1',
    'Indole': 'c12ccccc1[nH]cc2',
    'Quinoline': 'n1cccc2ccccc12',
    'Isoquinoline': 'c1cc2ccccc2[nH]c1',
    'Purine': 'n1c2ncnc2nc1',
    'Pyrazole': 'n1nccc1',
    'Oxazole': 'o1nccc1',
    'Thiazole': 's1nccc1',
    'Triazole': 'n1nncc1',
    'Tetrazole': 'n1nnnn1',
    'Urea': '[NX3H2]C(=O)[NX3H2]',
    'Thiourea': '[NX3H2]C(=S)[NX3H2]',
    'Guanidine': '[NX3H2]C(=N[NX3H2])[NX3H2]',
    'Carbamate': '[NX3H2]C(=O)[OX2H]',
    'Thiocarbamate': '[NX3H2]C(=S)[OX2H]',
    'Sulfonamide': '[NX3H2]S(=O)(=O)[#6]',
    'Sulfonate': '[OX2H]S(=O)(=O)[#6]',
    'Sulfate': '[OX2H]S(=O)(=O)[OX2H]',
    'Sulfite': '[OX2H]S(=O)([OX2H])[OX2H]',
    'Sulfonamide Ester': '[#6]S(=O)(=O)[OX2][#6]',
    'Sulfoxide Ester': '[#6]S(=O)([#6])[#6]',
    'Sulfone Ester': '[#6]S(=O)(=O)[#6]',
    'Sulfonamide Group': '[#6]S(=O)(=O)[NX3H2]',
    'Sulfonate Amide': '[#6]S(=O)(=O)[NX3H2]',
    'Sulfonate Imide': '[#6]S(=O)(=O)[NX2H]',
    'Sulfonate Anhydride': '[#6]S(=O)(=O)[OX2][#6]S(=O)(=O)[#6]',
    'Sulfonate Halide': '[#6]S(=O)(=O)[F,Cl,Br,I]',
    'Sulfonate Nitrile': '[#6]S(=O)(=O)[NX1]#[CX2]',
    'Sulfonate Isocyanate': '[#6]S(=O)(=O)[NX2]=[CX2]=[OX1]',
    'Sulfonate Thiocyanate': '[#6]S(=O)(=O)[SX2][CX2]#[NX1]',
    'Sulfonate Isothiocyanate': '[#6]S(=O)(=O)[NX2]=[CX2]=[SX1]',
}

def detect_fgs(mol):
    """Count functional groups in a molecule"""
    return {f'fg_{name}': len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
            for name, smarts in groups.items()}


def process_sdf(sdf_path: Path, output_dir: Path) -> str:
    """Process a single SDF into Parquet and return a status message"""
    out_file = output_dir / f"{sdf_path.stem}.parquet"
    rows = []
    supplier = Chem.SDMolSupplier(str(sdf_path))
    # progress bar per molecule
    for mol in tqdm(supplier, desc=f"{sdf_path.name}", unit='mol'):
        if mol is None:
            continue
        data = {'smiles': Chem.MolToSmiles(mol)}
        data.update(mol.GetPropsAsDict())
        data.update(detect_fgs(mol))
        rows.append(data)
    if rows:
        pd.DataFrame(rows).to_parquet(out_file, index=False)
        return f"Saved {len(rows)} molecules to {out_file.name}"
    else:
        return f"No valid molecules in {sdf_path.name}" 


def main():
    if len(sys.argv) < 3:
        print("Usage: python sdf_to_csv.py <sdf_dir> <out_dir> [--workers N]")
        sys.exit(1)
    sdf_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(exist_ok=True)
    # optional workers
    workers = None
    if '--workers' in sys.argv:
        idx = sys.argv.index('--workers')
        try:
            workers = int(sys.argv[idx + 1])
        except:
            pass
    if workers is None or workers < 2:
        # sequential
        for sdf in sdf_dir.glob('*.sdf'):
            msg = process_sdf(sdf, out_dir)
            print(msg)
    else:
        # parallel file-level
        sdfs = list(sdf_dir.glob('*.sdf'))
        max_workers = workers if workers <= len(sdfs) else len(sdfs)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for msg in ex.map(lambda p: process_sdf(p, out_dir), sdfs):
                print(msg)

if __name__ == '__main__':
    main()