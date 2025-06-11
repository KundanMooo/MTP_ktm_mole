#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem

# SMARTS definitions for functional groups
functional_groups = {
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
    'Sulfonate Ester': '[#6]S(=O)(=O)[OX2][#6]',
    'Sulfoxide Ester': '[#6]S(=O)([#6])[#6]',
    'Sulfone Ester': '[#6]S(=O)(=O)[#6]',
    'Sulfonamide Ester': '[#6]S(=O)(=O)[NX3H2]',
    'Sulfonate Amide': '[#6]S(=O)(=O)[NX3H2]',
    'Sulfonate Imide': '[#6]S(=O)(=O)[NX2H]',
    'Sulfonate Anhydride': '[#6]S(=O)(=O)[OX2][#6]S(=O)(=O)[#6]',
    'Sulfonate Halide': '[#6]S(=O)(=O)[F,Cl,Br,I]',
    'Sulfonate Nitrile': '[#6]S(=O)(=O)[NX1]#[CX2]',
    'Sulfonate Isocyanate': '[#6]S(=O)(=O)[NX2]=[CX2]=[OX1]',
    'Sulfonate Thiocyanate': '[#6]S(=O)(=O)[SX2][CX2]#[NX1]',
    'Sulfonate Isothiocyanate': '[#6]S(=O)(=O)[NX2]=[CX2]=[SX1]',
}

def load_molecules(sdf_path, label):
    suppl = Chem.SDMolSupplier(sdf_path)
    return [(mol, label) for mol in suppl if mol is not None]

def build_dataframe(data_with_ids, out_csv):
    # Compile SMARTS once
    smarts = {name: Chem.MolFromSmarts(patt)
              for name, patt in functional_groups.items()}

    rows = []
    for mol, label, mol_id in data_with_ids:
        feats = {'mol_id': mol_id}
        for name, patt in smarts.items():
            feats[name] = int(mol.HasSubstructMatch(patt))
        feats['target'] = int(label)
        rows.append(feats)

    df = pd.DataFrame(rows).set_index('mol_id')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv)
    print(f"✅ Saved features to {out_csv}")

def main(args):
    actives   = load_molecules(args.active_sdf,   label=1)
    inactives = load_molecules(args.inactive_sdf, label=0)
    n = min(len(actives), len(inactives))
    balanced = actives[:n] + inactives[:n]
    np.random.shuffle(balanced)
    data_with_ids = [(mol, lbl, idx) for idx, (mol, lbl) in enumerate(balanced)]
    build_dataframe(data_with_ids, args.out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--active_sdf",   required=True, help="Path to active.sdf")
    p.add_argument("--inactive_sdf", required=True, help="Path to inactive.sdf")
    p.add_argument("--out_csv",      required=True, help="Output CSV path")
    args = p.parse_args()
    main(args)
