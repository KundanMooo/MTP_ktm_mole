import os
import argparse
import multiprocessing as mp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.GraphDescriptors import BalabanJ
from tqdm import tqdm
import psutil

# Functional groups SMARTS patterns
FUNCTIONAL_GROUPS = {
    'Alcohol': '[#6][OX2H]',
    'Phenol': 'c[OX2H]',
    'Ether': '[#6][OX2][#6]',
    'Carboxylic_Acid': '[#6][CX3](=O)[OX2H]',
    'Ester': '[#6][CX3](=O)[OX2][#6]',
    'Amide': '[#6][CX3](=O)[NX3]',
    'Aldehyde': '[#6][CX3H1](=O)[#1]',
    'Ketone': '[#6][CX3](=O)[#6]',
    'Primary_Amine': '[#6][NX3H2]',
    'Secondary_Amine': '[#6][NX3H][#6]',
    'Tertiary_Amine': '[#6][NX3]([#6])[#6]',
    'Nitro': '[NX3](=O)=O',
    'Nitrile': '[NX1]#[CX2]',
    'Halogen': '[F,Cl,Br,I]',
    'Thiol': '[SX2H]',
    'Thioether': '[#6][SX2][#6]',
    'Sulfonamide': '[#6][SX4](=O)(=O)[NX3]'
}

def count_functional_groups(mol):
    """Count occurrences of each functional group in a molecule"""
    fg_counts = {}
    for fg_name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)
        fg_counts[fg_name] = len(matches)
    return fg_counts

def calc_descriptors_and_fg(args):
    idx, smiles = args
    mol = Chem.MolFromSmiles(smiles)
    
    # Calculate molecular descriptors
    desc = {
        'Index': idx,
        'MolWt': Descriptors.MolWt(mol),
        'ExactMolWt': Descriptors.ExactMolWt(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'LogP': MolLogP(mol),
        'MolMR': MolMR(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'RotatableBondCount': Lipinski.NumRotatableBonds(mol),
        'AromaticRingCount': rdMolDescriptors.CalcNumAromaticRings(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'RingCount': rdMolDescriptors.CalcNumRings(mol),
        'ChiralCenterCount': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        'FormalCharge': Chem.GetFormalCharge(mol),
        'MolarRefractivity': MolMR(mol),
        'HeteroatomCount': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1,6)),
        'BalabanJ': BalabanJ(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'Chi0n': rdMolDescriptors.CalcChi0n(mol),
        'Chi1n': rdMolDescriptors.CalcChi1n(mol),
        'Chi2n': rdMolDescriptors.CalcChi2n(mol),
        'Chi3n': rdMolDescriptors.CalcChi3n(mol),
        'Chi4n': rdMolDescriptors.CalcChi4n(mol),
        'Kappa1': rdMolDescriptors.CalcKappa1(mol),
        'Kappa2': rdMolDescriptors.CalcKappa2(mol),
        'Kappa3': rdMolDescriptors.CalcKappa3(mol)
    }
    
    # Add functional group counts
    fg_counts = count_functional_groups(mol)
    desc.update(fg_counts)
    
    return desc

def get_system_info():
    cpu_count = mp.cpu_count()
    logical_cores = psutil.cpu_count(logical=False)
    total_memory = psutil.virtual_memory().total / (1024**3)
    
    print(f"System Information:")
    print(f"  Logical CPU cores: {cpu_count}")
    print(f"  Physical CPU cores: {logical_cores}")
    print(f"  Total RAM: {total_memory:.1f} GB")
    
    return cpu_count

def process_batch_with_progress(batch_data, cores):
    with mp.Pool(processes=cores) as pool:
        results = list(tqdm(
            pool.imap(calc_descriptors_and_fg, batch_data),
            total=len(batch_data),
            desc="Processing batch",
            unit="mol"
        ))
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract RDKit molecular descriptors and functional groups from an SDF file.')
    parser.add_argument('-i', '--input', required=True, help='Input SDF path')
    parser.add_argument('-o', '--output', default='descriptors_with_fg.csv', help='Output CSV path')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('-c', '--cores', type=int, default=None, help='Number of CPU cores (default: all available)')
    args = parser.parse_args()

    cpu_count = get_system_info()
    
    cores_to_use = args.cores if args.cores else cpu_count
    cores_to_use = min(cores_to_use, cpu_count)
    
    print(f"Using {cores_to_use} CPU cores for processing")
    print(f"Functional groups to analyze: {list(FUNCTIONAL_GROUPS.keys())}")
    
    print(f'\nReading molecules from {args.input}...')
    
    supplier = Chem.SDMolSupplier(args.input)
    total_molecules = len([mol for mol in supplier])
    print(f'Found {total_molecules:,} molecules.')
    
    supplier = Chem.SDMolSupplier(args.input)
    all_results = []
    batch = []
    idx = 1
    
    overall_pbar = tqdm(total=total_molecules, desc="Overall progress", unit="mol")
    
    for mol in supplier:
        smiles = Chem.MolToSmiles(mol)
        batch.append((idx, smiles))
        idx += 1
        
        if len(batch) >= args.batch_size:
            print(f'\nProcessing batch {len(all_results)//args.batch_size + 1} of {(total_molecules + args.batch_size - 1)//args.batch_size}...')
            results = process_batch_with_progress(batch, cores_to_use)
            all_results.extend(results)
            overall_pbar.update(len(batch))
            batch = []
    
    if batch:
        print(f'\nProcessing final batch...')
        results = process_batch_with_progress(batch, cores_to_use)
        all_results.extend(results)
        overall_pbar.update(len(batch))
    
    overall_pbar.close()
    
    print(f'\nTotal molecules processed: {len(all_results):,}')
    df = pd.DataFrame(all_results).sort_values('Index').reset_index(drop=True)

    # Display functional group statistics
    fg_columns = list(FUNCTIONAL_GROUPS.keys())
    print(f'\nFunctional Group Statistics:')
    for fg in fg_columns:
        total_occurrences = df[fg].sum()
        molecules_with_fg = (df[fg] > 0).sum()
        print(f'  {fg}: {total_occurrences} total occurrences in {molecules_with_fg} molecules')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'\nDescriptors and functional groups saved to {args.output}')
    print(f'Total columns in CSV: {len(df.columns)}')

if __name__ == '__main__':
    main()