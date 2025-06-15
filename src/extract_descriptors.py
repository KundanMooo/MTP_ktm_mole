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

def calc_descriptors(args):
    idx, smiles = args
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    
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
    return desc

def check_gpu():
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"GPU(s) detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
            return True
        else:
            print("No GPU detected")
            return False
    except ImportError:
        print("GPUtil not installed - GPU detection unavailable")
        return False

def get_system_info():
    cpu_count = mp.cpu_count()
    logical_cores = psutil.cpu_count(logical=False)
    total_memory = psutil.virtual_memory().total / (1024**3)
    
    print(f"System Information:")
    print(f"  Logical CPU cores: {cpu_count}")
    print(f"  Physical CPU cores: {logical_cores}")
    print(f"  Total RAM: {total_memory:.1f} GB")
    
    gpu_available = check_gpu()
    
    return cpu_count, gpu_available

def process_batch_with_progress(batch_data, cores):
    with mp.Pool(processes=cores) as pool:
        results = list(tqdm(
            pool.imap(calc_descriptors, batch_data),
            total=len(batch_data),
            desc="Processing batch",
            unit="mol"
        ))
    return [r for r in results if r]

def main():
    parser = argparse.ArgumentParser(description='Extract RDKit molecular descriptors from an SDF file.')
    parser.add_argument('-i', '--input', required=True, help='Input SDF path')
    parser.add_argument('-o', '--output', default='descriptors.csv', help='Output CSV path')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('-c', '--cores', type=int, default=None, help='Number of CPU cores (default: all available)')
    args = parser.parse_args()

    cpu_count, gpu_available = get_system_info()
    
    # Use all cores if not specified
    cores_to_use = args.cores if args.cores else cpu_count
    cores_to_use = min(cores_to_use, cpu_count)
    
    print(f"Using {cores_to_use} CPU cores for processing")
    
    # Note about GPU usage
    if gpu_available:
        print("Note: RDKit descriptors are CPU-based. GPU acceleration not applicable for these calculations.")
    
    print(f'\nReading molecules from {args.input}...')
    
    # First pass to count molecules
    supplier = Chem.SDMolSupplier(args.input)
    total_molecules = len([mol for mol in supplier])
    print(f'Found {total_molecules:,} molecules.')
    
    # Second pass to process
    supplier = Chem.SDMolSupplier(args.input)
    all_results = []
    batch = []
    idx = 1
    
    # Create overall progress bar
    overall_pbar = tqdm(total=total_molecules, desc="Overall progress", unit="mol")
    
    for mol in supplier:
        smiles = Chem.MolToSmiles(mol) if mol else None
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
    
    print(f'\nTotal molecules processed successfully: {len(all_results):,}')
    df = pd.DataFrame(all_results).sort_values('Index').reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'Descriptors saved to {args.output}')

if __name__ == '__main__':
    main()