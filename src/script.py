import os
import argparse
import multiprocessing as mp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.GraphDescriptors import BalabanJ
from tqdm import tqdm
import psutil
import time
import glob
from pathlib import Path

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
    idx, smiles, filename = args
    mol = Chem.MolFromSmiles(smiles)
    
    # Calculate molecular descriptors
    desc = {
        'Index': idx,
        'FileName': filename,
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

def process_sdf_file(sdf_file, batch_size, cores, global_idx_start):
    """Process a single SDF file and return results"""
    filename = Path(sdf_file).stem
    supplier = Chem.SDMolSupplier(sdf_file)
    
    # Count molecules in file
    mol_count = sum(1 for _ in supplier)
    
    # Reset supplier
    supplier = Chem.SDMolSupplier(sdf_file)
    
    all_results = []
    batch = []
    idx = global_idx_start
    
    # Single progress bar for this file
    pbar = tqdm(
        total=mol_count,
        desc=f"Processing {filename}",
        unit="mol",
        ncols=100,
        leave=True,
        position=0
    )
    
    for mol in supplier:
        smiles = Chem.MolToSmiles(mol)
        batch.append((idx, smiles, filename))
        idx += 1
        
        if len(batch) >= batch_size:
            # Process batch
            with mp.Pool(processes=cores) as pool:
                results = pool.map(calc_descriptors_and_fg, batch)
            
            all_results.extend(results)
            pbar.update(len(batch))
            batch = []
    
    # Process final batch
    if batch:
        with mp.Pool(processes=cores) as pool:
            results = pool.map(calc_descriptors_and_fg, batch)
        
        all_results.extend(results)
        pbar.update(len(batch))
    
    pbar.close()
    
    return all_results, idx

def main():
    parser = argparse.ArgumentParser(description='Extract RDKit molecular descriptors and functional groups from SDF files in data folder.')
    parser.add_argument('-d', '--data_dir', default='../data', help='Data directory containing SDF files')
    parser.add_argument('-o', '--output_dir', default='../output', help='Output directory for parquet files')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help='Batch size for processing')
    args = parser.parse_args()

    start_time = time.time()
    
    cpu_count = get_system_info()
    print(f"Using all {cpu_count} CPU cores for processing")
    print(f"Batch size: {args.batch_size}")
    print(f"Functional groups to analyze: {list(FUNCTIONAL_GROUPS.keys())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all SDF files
    sdf_pattern = os.path.join(args.data_dir, "*.sdf")
    sdf_files = glob.glob(sdf_pattern)
    
    if not sdf_files:
        print(f"No SDF files found in {args.data_dir}")
        return
    
    print(f"\nFound {len(sdf_files)} SDF files:")
    total_size = 0
    for sdf_file in sdf_files:
        size = os.path.getsize(sdf_file) / (1024**3)  # GB
        total_size += size
        print(f"  {Path(sdf_file).name}: {size:.2f} GB")
    
    print(f"Total size: {total_size:.2f} GB")
    print(f"\nStarting processing...\n")
    
    all_results = []
    global_idx = 1
    
    for i, sdf_file in enumerate(sdf_files, 1):
        print(f"File {i}/{len(sdf_files)}: {Path(sdf_file).name}")
        
        # Process single SDF file
        results, next_idx = process_sdf_file(sdf_file, args.batch_size, cpu_count, global_idx)
        all_results.extend(results)
        global_idx = next_idx
        
        print(f"Completed {Path(sdf_file).name}: {len(results):,} molecules processed\n")
    
    print("=" * 80)
    print("All SDF files processed. Creating final parquet file...")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Display statistics
    total_time = time.time() - start_time
    avg_molecules_per_second = len(df) / total_time
    
    print(f"\nProcessing Summary:")
    print(f"  Total molecules processed: {len(df):,}")
    print(f"  Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Average processing rate: {avg_molecules_per_second:.1f} molecules/second")
    
    # File distribution
    print(f"\nMolecules per file:")
    file_counts = df['FileName'].value_counts().sort_index()
    for filename, count in file_counts.items():
        print(f"  {filename}: {count:,} molecules")
    
    # Functional group statistics
    fg_columns = list(FUNCTIONAL_GROUPS.keys())
    print(f'\nFunctional Group Statistics:')
    for fg in fg_columns:
        total_occurrences = df[fg].sum()
        molecules_with_fg = (df[fg] > 0).sum()
        percentage = (molecules_with_fg / len(df)) * 100
        print(f'  {fg}: {total_occurrences:,} total occurrences in {molecules_with_fg:,} molecules ({percentage:.1f}%)')
    
    # Save as parquet
    output_file = os.path.join(args.output_dir, 'molecular_descriptors_and_functional_groups.parquet')
    print(f'\nSaving to parquet file: {output_file}')
    
    # Convert to parquet with compression
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file, compression='snappy')
    
    # Final file info
    parquet_size = os.path.getsize(output_file) / (1024**3)  # GB
    compression_ratio = total_size / parquet_size if parquet_size > 0 else 0
    
    print(f"\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Parquet file size: {parquet_size:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Total columns: {len(df.columns)}")
    print(f"Total rows: {len(df):,}")

if __name__ == '__main__':
    main()