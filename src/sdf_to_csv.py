import os
import argparse
from multiprocessing import Pool, cpu_count
from rdkit import Chem, RDLogger
import pandas as pd
from tqdm import tqdm
from src.functional_groups import count_functional_groups, get_smarts_patterns

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def count_molecules(sdf_path):
    """Count molecules in SDF file."""
    suppl = Chem.SDMolSupplier(sdf_path)
    return sum(1 for mol in suppl if mol is not None)

def process_sdf_file(sdf_path):
    """Process one SDF file."""
    file_name = os.path.basename(sdf_path)
    total_mols = count_molecules(sdf_path)
    
    print(f"🧬 Processing: {file_name} ({total_mols} molecules)")
    
    suppl = Chem.SDMolSupplier(sdf_path)
    rows = []
    
    with tqdm(total=total_mols, desc=f"📁 {file_name}", unit="mol") as pbar:
        for mol in suppl:
            if mol is not None:
                # Get SDF properties
                props = mol.GetPropsAsDict()
                
                # Add SMILES and basic info
                props['SMILES'] = Chem.MolToSmiles(mol)
                props['num_atoms'] = mol.GetNumAtoms()
                props['num_bonds'] = mol.GetNumBonds()
                
                # Add functional groups
                props.update(count_functional_groups(mol))
                
                # Add source file
                props['source_file'] = file_name
                
                rows.append(props)
            
            pbar.update(1)
    
    print(f"✅ Completed: {file_name} ({len(rows)} molecules)")
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', default='data')
    parser.add_argument('--output', '-o', default='results.csv')
    parser.add_argument('--workers', '-w', type=int, default=None)
    args = parser.parse_args()
    
    # Find SDF files
    sdf_files = [
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith('.sdf')
    ]
    
    print(f"Found {len(sdf_files)} SDF files")
    
    # Setup parallel processing
    n_cores = cpu_count()
    workers = args.workers or min(n_cores, len(sdf_files))
    print(f"Using {workers} CPU cores")
    
    # Process files in parallel
    with Pool(workers) as pool:
        results = pool.map(process_sdf_file, sdf_files)
    
    # Combine results
    all_rows = []
    for rows in results:
        all_rows.extend(rows)
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Organize columns
    fg_cols = list(get_smarts_patterns().keys())
    system_cols = ['SMILES', 'num_atoms', 'num_bonds', 'source_file']
    orig_cols = [c for c in df.columns if c not in fg_cols + system_cols]
    
    df = df[orig_cols + ['SMILES', 'num_atoms', 'num_bonds'] + fg_cols + ['source_file']]
    
    # Save
    df.to_csv(args.output, index=False)
    
    print(f"✅ Saved {len(df)} molecules to {args.output}")
    print(f"📊 Columns: {len(df.columns)}")

if __name__ == '__main__':
    main()