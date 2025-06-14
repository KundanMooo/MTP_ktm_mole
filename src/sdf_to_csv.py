"""
Parallel SDF to CSV processor with clear progress tracking for each file.
Shows file name and molecule count for each SDF being processed.
"""

import os
import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from rdkit import Chem, RDLogger
import pandas as pd
from tqdm import tqdm

from src.functional_groups import count_functional_groups, get_smarts_patterns

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def count_molecules_in_sdf(sdf_path):
    """Quickly count molecules in SDF file."""
    suppl = Chem.SDMolSupplier(sdf_path)
    count = sum(1 for mol in suppl if mol is not None)
    return count

def process_sdf_file(sdf_path):
    """
    Process one SDF file and return results with progress tracking.
    """
    file_name = os.path.basename(sdf_path)
    
    # Count molecules first
    total_mols = count_molecules_in_sdf(sdf_path)
    
    if total_mols == 0:
        print(f"⚠️  {file_name}: No valid molecules found")
        return []
    
    print(f"🧬 Processing: {file_name} ({total_mols} molecules)")
    
    # Process molecules with progress bar
    suppl = Chem.SDMolSupplier(sdf_path)
    rows = []
    
    # Create progress bar with file name and molecule count
    desc = f"{file_name[:30]}..." if len(file_name) > 30 else file_name
    pbar = tqdm(
        total=total_mols,
        desc=f"📁 {desc}",
        unit="mol",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} molecules [{elapsed}<{remaining}]"
    )
    
    processed = 0
    for mol in suppl:
        if mol is not None:
            # Get all SDF properties
            props = mol.GetPropsAsDict()
            
            # Add SMILES
            props['SMILES'] = Chem.MolToSmiles(mol, isomericSmiles=True)
            
            # Add basic molecular properties
            props['num_atoms'] = mol.GetNumAtoms()
            props['num_bonds'] = mol.GetNumBonds()
            
            # Add functional group counts
            props.update(count_functional_groups(mol))
            
            # Add source file info
            props['source_file'] = file_name
            
            rows.append(props)
            processed += 1
            
        pbar.update(1)
    
    pbar.close()
    print(f"✅ Completed: {file_name} ({processed} molecules processed)")
    
    return rows

def get_file_info(data_dir):
    """Get information about all SDF files in the directory."""
    sdf_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith('.sdf')
    ])
    
    print("📊 SCANNING SDF FILES...")
    print("=" * 60)
    
    file_info = []
    total_molecules = 0
    
    for sdf_path in sdf_files:
        file_name = os.path.basename(sdf_path)
        mol_count = count_molecules_in_sdf(sdf_path)
        file_size = os.path.getsize(sdf_path) / (1024 * 1024)  # MB
        
        file_info.append({
            'path': sdf_path,
            'name': file_name,
            'molecules': mol_count,
            'size_mb': file_size
        })
        
        total_molecules += mol_count
        print(f"📄 {file_name:<40} | {mol_count:>6} molecules | {file_size:>5.1f} MB")
    
    print("=" * 60)
    print(f"📁 Total files: {len(sdf_files)}")
    print(f"🧬 Total molecules: {total_molecules:,}")
    print(f"💾 Total size: {sum(info['size_mb'] for info in file_info):.1f} MB")
    print()
    
    return file_info

def main(data_dir, output_csv, max_workers=None):
    """Main processing function."""
    
    if not os.path.isdir(data_dir):
        print(f"❌ Error: Directory '{data_dir}' does not exist!")
        return
    
    # Get file information
    file_info = get_file_info(data_dir)
    
    if not file_info:
        print(f"❌ No SDF files found in '{data_dir}'")
        return
    
    # Setup parallel processing
    n_cores = cpu_count()
    if max_workers is None:
        max_workers = min(n_cores, len(file_info))
    
    print(f"🖥️  Available CPU cores: {n_cores}")
    print(f"⚡ Using {max_workers} parallel workers")
    print(f"🎯 Processing {len(file_info)} files...")
    print()
    
    # Start parallel processing
    start_time = time.time()
    all_rows = []
    
    with Pool(processes=max_workers) as pool:
        # Submit all files for processing
        file_paths = [info['path'] for info in file_info]
        results = pool.map(process_sdf_file, file_paths)
        
        # Combine all results
        for rows in results:
            all_rows.extend(rows)
    
    processing_time = time.time() - start_time
    
    print()
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Total processing time: {processing_time:.1f} seconds")
    print(f"🧬 Total molecules processed: {len(all_rows):,}")
    print(f"⚡ Processing speed: {len(all_rows)/processing_time:.1f} molecules/second")
    
    if not all_rows:
        print("❌ No molecules were successfully processed!")
        return
    
    # Create DataFrame
    print("📊 Creating DataFrame...")
    df = pd.DataFrame(all_rows)
    
    # Organize columns
    fg_cols = list(get_smarts_patterns().keys())
    system_cols = ['SMILES', 'num_atoms', 'num_bonds', 'source_file']
    orig_props = [c for c in df.columns if c not in (fg_cols + system_cols)]
    
    # Reorder: original properties, SMILES, molecular info, functional groups, source file
    final_cols = orig_props + ['SMILES', 'num_atoms', 'num_bonds'] + fg_cols + ['source_file']
    df = df[final_cols]
    
    # Save to CSV
    print(f"💾 Saving to '{output_csv}'...")
    df.to_csv(output_csv, index=False)
    
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    
    print()
    print("✅ SUCCESS!")
    print("=" * 60)
    print(f"📄 Output file: {output_csv}")
    print(f"📊 Data shape: {len(df):,} molecules × {len(df.columns)} columns")
    print(f"💾 File size: {file_size_mb:.1f} MB")
    print(f"🧪 Functional groups analyzed: {len(fg_cols)}")
    
    # Show sample of functional group statistics
    print("\n🔬 FUNCTIONAL GROUP SUMMARY:")
    print("-" * 40)
    for fg in fg_cols[:8]:  # Show first 8 functional groups
        count = df[fg].sum()
        avg = df[fg].mean()
        print(f"{fg:<18}: {count:>6} total (avg: {avg:.2f} per molecule)")
    
    print(f"\n🎯 Data successfully exported to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert SDF files to CSV with parallel processing and clear progress tracking"
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='data',
        help="Directory containing SDF files (default: 'data')"
    )
    parser.add_argument(
        '--output', '-o',
        default='sdf_results.csv',
        help="Output CSV filename (default: 'sdf_results.csv')"
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)"
    )
    
    args = parser.parse_args()
    main(args.data_dir, args.output, args.workers)