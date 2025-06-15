import sys
from pathlib import Path
import pandas as pd

def combine_parquets(input_dir: Path, output_file: Path):
    """Combine all .parquet files in input_dir into a single parquet at output_file."""
    parquet_files = list(input_dir.glob('*.parquet'))
    
    dfs = []
    for pf in parquet_files:
        print(f"Loading {pf.name}...")
        df = pd.read_parquet(pf)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_file, index=False)
    print(f"Combined {len(parquet_files)} files into {output_file} ({len(combined):,} rows)")

def main():
    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    combine_parquets(input_dir, output_file)

if __name__ == '__main__':
    main()