import os
import time
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from src.featurization import featurize

def process_sdf(file_path, label):
    suppl = Chem.SDMolSupplier(str(file_path))
    total = len(suppl)
    print(f"→ {file_path.name}: {total} molecules")

    results = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(featurize, suppl), total=total, desc=f"Featurizing {file_path.name}"):
            if result is not None:
                result['label'] = label
                result['file'] = file_path.name
                results.append(result)

    print(f"✔ Processed {len(results)} valid molecules from {file_path.name}")
    return results

def detect_label(file_name: str) -> int:
    lower_name = file_name.lower()
    if "ins" in lower_name:
        return 0  # Inactive
    elif "act" in lower_name:
        return 1  # Active
    else:
        return -1  # Unknown

def main():
    data_dir = Path("data")
    sdf_files = list(data_dir.glob("*.sdf"))
    print(f"→ {len(sdf_files)} files found; using {cpu_count()} cores\n")

    all_data = []
    start = time.time()

    for path in sdf_files:
        label = detect_label(path.name)
        if label == -1:
            print(f"⚠ Skipping unknown file: {path.name}")
            continue
        all_data.extend(process_sdf(path, label))

    df = pd.DataFrame(all_data)
    df.to_csv("molecule_features.csv", index=False)
    print(f"\n✔ Saved {df.shape[0]} rows to molecule_features.csv")
    print(f"⏱ Elapsed time: {time.time() - start:.1f}s")

if __name__ == '__main__':
    freeze_support()  # For multiprocessing on Windows
    main()
