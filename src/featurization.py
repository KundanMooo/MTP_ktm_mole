# src/featurization.py
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from src.functional_groups import functional_groups

def extract_properties(mol):
    props = mol.GetPropsAsDict()
    # Convert keys to strings in case they're not
    return {str(k): v for k, v in props.items()}

def featurize(mol):
    if mol is None:
        return None

    features = extract_properties(mol)

    for name, smarts in functional_groups.items():
        patt = Chem.MolFromSmarts(smarts)
        features[name] = int(mol.HasSubstructMatch(patt))

    return features
