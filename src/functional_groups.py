
from rdkit import Chem

def get_smarts_patterns():
    """
    Returns:
        dict[str,str]: mapping functional group names -> SMARTS pattern
    """
    return {
        'alcohol': '[OX2H]',
        'ketone': '[CX3](=O)[#6]',
        'amine': '[NX3;H2,H1;!$(NC=O)]',
        'ester': '[CX3](=O)[OX2H0][#6]',
        'aldehyde': '[CX3H1](=O)[#6]',
        'carboxylic_acid': '[CX3](=O)[OX2H1]',
        'amide': '[CX3](=O)[NX3]',
        'phenol': '[OX2H][cX3]:[c]',
        'ether': '[OD2]([#6])[#6]',
        'nitrile': '[CX2]#N',
        'thiol': '[SX2H]',
        'sulfide': '[SX2]([#6])[#6]',
        'nitro': '[NX3+](=O)[O-]',
        'halide': '[F,Cl,Br,I]',
        'aromatic_ring': 'c1ccccc1',
    }

def count_functional_groups(mol):
    """
    Count each functional group in a molecule.
    """
    patterns = get_smarts_patterns()
    counts = {}
    for name, smarts in patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        counts[name] = len(mol.GetSubstructMatches(patt)) if patt else 0
    return counts
