import pandas as pd
import numpy as np
from typing import List, TypeVar

def CalculateWeightedAverage(df):
    medians = df.groupby(['CAS','Duration_Value',
       'effect','endpoint', 'SMILES','SMILES_Canonical_RDKit',
       'cmpdname'], as_index=False, dropna=False).median()
    counts = df.groupby(['CAS','Duration_Value',
       'effect','endpoint', 'SMILES','SMILES_Canonical_RDKit',
       'cmpdname'], as_index=False, dropna=False).count()
    counts.rename(columns={'labels': 'counts'}, inplace=True)

    medians['counts'] = counts['counts']
    for col in medians.columns:
        if col not in ['CAS','effect','endpoint', 'SMILES','SMILES_Canonical_RDKit','cmpdname','counts']:
            medians[[str(col)]] = medians[[str(col)]]*medians[['counts']].to_numpy()

    mean = medians.groupby(['CAS','endpoint','SMILES','SMILES_Canonical_RDKit','cmpdname'], as_index=False, dropna=False).sum(min_count=1)

    for col in mean.columns:
        try:
            if col not in ['CAS','endpoint','SMILES','SMILES_Canonical_RDKit','cmpdname','counts']:
                mean[str(col)] = mean[[str(col)]]/mean[['counts']].to_numpy()
        except:
            pass
    mean['L1error'] = abs(mean['residuals'])
    return mean.drop(columns=['counts'])
