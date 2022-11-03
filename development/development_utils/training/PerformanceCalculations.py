import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from typing import List, TypeVar
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
PyTorchDataLoader = TypeVar('torch.utils.data.DataLoader')

import pandas as pd
import numpy as np
from typing import List
from rdkit import Chem

def CalculateWeightedAverage(df):
    medians = df.groupby(['COMBINED_CAS','COMBINED_Duration_Value',
       'COMBINED_effect','COMBINED_endpoint', 'SMILES','SMILES_Canonical_RDKit',
       'cmpdname'], as_index=False, dropna=False).median()
    counts = df.groupby(['COMBINED_CAS','COMBINED_Duration_Value',
       'COMBINED_effect','COMBINED_endpoint', 'SMILES','SMILES_Canonical_RDKit',
       'cmpdname'], as_index=False, dropna=False).count()
    counts.rename(columns={'labels': 'counts'}, inplace=True)

    medians['counts'] = counts['counts']
    for col in medians.columns:
        if col not in ['COMBINED_CAS','COMBINED_effect','COMBINED_endpoint', 'SMILES','SMILES_Canonical_RDKit','cmpdname']:
            medians[[str(col)]] = medians[[str(col)]]*medians[['counts']].to_numpy()

    mean = medians.groupby(['COMBINED_CAS','COMBINED_endpoint','SMILES','SMILES_Canonical_RDKit','cmpdname'], as_index=False, dropna=False).sum(min_count=1)

    for col in mean.columns:
        try:
            if col not in ['COMBINED_CAS','COMBINED_endpoint','SMILES','SMILES_Canonical_RDKit','cmpdname']:
                mean[str(col)] = mean[[str(col)]]/mean[['counts']].to_numpy()
        except:
            pass
    mean['L1error'] = abs(mean['residuals'])
    return mean.drop(columns=['counts'])
