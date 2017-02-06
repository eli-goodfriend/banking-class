"""
test lookup categorization
not automated, needs hand check
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd
from initial_setup import directories as dirs

filein = 'test_parse.csv'
fileout = 'test_lookup.csv'
df = pd.read_csv(filein)

lookup_file = dirs.run_dir + 'model_data/lookup_table.csv'
common_merchants = pd.read_csv(lookup_file)

ts.lookupTransactions(df,common_merchants)
df.to_csv(fileout,index=False)
