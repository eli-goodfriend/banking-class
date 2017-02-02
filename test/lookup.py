"""
test lookup categorization
not automated, needs hand check
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd

filein = 'test_parse.csv'
fileout = 'test_lookup.csv'
df = pd.read_csv(filein)

common_merchants = pd.read_csv('../lookup_table.csv')

ts.lookupTransactions(df,common_merchants)
df.to_csv(fileout)
