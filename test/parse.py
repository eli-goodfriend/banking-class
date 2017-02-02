"""
test parsing
not automated, just do the parsing and see how well it worked
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd

filein = 'test_input.csv'
fileout = 'test_parse.csv'
df = pd.read_csv(filein)

fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' # TODO hardcode
us_cities = pd.read_pickle(fileCities)

ts.parseTransactions(df,'raw',us_cities)
df.to_csv(fileout,index=False)