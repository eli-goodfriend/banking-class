"""
determine the most popular companies to hand categorize 
this will seed the classification algorithm
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd

num_merchants = 100 # adjust this if you want a larger seed

filein = '/home/eli/Data/Narmi/anonymized_transactions.csv'
fileout = '../model_data/lookup_table.csv'

df = pd.read_csv(filein)
df.columns = ['raw','amount']
fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' # TODO hardcode
us_cities = pd.read_pickle(fileCities)

ts.parseTransactions(df,'raw',us_cities)

counts = df.merchant.value_counts().head(num_merchants)
counts = counts.to_frame('count')
counts['merchant'] = counts.index
counts['category'] = None

counts.to_csv(fileout,index=False)