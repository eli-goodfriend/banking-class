"""
determine the most popular 100 companies to hand categorize 
this will seed the classification algorithm
"""
import transact as ts
import pandas as pd

filein = '/home/eli/Data/Narmi/anonymized_transactions.csv'
fileout = 'lookup_table.csv'

df = pd.read_csv(filein)
df.columns = ['raw','amount']
fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' # TODO hardcode
us_cities = pd.read_pickle(fileCities)

ts.parseTransactions(df,'raw',us_cities)

counts = df.merchant.value_counts()
print counts
counts.to_csv(fileout,index=False)
    