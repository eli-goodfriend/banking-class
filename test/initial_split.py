"""
divide initial data set into training and test data
the test data will be hand labeled to be a gold standard
that's why there's only 400 test points
"""
import sys
sys.path.append("..")

import pandas as pd
import transact as ts
import directories as dirs
from initial_setup import directories as dirs

def pre_cat(df):
    # use lookup table to jumpstart hand categorization
    fileCities = dirs.data_dir + 'cities_by_state.pickle'
    us_cities = pd.read_pickle(fileCities)
    ts.parseTransactions(df,'raw',us_cities)
    
    common_merchants = pd.read_csv(dirs.run_dir + 'model_data/lookup_table.csv')
    ts.lookupTransactions(df,common_merchants)
    
    df.drop(['description','date','time','phone','merchant','country','state','city','cat_int'],1,inplace=True)

data_in = dirs.input_file
test_out = dirs.data_dir + 'test.csv'
train_out = dirs.data_dir + 'train.csv'
cv_out = dirs.data_dir + 'cv.csv'
num_test = 400
num_cv = 800
num_val = num_test + num_cv

narmi_data = pd.read_csv(data_in)

narmi_data.columns = ['raw','amount']

narmi_data = narmi_data.sample(frac=1).reset_index(drop=True) # shuffle

test_data = narmi_data.head(num_val)
train_data = narmi_data.tail(len(narmi_data) - num_val)
cv_data = test_data.head(num_cv)
test_data = test_data.tail(num_test)

pre_cat(test_data)
pre_cat(cv_data)

test_data.to_csv(test_out,index=False)
cv_data.to_csv(cv_out,index=False)
train_data.to_csv(train_out,index=False)
