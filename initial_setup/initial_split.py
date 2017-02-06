"""
divide initial data set into training and test data
the test data will be hand labeled to be a gold standard
that's why there's only 400 test points

TODO I'm not sure what data format I should be working in
since I was given csv, I'm sticking with csv
"""
import pandas as pd
import transact as ts

def pre_cat(df):
    # use lookup table to jumpstart hand categorization
    fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' # TODO hardcode
    us_cities = pd.read_pickle(fileCities)
    ts.parseTransactions(df,'raw',us_cities)
    
    common_merchants = pd.read_csv('lookup_table.csv') # TODO
    ts.lookupTransactions(df,common_merchants)
    
    df.drop(['description','date','time','phone','merchant','country','state','city','cat_int'],1,inplace=True)

data_in = '/home/eli/Data/Narmi/anonymized_transactions.csv'
test_out = '/home/eli/Data/Narmi/test.csv'
train_out = '/home/eli/Data/Narmi/train.csv'
cv_out = '/home/eli/Data/Narmi/cv.csv'
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
