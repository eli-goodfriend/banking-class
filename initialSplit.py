"""
divide initial data set into training and test data
the test data will be hand labeled to be a gold standard
that's why there's only 400 test points

TODO I'm not sure what data format I should be working in
since I was given csv, I'm sticking with csv
"""
import pandas as pd

data_in = '/home/eli/Data/Narmi/anonymized_transactions.csv'
test_out = '/home/eli/Data/Narmi/test.csv'
train_out = '/home/eli/Data/Narmi/train.csv'
num_test = 400

narmi_data = pd.read_csv(data_in)

narmi_data.columns = ['raw','amount']

narmi_data = narmi_data.sample(frac=1).reset_index(drop=True) # shuffle

test_data = narmi_data.head(num_test)
train_data = narmi_data.tail(len(narmi_data) - num_test)

test_data.to_csv(test_out)
train_data.to_csv(train_out)
