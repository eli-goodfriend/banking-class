"""
train and test the model on a hand categorized gold standard
"""
import transact as ts
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

modelname = 'transaction_logreg'
filein = '/home/eli/Data/Narmi/train_cat.csv'
fileout = '/home/eli/Data/Narmi/train_cat.csv'

# running the parser takes most of the time right now, so option to shut it off
ts.run_cat(filein,modelname,fileout,new_run=True,run_parse=False)

modelname = 'transaction_logreg'
filein = '/home/eli/Data/Narmi/test.csv'
fileout = '/home/eli/Data/Narmi/test_cat.csv'

ts.run_cat(filein,modelname,fileout,new_run=False)

testData = pd.read_csv(fileout)
testData.loc[testData.truth=='food','truth'] = 0 # TODO messed this up
testData.loc[testData.truth=='transportation','truth'] = 1
testData.loc[testData.truth=='retail','truth'] = 2
testData.loc[testData.truth=='unknown','truth'] = -1
testData.truth = testData.truth.astype(np.int64)

acc = metrics.accuracy_score(testData.truth, testData.category)
print "Overall accuracy is " + str(acc*100.) + "%"







