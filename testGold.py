"""
test the model on a hand categorized gold standard
"""
import pickle
import pandas as pd
import transact as ts
from sklearn import linear_model
from sklearn import metrics
import numpy as np


modelName = 'transaction_logreg'
testFile = '/home/eli/Data/Narmi/test.csv'
testData = pd.read_csv(testFile)

modelFileLoad = open(modelName, 'rb')
logreg = pickle.load(modelFileLoad)

fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' 
us_cities = pd.read_pickle(fileCities)

ts.parseTransactions(testData,'raw',us_cities)

ts.lookupTransactions(testData) # makes category column to hold lookups

testData.merchant = testData.merchant.str.upper() # TODO this earlier
testData.loc[testData.truth=='food','truth'] = 0 # TODO messed this up
testData.loc[testData.truth=='transportation','truth'] = 1
testData.loc[testData.truth=='retail','truth'] = 2
testData.loc[testData.truth=='unknown','truth'] = -1
testData.truth = testData.truth.astype(np.int64)
catData = testData[testData.category >= 0]
uncatData = testData[testData.category < 0]
print str(float(len(catData))/float(len(testData)) * 100.) + "% of transactions categorized with lookup."

X = ts.extract(uncatData) # uses hashing vectorizer
uncat_pred = logreg.predict(X)
uncatData.category = uncat_pred
testData = pd.concat([catData, uncatData])
testData.sort_index(inplace=True)

y = uncatData.truth.tolist()
acc = logreg.score(X,y)
print "Accuracy of logistic regression is " + str(acc*100.) + "%"

acc = metrics.accuracy_score(testData.truth, testData.category)
print "Overall accuracy is " + str(acc*100.) + "%"





