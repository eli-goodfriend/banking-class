"""
spin up the logistic regression using the training data set
this will create an initial trained logistic regression
future iterations will refine the logistic regression and add new (words) features
the bulk of the operation is reading new transactions and refining the logreg
"""
import pandas as pd
import transact as ts
from sklearn import linear_model
import numpy as np
import time
import pickle

"""
normally would run this, but it takes too long to run every time when debugging
fileTrans = '/home/eli/Data/Narmi/train.csv'
fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' 
fileClean = '/home/eli/Data/Narmi/train_clean.csv'

narmi_data = pd.read_csv(fileTrans)
us_cities = pd.read_pickle(fileCities)

start = time.time()
ts.parseTransactions(narmi_data,'raw',us_cities)
end = time.time()


time_per_pt = (end - start) / len(narmi_data)
print "Time per data point = " + str(time_per_pt) + " seconds."
narmi_data.to_csv(fileClean)
"""

modelName = 'transaction_logreg'
fileClean = '/home/eli/Data/Narmi/train_clean.csv'
fileCat = '/home/eli/Data/Narmi/train_cat.csv'
transData = pd.read_csv(fileClean)

ts.lookupTransactions(transData) # makes category column to hold lookups

transData.merchant = transData.merchant.str.upper() # TODO this earlier
catData = transData[transData.category >= 0]
uncatData = transData[transData.category < 0]

X = ts.extract(catData) # uses hashing vectorizer
y = catData.category.tolist()
logreg = linear_model.SGDClassifier(loss='log')
logreg.partial_fit(X,y,np.unique(y))
acc = logreg.score(X,y)
print "Accuracy of pre-categorized train set is " + str(acc*100) + "%"

X = ts.extract(uncatData)
uncat_pred = logreg.predict(X)
# TODO ... do I retrain the model with a partial_fit on its new "correct" answers?

uncatData.category = uncat_pred
transData = pd.concat([catData, uncatData])
transData.sort_index(inplace=True) # for convenience of comparing to original order
transData.to_csv(fileCat)

# Saving logistic regression model from training set 1
modelFileSave = open(modelName, 'wb')
pickle.dump(logreg, modelFileSave)
modelFileSave.close()



