"""
spin up the logistic regression using the training data set
this will create an initial trained logistic regression
future iterations will refine the logistic regression and add new (words) features
the bulk of the operation is reading new transactions and refining the logreg
"""
import pandas as pd
import parse as ps
import lookupCat as lc
import extractFeatures as ef
from sklearn import linear_model
import time

"""
normally would run this, but it takes too long to run every time when debugging
fileTrans = '/home/eli/Data/Narmi/train.csv'
fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' 
fileClean = '/home/eli/Data/Narmi/train_clean.csv'

narmi_data = pd.read_csv(fileTrans)
us_cities = pd.read_pickle(fileCities)

start = time.time()
ps.parseTransactions(narmi_data,'raw',us_cities)
end = time.time()


time_per_pt = (end - start) / len(narmi_data)
print "Time per data point = " + str(time_per_pt) + " seconds."
narmi_data.to_csv(fileClean)
"""

fileClean = '/home/eli/Data/Narmi/train_clean.csv'
transData = pd.read_csv(fileClean)

lc.lookupTransactions(transData) # makes category column to hold lookups

transData.merchant = transData.merchant.str.upper() # TODO this earlier
catData = transData[transData.category >= 0]
uncatData = transData[transData.category < 0]

X = ef.extract(catData)
y = catData.category.tolist()
logreg = linear_model.LogisticRegression()
logreg.fit(X,y)
acc = logreg.score(X,y)
print "Accuracy of pre-categorized train set is " + str(acc*100) + "%"





