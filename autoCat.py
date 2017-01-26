"""
train model on pre-categorized data
"""

import pandas as pd
from sklearn import linear_model
import extractFeatures as ef

# TODO should move to SQL
datafile = '/home/eli/Data/Narmi/train_lookupCat.csv'
transData = pd.read_csv(datafile,encoding='latin-1')

# TODO also do this earlier
transData.merchant = transData.merchant.str.upper()

catData = transData[transData.category >= 0]
uncatData = transData[transData.category < 0]

X = ef.extract(catData)
y = catData.category.tolist()

logreg = linear_model.LogisticRegression()
logreg.fit(X,y)

acc = logreg.score(X,y)
    
print "Accuracy of pre-categorized train set is " + str(acc*100) + "%"

# predict categories for the as-yet uncategorized data
#newDocuments = uncatData.merchant.tolist()
#X = vectorizer.fit_transform(newDocuments).toarray()
#newCats = logreg.predict(X)



