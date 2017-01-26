"""
automatically categorize transactions based on lookup-labeled transactions
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

# TODO should move to SQL
datafile = '/home/eli/Dropbox/Code/Narmi/categorized.csv'
transData = pd.read_csv(datafile,encoding='latin-1')

transData = transData[['merchant','date','time','isFood','isTransport','isRetail','isUnknown']]

# make target column
# TODO make this less janky, make it this way to start with
transData['category'] = 1*transData.isFood + 2*transData.isTransport + 3*transData.isRetail - 1
# TODO also do this earlier
transData.merchant = transData.merchant.str.upper()

catData = transData[~transData.isUnknown]
uncatData = transData[transData.isUnknown]

# turn time string into number from [0,1) 
# TODO do this earlier and less jankily
# TODO use datetime?
catData['timenum'] = ''
for index, row in catData.iterrows():
    timeString = row.time
    if isinstance(timeString, basestring): # a full time stamp
        hour = float(timeString[0:2])
        minute = float(timeString[3:5])
        second = float(timeString[6:8])
        timenum = hour/24. + minute/(24.*60.) + second/(24.*60.*60.)
        catData.set_value(index,'timenum',timenum)

# turn merchant strings into vectors
# TODO from pre-categorized data or all data
# TODO what about fuzzy matches: do that now or when defining merchants
# TODO like make a list of merchants and if a new one is similar enough, change
#      it to a pre-existing merchant
# cut one: just do what sklearn tells us to do
vectorizer = CountVectorizer(min_df=1)

documents = catData.merchant.tolist()
wordCounts = vectorizer.fit_transform(documents)

# change everything to arrays and fit
X = wordCounts.toarray()
y = catData.category.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred) # unsurprisingly, is very good
    
print "Accuracy of test set is " + str(acc*100) + "%"

# predict categories for the as-yet uncategorized data
#newDocuments = uncatData.merchant.tolist()
#X = vectorizer.fit_transform(newDocuments).toarray()
#newCats = logreg.predict(X)



