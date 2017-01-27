"""
extract features using sklearn vectorizer
input: df cleaned transaction dataframe
output: X an array of features
"""
from sklearn.feature_extraction.text import HashingVectorizer

def extract(df):
    # turn time string into number from [0,1) 
    # TODO do this earlier and less jankily
    # TODO use datetime?
    # TODO not actually including this yet
#    df['timenum'] = None
#    for index, row in df.iterrows():
#        timeString = row.time
#        if isinstance(timeString, basestring): # a full time stamp
#            hour = float(timeString[0:2])
#            minute = float(timeString[3:5])
#            second = float(timeString[6:8])
#            timeFeat = hour/24. + minute/(24.*60.) + second/(24.*60.*60.)
#            df.set_value(index,'timenum',timeFeat)
            
    # turn merchant strings into vectors
    # TODO from pre-categorized data or all data
    # TODO what about fuzzy matches: do that now or when defining merchants
    # TODO like make a list of merchants and if a new one is similar enough, change
    #      it to a pre-existing merchant
    # cut one: just do what sklearn tells us to do
    vectorizer = HashingVectorizer(n_features = 2 ** 5)
    
    documents = df.merchant.tolist()
    wordCounts = vectorizer.fit_transform(documents)
    
    # change everything to arrays and fit
    X = wordCounts.toarray()
    
    return X
            
            
