"""
collection of functions for learning to categorize banking transactions
TODO should this be a class?
"""
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
import pandas as pd
import numpy as np
import pickle

# --- functions for parsing the raw input ---
def throwOut(df,col,regex):
    df[col] = df[col].str.replace(regex,'',case = False,flags = re.IGNORECASE)
    
def move(df,colIn,colOut,regex):
    df[colOut] = df[colIn].str.extract(regex,flags = re.IGNORECASE,expand = True)
    throwOut(df,colIn,regex)
    
def strip(df,col):
    df[col] = df[col].str.strip()
    
def makeDesc(df,col):
    df['description'] = df[col]

def cleanData(df):
    # use regex to pull out date and time
    move(df,'description','time','([0-9][0-9]:[0-9][0-9]:[0-9][0-9])')  
    move(df,'description','date','([0-1][0-9]/[0-3][0-9])')

    # remove the phrase 'Branch Cash Withdrawal' since it is in every entry
    throwOut(df,'description','Branch Cash Withdrawal')
    
    # pull out any phone numbers
    # TODO misses 800-COMCAST
    move(df,'description','phone','([0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9])')
    
    # remove the POS designation from the front of descriptions
    strip(df,'description')
    throwOut(df,'description','^POS ')
    
def findLocations(df, state_data):
    # does it end with a country code?
    # TODO country code list, not just US
    move(df,'description','country','(US)$')
    strip(df,'description')
    
    # find the state # TODO untidy
    us_states = state_data['state']
    states = us_states.to_string(header=False, index=False)
    states = re.sub('\n','|',states)
    regex = '(' + states + ')$'
    move(df,'description','state',regex)
    strip(df,'description')
    # TODO what if the state isn't at the end?    
    # TODO misclassifies BARBRI as BARB located in RI

    
    # find if any cities in this state are present as a substring
    # if there are, save them under 'city' column
    # TODO misses Dulles airport (probably among other airport)
    # TODO misses cities that get cut off bc they are too long
    # TODO misses cities that aren't in database bc they are technically neighborhoods
    all_cities = [city for row in state_data['cities'] for city in row]
    df['city'] = ""
    regex = '(' + '|'.join(all_cities) + ')$'
    move(df,'description','city',regex)
    strip(df,'description')
    # TODO what if there's another city name in the string for whatever reason?
    # --- what if it finds a city name that's not a city in that state?
    # TODO what if the city isn't at the end?

def findMerchant(df):
    df['merchant'] = df['description']
    df.merchant = df.merchant.str.upper() # unify representation for fitting

    # clean out known initial intermediary flags
    # TODO keep this in a separate column?
    # TODO get a list of these from somewhere?
    third_parties = ['SQ \*','LEVELUP\*','PAYPAL \*','SQC\*']
    regex = '^(' + '|'.join(third_parties) + ')'
    throwOut(df,'merchant',regex)
    
    # clean out strings that are more than one whitespace unit from the left
    throwOut(df,'merchant','\s\s+.+$')
    strip(df,'merchant')
    
    # clean out the chunks of Xs that come from redacting ID numbers
    throwOut(df,'merchant','X+-?X+')
    strip(df,'merchant')
    
    # clean out strings that look like franchise numbers
    # TODO this is not correct: too broad
    throwOut(df,'merchant','[#]?([0-9]){3,999}$')
    strip(df,'merchant')
    
def parseTransactions(df,col,cities):
    """
    parseTransactions regex parser
    inputs df: pandas dataframe with unparsed transactions
           col: name of column with unparsed transactions
           cities: pandas dataframe of city names
    updates df with new parsed columns       
    """
    makeDesc(df,col) # initialize the description field

    print 'basic cleaning...'
    cleanData(df) # clean dates, times, phone numbers, and headers

    print 'finding locations...'
    findLocations(df,cities) # find locations, if applicable

    print 'finding merchants...'
    findMerchant(df) # extract merchant from transaction description
    
# --- functions for looking up known merchants ---
def lookupTransactions(transData):
    """
    categorize transactions for major retailers from a lookup table
    input transData: dataframe containing cleaned transaction data, 
                     with a merchant column
    updates transData with a 'category' column, containing
        NaN: unknown
        0: food
        1: transport
        2: retail
        TODO standardize this from a file
    """
    # TODO janky
    foodMerchants = ['pizza','safeway','food','grocer','cafe','chipotle','mc[.]donalds','deli']
    transportMerchants = ['lyft','uber','greyhound']
    retailMerchants = ['target','walmart','amazon']
    # next step healthMerchants = ['pharmacy','gym']
    # next step entertainmentMerchants = ['cinema','theater','theatre']
    
    # TODO this is also janky
    regex = '|'.join(foodMerchants)
    transData['isFood'] = transData['merchant'].str.contains(regex,flags = re.IGNORECASE)
    
    regex = '|'.join(transportMerchants)
    transData['isTransport'] = transData['merchant'].str.contains(regex,flags = re.IGNORECASE)
    
    regex = '|'.join(retailMerchants)
    transData['isRetail'] = transData['merchant'].str.contains(regex,flags = re.IGNORECASE)
    
    transData['isUnknown'] = ~(transData.isFood | transData.isTransport | transData.isRetail)
    
    # TODO janky and not generalizable
    transData['category'] = None
    transData['category'] = 1*transData.isFood + 2*transData.isTransport + 3*transData.isRetail - 1
    ll = transData.columns.get_loc('isFood')
    ul = ll+4
    transData.drop(transData.columns[ll:ul], axis=1, inplace=True)

# --- functions for extracting features for fitting ---
def extract(df):
    """
    extract features from merchant text using sklearn vectorizer
                          times using own function
                          amount data just as such
    input: df cleaned transaction dataframe
    output: X an array of features
    """
    # turn time string into number from [0,1) 
    # TODO janky
    # TODO use datetime?
    time_feature = np.zeros((len(df),1)) # TODO this is setting NaNs to 0
    idx = 0
    for index, row in df.iterrows():
        timeString = row.time
        if isinstance(timeString, basestring): # a full time stamp
            hour = float(timeString[0:2])
            minute = float(timeString[3:5])
            second = float(timeString[6:8])
            timeFeat = hour/24. + minute/(24.*60.) + second/(24.*60.*60.)
            time_feature[idx,0] = timeFeat
        idx+=1
        
    # turn amount column into an array
    amount_feature = df.amount.values
    amount_feature.shape = (len(amount_feature),1)
            
    # turn merchant strings into vectors
    # TODO from pre-categorized data or all data
    # TODO what about fuzzy matches: do that now or when defining merchants
    # TODO like make a list of merchants and if a new one is similar enough, change
    #      it to a pre-existing merchant
    # cut one: just do what sklearn tells us to do
    vectorizer = HashingVectorizer(n_features=2**5)
    
    documents = df.merchant.tolist()
    wordCounts = vectorizer.fit_transform(documents)
    X = wordCounts.toarray()
    
    # combine
    # TODO time helps a little, amount totally f's everything
    X = np.concatenate((amount_feature,time_feature,X),axis=1)
    
    return X
    
# --- driver functions ---
def cat_df(df,model,locations,new_run,run_parse):    
    if run_parse: parseTransactions(df,'raw',locations)
    
    lookupTransactions(df)
    
    catData = df[df.category >= 0]
    uncatData = df[df.category < 0]
    print str(float(len(catData))/float(len(df)) * 100.) + "% of transactions categorized with lookup."
  
    X = extract(catData) # uses hashing vectorizer
    y = catData.category.tolist()
    if new_run:
        model.partial_fit(X,y,np.unique(y))
    else:
        model.partial_fit(X,y)

    X = extract(uncatData) # uses hashing vectorizer
    uncat_pred = model.predict(X)
    uncatData.category = uncat_pred
    df = pd.concat([catData, uncatData])
    df.sort_index(inplace=True)
    
    return df

def run_cat(filename,modelname,fileout,new_run=True,run_parse=True):
    df = pd.read_csv(filename)
    
    if new_run:
        model = linear_model.SGDClassifier(loss='log') # TODO hardcode
    else:
        modelFileLoad = open(modelname, 'rb')
        model = pickle.load(modelFileLoad)
    
    fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' # TODO hardcode
    us_cities = pd.read_pickle(fileCities)
    
    df = cat_df(df,model,us_cities,new_run,run_parse)
    
    df.to_csv(fileout)
    
    # Saving logistic regression model from training set 1
    modelFileSave = open(modelname, 'wb')
    pickle.dump(model, modelFileSave)
    modelFileSave.close()

    
    
    
    
    
    
    
    



