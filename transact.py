"""
collection of functions for learning to categorize banking transactions
TODO should this be a class?
"""
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn import metrics
from nltk import tokenize
import pandas as pd
import numpy as np
import scipy.spatial.distance as scipy_dist
import cPickle as pickle
import gensim

# --- general utility functions
def cat_to_int(df):
    cats_file = 'cats.txt' # TODO hardcode
    cats = pd.read_csv(cats_file,squeeze=True,header=None)
    
    df.cat_int = None
    for i, row in df.iterrows():
        category = row.category
        for ii, rowrow in cats.iteritems():
            if category == rowrow:
                df.set_value(i,'cat_int',ii)
                break
            
def int_to_cat(df):
    cats_file = 'cats.txt' # TODO hardcode
    cats = pd.read_csv(cats_file,squeeze=True,header=None)
    
    df.category = None
    for i, row in df.iterrows():
        cat_int = row.cat_int
        for ii, rowrow in cats.iteritems():
            if cat_int == ii:
                df.set_value(i,'category',rowrow)
                break
    
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
    third_parties = ['...\*','LEVELUP\*','PAYPAL \*']
    regex = '^(' + '|'.join(third_parties) + ')'
    throwOut(df,'merchant',regex)
    
    # clean out strings that are more than one whitespace unit from the left
    throwOut(df,'merchant','\s\s+.+$')
    strip(df,'merchant')
    
    # clean out the chunks of Xs that come from redacting ID numbers
    throwOut(df,'merchant','X+-?X+')
    
    # clean out the leftover payment IDs
    throwOut(df,'merchant','( ID:.*| PAYMENT ID:.*| PMT ID:.*)')
    strip(df,'merchant')
    
    # clean out strings that look like franchise numbers
    # TODO this is not correct: too broad
    throwOut(df,'merchant','[#]?[ ]?([0-9]){1,999}$')
    strip(df,'merchant')
    
    # clean out strings that aren't helping
    throwOut(df,'merchant','([ ]?-[ ]?|[_])')
    strip(df,'merchant')
    
    # clean out anything .com
    throwOut(df,'merchant','[.]com.*$')
    strip(df,'merchant')
    
    # clean out final single characters that also aren't helping
    throwOut(df,'merchant',' .$')
    strip(df,'merchant')
    
    # finally, if this leaves an empty merchant string, fill it with a blank space
    df.merchant = df.merchant.str.replace('^$',' ')
    
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
def lookupTransactions(df,common_merchants):
    """
    categorize transactions for major retailers from a lookup table
    """
    df['category'] = None
    
    for i, row in df.iterrows():
        merchant = row.merchant
        for ii, rowrow in common_merchants.iterrows():
            if merchant == rowrow.merchant:
                df.set_value(i,'category',rowrow.category)
                break
            
    df['cat_int'] = None
    cat_to_int(df)
    
# --- functions for extracting features for fitting ---
def make_amount_feature(df):
    # turn amount column into an array
    amount_feature = df.amount.values
    amount_feature.shape = (len(amount_feature),1)
    
    return amount_feature
    
def make_word_feature(df,embeddings):
    # use embeddings to vectorize merchant description
    # TODO first cut is averaging words in description
    #      but that might not work
    merchants = df.merchant.tolist()
    veclen = len(embeddings['food'])
    word_feature = np.zeros((len(merchants),veclen))
    for idx, merchant in enumerate(merchants):
        num_known = 0
        try:
            words = tokenize.word_tokenize(merchant)
            words = [word.lower() for word in words]
            for word in words:
                wordvec = embeddings[word]
                word_feature[idx,:] += wordvec
                num_known += 1
        except:
            pass
        word_feature[idx,:] = word_feature[idx,:] / float(max(num_known,1))
        
    return word_feature

def extract(df,embeddings,model_type='logreg'):
    amount_feature = make_amount_feature(df)
    X = make_word_feature(df,embeddings)
    
    X = np.concatenate((amount_feature,X),axis=1)
    
    X = preprocessing.normalize(X)
    
    return X

def train_model(catData,model,embeddings,model_type='logreg',new_run=False):    
    X = extract(catData,embeddings,model_type=model_type) 
    y = catData.cat_int.tolist()
    if new_run:
        model.partial_fit(X,y,np.unique(y))
    else:
        model.partial_fit(X,y)

def use_model(uncatData,model,embeddings,cutoff,model_type='logreg'):
    X = extract(uncatData,embeddings,model_type=model_type)
    if (model_type=='logreg') or (model_type=='naive-bayes'):
        probs = model.predict_proba(X) # TODO am I doing this the long way? 
        uncat_pred = np.argmax(probs,axis=1)    
        uncat_prob = np.amax(probs,axis=1)
        uncat_pred[uncat_prob<cutoff] = 3 # TODO this is 'unknown' but no one knows that
    else:
        uncat_pred = model.predict(X)
    
    uncatData.cat_int = uncat_pred
    
# --- driver functions ---
def cat_df(df,model,locations,embeddings,new_run,run_parse,cutoff=0.80,
           model_type='logreg'):    
    if run_parse: parseTransactions(df,'raw',locations)
    
    print "pre-categorizing 100 most common merchants"
    common_merchants = pd.read_csv('lookup_table.csv') # TODO
    lookupTransactions(df,common_merchants)
    
    catData = df[~df.category.isnull()]
    uncatData = df[df.category.isnull()]
    print str(float(len(catData))/float(len(df)) * 100.) + "% of transactions categorized with lookup."
    
    print "training model on known merchants"
    train_model(catData,model,embeddings,model_type=model_type,new_run=new_run)
    print "predicting remaining transactions using model"
    use_model(uncatData,model,embeddings,cutoff,model_type=model_type)

    df = pd.concat([catData, uncatData])
    df.sort_index(inplace=True)
    
    int_to_cat(df)
    
    return df

def run_cat(filename,modelname,fileout,embeddings,new_run=True,run_parse=True,
            model_type='logreg',C=1.0,
            alpha=0.0001, cutoff=0.80, n_iter=100):
    df = pd.read_csv(filename)
    
    if new_run:
        if model_type=='logreg':
            model = linear_model.SGDClassifier(loss='log',warm_start=True,
                                           n_iter=n_iter,alpha=alpha,
                                           random_state=42) # TODO debug only for seed
        elif model_type=='passive-aggressive':
            model = linear_model.PassiveAggressiveClassifier(C=C,warm_start=True,
                                                             random_state=42)
        elif model_type=='naive-bayes':
            model = naive_bayes.GaussianNB()
        else:
            raise NameError('model_type must be logreg, passive-aggressive, or naive-bayes')
    else:
        modelFileLoad = open(modelname, 'rb')
        model = pickle.load(modelFileLoad)
    
    fileCities = '/home/eli/Data/Narmi/cities_by_state.pickle' # TODO hardcode
    us_cities = pd.read_pickle(fileCities)
    
    df = cat_df(df,model,us_cities,embeddings,new_run,run_parse,cutoff=cutoff,
                model_type=model_type)
    
    df.to_csv(fileout,index=False)
    
    # Saving logistic regression model from training set 1
    modelFileSave = open(modelname, 'wb')
    pickle.dump(model, modelFileSave)
    modelFileSave.close()

    
# ------ testing functions
def run_test(train_in, train_out, test_in, test_out, modelname, embeddings, run_parse=True,
             model_type='logreg',C=1.0,
             alpha=0.0001, cutoff=0.80, n_iter=100):    
    # running the parser takes most of the time right now, so option to shut it off
    run_cat(train_in,modelname,train_out,embeddings,new_run=True,run_parse=run_parse,
            model_type=model_type,C=C,
            alpha=alpha, cutoff=cutoff, n_iter=n_iter)
    
    run_cat(test_in,modelname,test_out,embeddings,new_run=False,
            model_type=model_type,C=C,
            alpha=alpha, cutoff=cutoff, n_iter=n_iter)
    
    testData = pd.read_csv(test_out)
    precision = metrics.precision_score(testData.truth, testData.category, average='weighted')
    print "Overall precision is " + str(precision*100.) + "%"
    
    return precision
    #TODO why does the train_cat end up with so many columns?
    
    
    
    
    
    



