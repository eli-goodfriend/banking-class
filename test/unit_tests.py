"""
'unit' tests
need to be checked by hand by examining csv output
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd
import cPickle as pickle
from sklearn import linear_model
from initial_setup import directories as dirs

try:
    cat = embeddings['cat']
except:
    print 'Loading embeddings, this may take a few minutes...'
    embedding_name = dirs.data_dir + 'glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)

def test_parse():
    print 'Testing parsing...'
    filein = 'test_input.csv'
    fileout = 'test_parse.csv'
    df = pd.read_csv(filein)
    
    fileCities = dirs.data_dir + 'cities_by_state.pickle'
    us_cities = pd.read_pickle(fileCities)
    
    ts.parseTransactions(df,'raw',us_cities)
    df.to_csv(fileout,index=False)
    
def test_lookup():
    print 'Testing lookup...'
    filein = 'test_parse.csv'
    fileout = 'test_lookup.csv'
    df = pd.read_csv(filein)
    
    lookup_file = dirs.run_dir + 'model_data/lookup_table.csv'
    common_merchants = pd.read_csv(lookup_file)
    
    ts.lookupTransactions(df,common_merchants)
    df.to_csv(fileout,index=False)

def test_extract():
    print 'Testing extraction...'
    filein = 'test_lookup.csv'
    df = pd.read_csv(filein)
    
    X = ts.extract(df,embeddings,model_type='None')
    
    print X.shape
    
def test_cat():
    print 'Testing categorization...'
    filein = 'test_lookup.csv'
    fileout = 'test_cat.csv'
    df = pd.read_csv(filein)
    
    model = linear_model.SGDClassifier(loss='log')
    
    catData = df[~df.category.isnull()]
    uncatData = df[df.category.isnull()]
    print str(float(len(catData))/float(len(df)) * 100.) + "% of transactions categorized with lookup."
    
    ts.train_model(catData,model,embeddings,model_type='logreg',new_run=True)
    ts.use_model(uncatData,model,embeddings,0.0,model_type='logreg')
    
    df = pd.concat([catData, uncatData])
    df.sort_index(inplace=True)
    
    df.to_csv(fileout,index=False)
    
def test_all():
    test_parse()
    test_lookup()
    test_extract()
    test_cat()
    
test_all()
    
    
    
    
    
    