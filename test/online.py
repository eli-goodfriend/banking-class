"""
canned version of the intended online training use
artificially create a "stream" of incoming transactions from a chopped up
    input file
then progressively train the logistic regression on each incoming dataset

this implementation uses training and test sets to see the improvement in
the logistic regression with additional data

however, the input file `filein` can be the full dataset if testing is not
performed
"""
import sys
sys.path.append("..")
import os

import cPickle as pickle
import pandas as pd
from sklearn import metrics
import transact as ts
from initial_setup import directories as dirs

modelname = dirs.run_dir + 'model_data/production_logreg'
filein = dirs.data_dir + 'train.csv'
fileout = dirs.data_dir + 'online_cat.csv'
online_dir = dirs.data_dir + '/online_csv/'

try:
    cat = embeddings['cat']
except:
    print 'Loading embeddings, this may take a few minutes...'
    embedding_name = dirs.data_dir + 'glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)

def chop_up_input(start_up_size, chunk_size):
    df = pd.read_csv(filein)
    
    if not os.path.exists(online_dir):
        os.makedirs(online_dir)
    
    num_files = int( (len(df) - start_up_size) / chunk_size )
    
    # initialize logistic regression with larger chunk
    filename = online_dir + 'online0.csv'
    df_chunk = df[0:start_up_size]
    df_chunk.columns = ['raw','amount']
    df_chunk.to_csv(filename,index=False)
    
    # run online updates with smaller chunks
    for file_num in range(1,num_files):
        filename = online_dir + 'online' + str(file_num) + '.csv'
        start = (file_num-1)*chunk_size + start_up_size
        stop = start + chunk_size
        df_chunk = df[start:stop]
        df_chunk.columns = ['raw','amount']
        df_chunk.to_csv(filename,index=False)
        
    return num_files
        
def start_up():
    filein  = online_dir + 'online0.csv'
    fileout = online_dir + 'online0_cat.csv'
    ts.run_cat(filein,modelname,fileout,embeddings,new_run=True)

def train_online(file_num):
    filein  = online_dir + 'online' + str(file_num) + '.csv'
    fileout = online_dir + 'online' + str(file_num) + '_cat.csv'
    ts.run_cat(filein,modelname,fileout,embeddings,new_run=False)
    
def test_online():
    test_in = dirs.data_dir + 'test.csv'
    test_out = dirs.data_dir + 'test_cat.csv'
    
    ts.run_cat(test_in,modelname,test_out,embeddings,new_run=False)
    
    testData = pd.read_csv(test_out)
    precision = metrics.precision_score(testData.truth, testData.category, 
                                        average='weighted')
    
    return precision
    
def run_example(start_up_size=1000, chunk_size=100, run_test=False):
    print "Dividing data into mock up streams"
    num_files = chop_up_input(start_up_size, chunk_size)
    
    print "Initializing logistic regression"
    start_up()
    
    for file_num in range(1,num_files+1):
        print "Training on file " + str(file_num) + " / " + str(num_files)
        train_online(file_num)
        
        if run_test: 
            prec = test_online()
            print "Precision = ", str(prec)
        

run_example(run_test=True)









