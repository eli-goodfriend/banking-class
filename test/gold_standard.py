"""
train and test the model on a hand categorized gold standard
"""
import sys
sys.path.append("..")

import transact as ts
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from initial_setup import directories as dirs

train_in = dirs.data_dir + 'train_cat.csv' # only do parsing once
train_out = dirs.data_dir + 'train_cat.csv'
cv_in = dirs.data_dir + 'cv.csv'
cv_out = dirs.data_dir + 'cv_cat.csv'
test_in = dirs.data_dir + 'test.csv'
test_out = dirs.data_dir + 'test_cat.csv'

# glove's pretrained model, loaded into dictionary
try:
    cat = embeddings['cat']
except:
    embedding_name = dirs.data_dir + 'glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)
    
def test_alphas(alphas, modelname, cutoff=0.50, n_iter=1):
    # effect of alpha
    prec = np.empty(len(alphas))
    idx = 0
    for alpha in alphas:
        prec[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, 
                                modelname, embeddings, run_parse=False,
                                alpha=alpha, cutoff=cutoff, n_iter=n_iter)
        idx+=1
    plt.plot(alphas,prec,'ro')
    plt.xscale('log')
    plt.show()
    
def test_C(Cs, modelname, n_iter=1):
    prec = np.empty(len(Cs))
    idx = 0
    for C in Cs:
        prec[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, modelname,
                                 embeddings, run_parse=False,
                                model_type='passive-aggressive',
                               C=C, n_iter=n_iter)
        idx+=1
    plt.plot(Cs,prec,'ro')
    plt.xscale('log')
    plt.show()
    
def test_cutoffs(cutoffs, modelname, alpha=1.0, n_iter=1):
    prec = np.empty(len(cutoffs))
    idx = 0
    for cutoff in cutoffs:
        prec[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, 
                                modelname, embeddings, run_parse=False,
                               alpha=alpha, cutoff=cutoff,n_iter=n_iter)
        idx+=1
    plt.plot(cutoffs,prec,'ro')
    plt.show()
    
def test_n_iter(n_iters, modelname, alpha=1.0, C=10., cutoff=0.5):
    prec = np.empty(len(n_iters))
    idx = 0
    for n_iter in n_iters:
        prec[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, 
                                modelname, embeddings, run_parse=False,
                               alpha=alpha, C=C, cutoff=cutoff, n_iter=n_iter)
        idx+=1
    plt.plot(n_iters,prec,'ro')
    plt.show()

def test_logreg(param_search=False):
    # seed = 42, very dependent on seed TODO seed as input or loop over several
    # best accuracy = 82%
    # best precision = 90%
    # no difference between log and modified_huber losses
    # no difference between averaged and not
    # no difference between l2 and elasticnet
    modelname = dirs.run_dir + 'model_data/transaction_logreg'

    prec = ts.run_test(train_in, train_out, test_in, test_out, modelname, 
                       embeddings, 
                      model_type='logreg',run_parse=False,
                alpha=1.0, cutoff=0.50, n_iter=1)
    
    if param_search:
        alphas = [10., 1., 0.1, 0.01, 0.001, 0.0001]
        cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        n_iters = [1, 5, 10]
        
        test_alphas(alphas, modelname, cutoff=0.5, n_iter=1)
        
        test_cutoffs(cutoffs, modelname, alpha=1.0, n_iter=1)
        
        test_n_iters(n_iters, modelname, alpha=1.0, C = 10., cutoff=0.5)
        
def test_passive_aggressive(param_search=False):
    # seed = 42, very dependent on seed
    # best accuracy = 82%
    # best precision = 85%
    # this is the same as logreg
    # no difference between the two loss functions
    modelname = dirs.run_dir + 'model_data/transaction_passive-aggressive'

    prec = ts.run_test(train_in, train_out, test_in, test_out, modelname, 
                       embeddings, run_parse=False,
                      model_type='passive-aggressive',
                      C=10., n_iter=1)
    
    if param_search:
        Cs = [0.01, 0.1, 1.0, 10., 100., 1000.]
        n_iters = [1, 5, 10]
        
        test_Cs(Cs, modelname, n_iter=1)
        
        test_n_iters(n_iters, modelname, alpha = 1.0, C = 10., cutoff = 0.5)
        
def test_naive_bayes(param_search=False):
    # best accuracy = 74%
    # best precision = 84%
    # not good!
    modelname = dirs.run_dir + 'model_data/transaction_naive-bayes'

    prec = ts.run_test(train_in, train_out, test_in, test_out, modelname, 
                       embeddings, run_parse=False,
                      model_type='naive-bayes', cutoff=0.7)
    
    if param_search:
        cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        test_cutoffs(cutoffs, modelname)

def test_all():
    test_logreg()
    test_passive_aggressive()
    test_naive_bayes()

test_logreg()



