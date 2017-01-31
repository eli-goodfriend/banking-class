"""
automatically search parameters for a set of models
to optimize accuracy in cv set
"""
import transact as ts
import numpy as np
import matplotlib.pyplot as plt

train_in = '/home/eli/Data/Narmi/train_cat.csv' # don't redo parsing
train_out = '/home/eli/Data/Narmi/train_cat.csv'
cv_in = '/home/eli/Data/Narmi/cv.csv'
cv_out = '/home/eli/Data/Narmi/cv_cat.csv'
test_in = '/home/eli/Data/Narmi/test.csv'
test_out = '/home/eli/Data/Narmi/test_cat.csv'


# logistic regression
modelname = 'transaction_logreg'
alphas = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_feats = [2**4, 2**5, 2**6, 2**7]
n_iters = [5, 10, 50, 100]

acc = np.empty((len(alphas),len(cutoffs),len(n_feats),len(n_iters)))
it = np.nditer(acc, flags=['multi_index'])
while not it.finished:
    indices = it.multi_index
    print indices
    acc[indices] = ts.run_test(train_in, train_out, cv_in, cv_out, 
                                modelname, run_parse=False,
                                alpha=alphas[indices[0]], 
                                cutoff=cutoffs[indices[1]], 
                                n_feat=n_feats[indices[2]], 
                                n_iter=n_iters[indices[3]])
    it.iternext()
    
best_idx = acc.argmax()
best_idx = np.unravel_index(best_idx,acc.shape())

print "Best alpha = " + str(alphas[best_idx[0]])





