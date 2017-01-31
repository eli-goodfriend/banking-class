"""
try being passive aggressive
"""

import transact as ts
import numpy as np
import matplotlib.pyplot as plt

modelname = 'transaction_passive-aggressive'
train_in = '/home/eli/Data/Narmi/train_cat.csv'
train_out = '/home/eli/Data/Narmi/train_cat.csv'
test_in = '/home/eli/Data/Narmi/test.csv'
test_out = '/home/eli/Data/Narmi/test_cat.csv'

# our best middle of the road
# TODO this should be with a separate test set, with the parameter selection
#      runs using a cv set
# seed = 42, very dependent on seed
# best accuracy = 69.25% with unknown and unknowable as separate category
# this is exactly the same as logreg
acc = ts.run_test(train_in, train_out, test_in, test_out, modelname, run_parse=False,
                  model_type='passive-aggressive',
            C=10.0, cutoff=0.50, n_feat=2**6, n_iter=5)


Cs = [0.01, 1.0, 10.0, 50., 100.]
n_feats = [2**4, 2**5, 2**6, 2**7]
n_iters = [5, 10, 50, 100]

# effect of C
acc = np.empty(len(Cs))
idx = 0
for C in Cs:
    acc[idx] = ts.run_test(train_in, train_out, test_in, test_out, modelname, run_parse=False,
                            model_type='passive-aggressive',
                           C=C, cutoff=0.50, n_feat=2**6, n_iter=5)
    idx+=1
plt.plot(Cs,acc,'ro')
plt.xscale('log')
plt.show()

# effect of n_feat
acc = np.empty(len(n_feats))
idx = 0
for n_feat in n_feats:
    acc[idx] = ts.run_test(train_in, train_out, test_in, test_out, modelname, run_parse=False,
                            model_type='passive-aggressive',
                           C=10.0, cutoff=0.50, n_feat=n_feat, n_iter=5)
    idx+=1
plt.plot(n_feats,acc,'ro')
plt.show()

# effect of n_iter
acc = np.empty(len(n_iters))
idx = 0
for n_iter in n_iters:
    acc[idx] = ts.run_test(train_in, train_out, test_in, test_out, modelname, run_parse=False,
                            model_type='passive-aggressive',
                           C=10.0, cutoff=0.50, n_feat=2**6, n_iter=n_iter)
    idx+=1
plt.plot(n_iters,acc,'ro')
plt.show()