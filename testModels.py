"""
try being passive aggressive
"""

import transact as ts
import numpy as np
import matplotlib.pyplot as plt
from initial_setup import directories as dirs

modelname = dirs.run_dir + 'model_data/transaction_passive-aggressive'
train_in = dirs.data_dir + 'train_cat.csv'
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


# our best middle of the road
# TODO this should be with a separate test set, with the parameter selection
#      runs using a cv set
# seed = 42, very dependent on seed
# best accuracy = 82%
# best precision = 85%
# this is the same as logreg
# no difference between the two loss functions
prec = ts.run_test(train_in, train_out, test_in, test_out, modelname, embeddings, run_parse=False,
                  model_type='passive-aggressive',
                  C=10., n_iter=1)

"""
Cs = [0.01, 0.1, 1.0, 10., 100., 1000.]
n_iters = [1, 5, 10]

# effect of C
prec = np.empty(len(Cs))
idx = 0
for C in Cs:
    prec[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, modelname, embeddings, run_parse=False,
                            model_type='passive-aggressive',
                           C=C, n_iter=1)
    idx+=1
plt.plot(Cs,prec,'ro')
plt.xscale('log')
plt.show()

# effect of n_iter
prec = np.empty(len(n_iters))
idx = 0
for n_iter in n_iters:
    prec[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, modelname, embeddings, run_parse=False,
                            model_type='passive-aggressive',
                           C=10., n_iter=n_iter)
    idx+=1
plt.plot(n_iters,prec,'ro')
plt.show()
"""