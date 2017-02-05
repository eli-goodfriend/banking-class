"""
try being passive aggressive
"""

import transact as ts
import numpy as np
import matplotlib.pyplot as plt

modelname = 'transaction_passive-aggressive'
train_in = '/home/eli/Data/Narmi/train_cat.csv'
train_out = '/home/eli/Data/Narmi/train_cat.csv'
cv_in = '/home/eli/Data/Narmi/cv.csv'
cv_out = '/home/eli/Data/Narmi/cv_cat.csv'
test_in = '/home/eli/Data/Narmi/test.csv'
test_out = '/home/eli/Data/Narmi/test_cat.csv'

# glove's pretrained model, loaded into dictionary
try:
    cat = embeddings['cat']
except:
    embedding_name = '/home/eli/Data/Narmi/glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)


# our best middle of the road
# TODO this should be with a separate test set, with the parameter selection
#      runs using a cv set
# seed = 42, very dependent on seed
# best accuracy = 81.75%
# this is the same as logreg
# no difference between the two loss functions
acc = ts.run_test(train_in, train_out, test_in, test_out, modelname, embeddings, run_parse=False,
                  model_type='passive-aggressive',
            C=0.001, cutoff=0.50, n_iter=1)

"""
Cs = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_iters = [1, 5, 10]

# effect of C
acc = np.empty(len(Cs))
idx = 0
for C in Cs:
    acc[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, modelname, embeddings, run_parse=False,
                            model_type='passive-aggressive',
                           C=C, cutoff=0.50, n_iter=1)
    idx+=1
plt.plot(Cs,acc,'ro')
plt.xscale('log')
plt.show()

# effect of n_iter
acc = np.empty(len(n_iters))
idx = 0
for n_iter in n_iters:
    acc[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, modelname, embeddings, run_parse=False,
                            model_type='passive-aggressive',
                           C=0.001, cutoff=0.50, n_iter=n_iter)
    idx+=1
plt.plot(n_iters,acc,'ro')
plt.show()
"""