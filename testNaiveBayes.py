"""
test naive bayes
"""
import transact as ts
import numpy as np
import matplotlib.pyplot as plt

modelname = 'model_data/transaction_naive-bayes'
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
# best accuracy = 74%
# best precision = 84%
# not good!
prec = ts.run_test(train_in, train_out, test_in, test_out, modelname, embeddings, run_parse=False,
                  model_type='naive-bayes', cutoff=0.7)

"""
cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# effect of cutoff
prec = np.empty(len(cutoffs))
idx = 0
for cutoff in cutoffs:
    prec[idx] = ts.run_test(train_in, train_out, cv_in, cv_out, modelname, embeddings, run_parse=False,
                           cutoff=cutoff)
    idx+=1
plt.plot(cutoffs,prec,'ro')
plt.show()
"""
