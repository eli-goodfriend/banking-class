"""
open pre-trained glove word embedding
data from http://nlp.stanford.edu/projects/glove/
"""
import os
import numpy as np
import cPickle as pickle
import directories as dirs

GLOVE_DIR = dirs.data_dir
model_name = dirs.data_dir + 'glove_embeddings'

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

modelFileSave = open(model_name, 'wb')
pickle.dump(embeddings_index, modelFileSave)
modelFileSave.close()

