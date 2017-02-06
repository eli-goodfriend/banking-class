"""
test feature extraction
this is not automated, needs hand inspection
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd
import cPickle as pickle
from initial_setup import directories as dirs

filein = 'test_lookup.csv'
df = pd.read_csv(filein)

try:
    cat = embeddings['cat']
except:
    embedding_name = dirs.data_dir + 'glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)

X = ts.extract(df,embeddings,model_type='None')

print X.shape
