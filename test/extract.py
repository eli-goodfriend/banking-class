"""
test feature extraction
this is not automated, needs hand inspection
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd
import cPickle as pickle

filein = 'test_lookup.csv'
df = pd.read_csv(filein)

try:
    cat = embeddings['cat']
except:
    embedding_name = '/home/eli/Data/Narmi/glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)

X = ts.extract(df,embeddings,model_type='None')

print X.shape
