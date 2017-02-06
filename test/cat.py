"""
test categorization using models
this isn't going to work well, since it isn't trained
this is to test that it runs to completion
this is not automated, it needs hand checking of the output
"""
import sys
sys.path.append("..")

import transact as ts
import pandas as pd
from sklearn import linear_model
import numpy as np
from initial_setup import directories as dirs

filein = 'test_lookup.csv'
fileout = 'test_cat.csv'
df = pd.read_csv(filein)

try:
    cat = embeddings['cat']
except:
    embedding_name = dirs.data_dir + 'glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)

model = linear_model.SGDClassifier(loss='log')

catData = df[~df.category.isnull()]
uncatData = df[df.category.isnull()]
print str(float(len(catData))/float(len(df)) * 100.) + "% of transactions categorized with lookup."

ts.train_model(catData,model,embeddings,model_type='logreg',new_run=True)
ts.use_model(uncatData,model,embeddings,0.0,model_type='logreg')

df = pd.concat([catData, uncatData])
df.sort_index(inplace=True)

df.to_csv(fileout,index=False)


