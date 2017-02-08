"""
visualize the pre-trained word embedding
"""
import sys
sys.path.append("..")

import cPickle as pickle
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from initial_setup import directories as dirs

try:
    cat = embeddings['cat']
except:
    print 'Loading embeddings, this may take a few minutes...'
    embedding_name = dirs.data_dir + 'glove_embeddings'
    embeddingFileLoad = open(embedding_name, 'rb')
    embeddings = pickle.load(embeddingFileLoad)
    
word_list = ['mcdonalds','hamburger','food','lunch','subway','metro','ticket',\
             'fare','movie','cinema','film','show','concert']
vec = np.empty((len(word_list),300))   

for idx, word in enumerate(word_list):
    word_vec = embeddings[word]
    vec[idx] = word_vec
    
pca = PCA(n_components=2)
pca.fit(vec)
small_vec = pca.transform(vec)

#plt.plot(small_vec[:,0],small_vec[:,1],'ro')

fig, ax = plt.subplots()
ax.scatter(small_vec[:,0], small_vec[:,1])

for i, txt in enumerate(word_list):
    ax.annotate(txt, (small_vec[i,0],small_vec[i,1]), size=12)
plt.savefig('wordembed.png',bbox_inches='tight')
    
    
    
    
    