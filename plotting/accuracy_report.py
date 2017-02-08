"""
make a report on the accuracy of the test
"""
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initial_setup import directories as dirs

plt.style.use('ggplot')

fileCat = dirs.data_dir + 'test_cat.csv'
df = pd.read_csv(fileCat)

######################################################
# plot the "truth" as motivation
######################################################
cats = df.category.unique()
amounts = [None]*len(cats)
labels = ['Food','Not clear','Transportation','Entertainment','Retail','Bills']
colors = ['red','orange','yellow','green','blue','purple']
          
for idx in range(len(cats)):
    cat = cats[idx]
    thisCat = df.amount[df.truth == cat]
    total = abs(sum(thisCat))
    amounts[idx] = total

fig = plt.figure()
plt.pie(amounts,labels=labels,startangle=10,colors=colors)
plt.axis('equal')
plt.title('Total spent by category')
plt.show()
fig.savefig("total_by_cat.png",bbox_inches='tight')
fig.clf()

counts = [None]*len(cats)
for idx in range(len(cats)):
    cat = cats[idx]
    thisCat = df.amount[df.truth == cat]
    total = len(thisCat)
    counts[idx] = total

fig = plt.figure()
plt.pie(counts, labels=labels,startangle=-90,colors=colors,autopct='%1.1f%%')
plt.axis('equal')
plt.title('Number of transactions by category')
plt.show()
fig.savefig("trans_by_cat.png",bbox_inches='tight')
fig.clf()

fig = plt.figure()
y_pos = np.arange(len(counts))
#plt.pie(averages, labels=labels,startangle=-90,colors=colors)
plt.bar(y_pos, counts, align='center', color = colors)
plt.title('Number of transactions by category')
plt.xticks(y_pos, labels, rotation = 45)
plt.show()
fig.savefig("trans_by_cat_bar.png",bbox_inches='tight')
fig.clf()

averages = np.divide(amounts, counts)
fig = plt.figure()
y_pos = np.arange(len(averages))
#plt.pie(averages, labels=labels,startangle=-90,colors=colors)
plt.bar(y_pos, averages, align='center', color = colors)
plt.title('Average transaction value by category')
plt.xticks(y_pos, labels, rotation = 45)
plt.show()
fig.savefig("avg_by_cat.png",bbox_inches='tight')
fig.clf()

######################################################
# plot accuracy
######################################################
predicted = df.category
true = df.truth
from sklearn import metrics
acc = metrics.accuracy_score(df.truth, df.category)
print "Accuracy = ",acc
prec = metrics.precision_score(df.truth, df.category,average='weighted')
print "Precision = ",prec
recall = metrics.recall_score(df.truth, df.category,average='weighted')
print "Recall = ",recall


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cnf_matrix = metrics.confusion_matrix(df.truth, df.category)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#### TODO I don't think these are the right labels
labels = ['Bills','Entertainment','Food','Health','Retail','Transportation','Not clear']
fig = plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix')
plt.show()
fig.savefig("confused.png",bbox_inches='tight')
fig.clf()

print metrics.classification_report(df.truth, df.category)