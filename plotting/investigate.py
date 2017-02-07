"""
investigate interesting things in the categorized data
"""
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from initial_setup import directories as dirs

plt.style.use('ggplot')

fileCat = dirs.data_dir + 'train_cat.csv'
df = pd.read_csv(fileCat)

cats = df.category.unique()
amounts = [None]*len(cats)
labels = ['Food','Not clear','Transportation','Bills','Entertainment','Retail','Health']
colors = ['red','orange','yellow','green','blue','darkblue','purple']
          
for idx in range(len(cats)):
    cat = cats[idx]
    thisCat = df.amount[df.category == cat]
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
    thisCat = df.amount[df.category == cat]
    total = len(thisCat)
    counts[idx] = total

fig = plt.figure()
plt.pie(counts, labels=labels,startangle=-90,colors=colors)
plt.axis('equal')
plt.title('Number of transactions by category')
plt.show()
fig.savefig("trans_by_cat.png",bbox_inches='tight')
fig.clf()

averages = np.divide(amounts, counts)
fig = plt.figure()
y_pos = np.arange(len(averages))
#plt.pie(averages, labels=labels,startangle=-90,colors=colors)
plt.bar(y_pos, averages, align='center', color = colors)
plt.title('Average transaction value by category')
plt.xticks(y_pos, labels)
plt.show()
fig.savefig("avg_by_cat.png",bbox_inches='tight')
fig.clf()


# of transactions by hour of day, grouped by category
df_time = df[~df.time.isnull()]
df_time['hour'] = df_time['time'].str.replace(':[0-9][0-9]:[0-9][0-9]','')
df_time.hour = df_time.hour.astype(int)
df_time.hour = (df_time.hour - 5) % 24
df_time.drop('cat_int',1,inplace=True)
df_time.amount = abs(df_time.amount)
df_time = df_time[~(df_time.category=='unknown')]

labels = ['Bills','Entertainment','Food','Health','Retail','Transportation']
colors = ['red','orange','yellow','green','blue','purple']
ax = df_time.groupby(['hour', 'category']).size().unstack().plot(kind='bar', stacked=True, color = colors)
lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=labels)
fig = ax.get_figure()
plt.ylabel('Number of transactions')
fig.savefig("trans_per_hour.png", bbox_extra_artists=(lgd,), bbox_inches='tight')


ax = df_time.groupby(['hour', 'category']).sum().unstack().plot(kind='bar', stacked=True, color = colors)
lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=labels)
fig = ax.get_figure()
plt.ylabel('Amount of transactions US$')
fig.savefig("amount_per_hour.png", bbox_extra_artists=(lgd,), bbox_inches='tight')   
  
       
ax = df_time.groupby(['hour', 'category']).mean().unstack().plot(kind='bar', stacked=True, color = colors)
lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=labels)
fig = ax.get_figure()
plt.ylabel('Average amount of transactions US$')
fig.savefig("avg_amount_per_hour.png", bbox_extra_artists=(lgd,), bbox_inches='tight')         





