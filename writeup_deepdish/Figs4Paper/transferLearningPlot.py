# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:26:43 2017

@author: agoswami
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'F:\hackerreborn++\stanford_cs231n\cs231n_haichenrepobb\project_final\deepdish_writeup\Figs4Paper\transferLearning.csv', header=None, names=['epoch1_train', 'epoch1_val', 'epoch2_train', 'epoch2_val'])

epoch1 = df[['epoch1_train', 'epoch1_val']]
epoch2 = df[['epoch2_train', 'epoch2_val']]

all_dfs = [epoch1, epoch2]

# Give all df's common column names
for df in all_dfs:
    df.columns = ['Train', 'Val']

df2 = pd.concat(all_dfs).reset_index(drop=True)

plt.plot(df2['Train'], '-o', label='train')
plt.plot(df2['Val'], '-o', label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('transferLearning.png')
plt.show()