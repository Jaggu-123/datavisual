import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# Feature Engineering
red_wine['wine_type'] = "red"
white_wine['wine_type'] = 'white'

red_wine['quality_label'] = red_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'], categories=['low', 'medium', 'high'])

white_wine['quality_label'] = white_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'], categories=['low', 'medium', 'high'])

# Preview `value_counts()` of the `quality_label` attribute:
print(red_wine['quality_label'].value_counts())
print()
print(white_wine['quality_label'].value_counts())

wines = pd.concat([red_wine, white_wine], axis=0,)

# Re-shuffle records just to randomize data points.
# `drop=True`: this resets the index to the default integer index.
wines = wines.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(wines.head())

# Exploratory Analysis

subset_attributes = ['residual sugar',        #1
                     'total sulfur dioxide',  #2
                     'sulphates',             #3
                     'alcohol',               #4
                     'volatile acidity',      #5
                     'quality']               #6

rs = round(red_wine[subset_attributes].describe(), 2)
print(rs)

ws = round(white_wine[subset_attributes].describe(), 2)

pd.concat([rs, ws], axis=1,
          keys=['🔴 Red Wine Statistics',
                '⚪️ White Wine Statistics'])

subset_attributes = ['alcohol', 'volatile acidity', 'pH', 'quality']

ls = round(wines[wines['quality_label'] == 'low'][subset_attributes].describe(), 2)
ms = round(wines[wines['quality_label'] == 'medium'][subset_attributes].describe(), 2)
hs = round(wines[wines['quality_label'] == 'high'][subset_attributes].describe(), 2)

pd.concat([ls, ms, hs], axis=1,
          keys=['👎 Low Quality Wine',
                '👌 Medium Quality Wine',
                '👍 High Quality Wine'])

fig = wines.hist(bins=15,
                 color='steelblue',
                 edgecolor='black', linewidth=1.0,
                 xlabelsize=10, ylabelsize=10,
                 xrot=45, yrot=0,
                 figsize=(10,9),
                 grid=False)

plt.tight_layout(rect=(0, 0, 1.5, 1.5))
plt.show()