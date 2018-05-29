import pymongo
import pandas as pa
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from itertools import islice
import numpy as np

#The categories of stars 1,2,3,4,5 reduced to 0,1
def MulticlassToBinary(df):
    df['targets'] = df['stars']
    df.loc[df['stars'] <= 3, 'targets'] = 0
    df.loc[df['stars'] > 3, 'targets'] = 1
    df.drop("stars", axis=1, inplace=True)
    return df

 # Undersample xs, ys to balance classes.
def balance_classes(xs, ys):
    freqs = Counter(ys)
# the least common class is the maximum number we want for all classes
    max_allowable = freqs.most_common()[-1][1]
    num_added = {clss: 0 for clss in freqs.keys()}
    new_ys = []
    new_xs = []
    for i, y in enumerate(ys):
        if num_added[y] < max_allowable:
            new_ys.append(y)
            new_xs.append(xs[i])
            num_added[y] += 1

    return new_xs, new_ys