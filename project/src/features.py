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
    return df