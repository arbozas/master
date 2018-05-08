import pymongo
import pandas as pa
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from itertools import islice
import numpy as np

def codenotused(df):
    cvec = CountVectorizer(min_df=.0025, max_df=.1, ngram_range=(1,2))
    cvec.fit(df.finalReviews)
    print(list(islice(cvec.vocabulary_.items(), 20)))
    cvec_counts = cvec.transform(df.finalReviews)

    print ('sparse matrix shape:', cvec_counts.shape)
    print ('nonzero count:', cvec_counts.nnz)
    print ('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))

    occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
    counts_df = pa.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
    print(counts_df.sort_values(by='occurrences', ascending=False).head(20))

    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(cvec_counts)

    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pa.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    print(weights_df.sort_values(by='weight', ascending=False).head(20))
