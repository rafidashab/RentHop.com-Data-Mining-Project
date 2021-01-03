import pandas as pd
import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer

if ((len(sys.argv)) == 1):
	sys.exit("test or train arg is missing.")

test_train = sys.argv[1].strip().lower()
if (test_train != 'test' and test_train != 'train'):
	sys.exit("Invalid argument passed.")

dataset_dir = 'data/{}/description_cleaned.csv'.format(test_train)
output_dir = 'data/{}/'.format(test_train)

data = pd.read_csv(dataset_dir)

corpus = data['description'].values

tfidf_vectorizer=TfidfVectorizer(encoding = 'utf-8', stop_words = 'english', use_idf=True, lowercase = True, max_features = 10)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(corpus)

tfidfs = tfidf_vectorizer_vectors.todense()
df = pd.DataFrame(tfidfs, columns = tfidf_vectorizer.get_feature_names())
df['listing_id'] = data['listing_id']
df.set_index('listing_id')

df.to_csv(output_dir + 'description_features.csv')