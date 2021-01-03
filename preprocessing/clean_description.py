import pandas as pd
import numpy as np
import io
import nltk
import sys

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

if ((len(sys.argv)) == 1):
	sys.exit("test or train arg is missing.")

test_train = sys.argv[1].strip().lower()
if (test_train != 'test' and test_train != 'train'):
	sys.exit("Invalid argument passed.")

dataset_dir = 'data/{}.json'.format(test_train)
output_dir = 'data/{}/'.format(test_train)

stop_words = set(stopwords.words('english')) 
porter = PorterStemmer()

data = pd.read_json(dataset_dir, convert_dates = ['created'])

def parse_text(text):
	soup = BeautifulSoup(desc)
	return ' ' + soup.get_text()

def clean_text(text):
	text = text.lower()
	tokens = word_tokenize(text)
	words = []
	for token in tokens:
		if (token.isalpha()):
			words.append(porter.stem(token))
	cleaned_text = ' '.join(words)
	return cleaned_text

descriptions = data['description'].values
listing_ids = data['listing_id'].values

corpus = []

i = 1
n = len(descriptions)
for desc in descriptions:
	text = parse_text(desc)
	text = clean_text(text)
	corpus.append(text.encode('utf-8'))
	print('{} of {} completed.'.format(i,n))
	i = i+1

cleaned_descriptions = pd.DataFrame({'listing_id': listing_ids, 'description': corpus})
cleaned_descriptions.to_csv(output_dir + 'description_cleaned.csv')