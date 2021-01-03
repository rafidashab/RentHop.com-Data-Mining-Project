import pandas as pd
import numpy as np
import sys

if ((len(sys.argv)) == 1):
	sys.exit("test or train arg is missing.")

test_train = sys.argv[1].strip().lower()
if (test_train != 'test' and test_train != 'train'):
	sys.exit("Invalid argument passed.")

dataset_dir = 'data/{}.json'.format(test_train)
output_dir = 'data/{}/'.format(test_train)

data = pd.read_json(dataset_dir, convert_dates = ['created'])

feature_dict = {}

feat_lists = data['features'].values

# create dict of feature and frequency of feature
for feat_list in feat_lists:
	for feat in feat_list:
		feat = feat.lower().strip()
		if feat in feature_dict:
			feature_dict[feat] += 1
		else:
			feature_dict[feat] = 1

# Find n most frequent features
selected_features = []
n = 10
for i in range(n):
	key = max(feature_dict, key=feature_dict.get)
	selected_features.append(key)
	feature_dict.pop(key, None)

# BINARY ENCODING
doc_vectors = []
for feat_list in feat_lists:
	doc_vector = []
	for feature in selected_features:
		if feature in [feat.lower().strip() for feat in feat_list]:
			doc_vector.append(1)
		else:
			doc_vector.append(0)
	doc_vectors.append(doc_vector)

doc_matrix = np.matrix(doc_vectors)
df = pd.DataFrame(doc_matrix, columns = selected_features)

df['listing_id'] = pd.Series(data['listing_id'].values)
df.set_index('listing_id')

df.to_csv(output_dir + 'featurelist_features.csv')