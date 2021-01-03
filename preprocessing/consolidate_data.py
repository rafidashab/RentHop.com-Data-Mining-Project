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

# It would be best to clean data, i.e deal with missing data and outliers before this step on test and train sets
data = pd.read_json(dataset_dir, convert_dates = ['created'])

# We only need a subset of columns (features) for classification + listing_id
# Suggestion: transform date to hour integer feature
if test_train == 'train':
	data = data[['listing_id', 'bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'interest_level', 'created']]
else:
	data = data[['listing_id', 'bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'created']]
data.set_index('listing_id')

# Read in extracted features
description_features = pd.read_csv(output_dir + 'description_features.csv')
description_features.set_index('listing_id')

featurelist_features = pd.read_csv(output_dir + 'featurelist_features.csv')
featurelist_features.set_index('listing_id')

# Join based on listing_id
data = pd.merge(data, description_features, on = 'listing_id')
data = pd.merge(data, featurelist_features, on = 'listing_id')
data = data.drop(columns = ['Unnamed: 0_x', 'Unnamed: 0_y'])
data.set_index('listing_id')

data['hour'] = data['created'].dt.hour
data = data.drop(columns = ['created'])

data.to_csv(output_dir + 'consolidated_data.csv', index = False)