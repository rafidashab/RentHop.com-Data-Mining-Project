import pandas as pd
import numpy as np
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# TODO: in preprocessing
def prunePriceOutliers(df):
	return df[np.abs(df['price'] - df['price'].mean()) <= (3*df['price'].std())]

data = prunePriceOutliers(pd.read_csv('data/train/consolidated_data.csv').dropna())
test_data = pd.read_csv('data/test/consolidated_data.csv')

# Class labels
y = data['interest_level']

# Remove TF-IDF scores for description terms
description_features = list(pd.read_csv('data/train/description_features.csv').columns)[1:]
description_features = []

# Features
X = data.drop(columns = ['listing_id', 'interest_level', 'longitude', 'latitude'] + description_features)
X_test = test_data.drop(columns = ['listing_id', 'longitude', 'latitude'] + description_features)

def runClassification(pipeline, log_loss, accuracy, predict_test):
	if (log_loss):

		log_loss_scores = cross_validate(pipeline, X, y, scoring = 'neg_log_loss', cv = 5, return_train_score = True)
		
		print('Mean log_loss score for training data: {}'.format(-1 * np.mean(log_loss_scores['train_score'])))
		print('Mean log_loss score for validation data: {}'.format(-1 * np.mean(log_loss_scores['test_score'])))

		print('Mean train time: {} seconds'.format(np.mean(log_loss_scores['fit_time'])))
		print('Mean validation time: {} seconds'.format(np.mean(log_loss_scores['score_time'])))

	if (accuracy):

		accuracy_scores = cross_validate(pipeline, X, y, scoring = 'accuracy', cv = 5, return_train_score = True)

		print('Mean log_loss score for training data: {}'.format(np.mean(accuracy_scores['train_score'])))
		print('Mean log_loss score for validation data: {}'.format(np.mean(accuracy_scores['test_score'])))

		print('Mean train time: {} seconds'.format(np.mean(accuracy_scores['fit_time'])))
		print('Mean validation time: {} seconds'.format(np.mean(accuracy_scores['score_time'])))

	if (predict_test):
		pipeline.fit(X, y)

		# Time prediction on test data
		start = time.time()
		test_prediction = pd.DataFrame(pipeline.predict_proba(X_test), columns = pipeline.classes_)
		end = time.time()
		print('Predicting test data took {} seconds.'.format(end - start))

		# Save test predictions to csv
		test_prediction['listing_id'] = test_data['listing_id']
		test_prediction.to_csv('data/test/prediction.csv', header = True, index = False)

####### VERSION 1 #######

rf1 = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
pipeline1 = make_pipeline(MinMaxScaler(), rf1)

####### - #######


####### VERSION 2 #######

rf2 = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 5, max_depth = 1000, criterion = 'gini')
pipeline2 = make_pipeline(MinMaxScaler(), rf2)

####### - #######


# Change pipeline1 to pipeline2 or vice versa
runClassification(pipeline2, False, False, True)