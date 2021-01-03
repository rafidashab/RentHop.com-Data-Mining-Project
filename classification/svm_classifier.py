import pandas as pd
import numpy as np
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn import svm

data = pd.read_csv('data/train/consolidated_data.csv').dropna()
test_data = pd.read_csv('data/test/consolidated_data.csv')

# Class labels
y = data['interest_level']

# Features
X = data.drop(columns = ['listing_id', 'interest_level'])
X_test = test_data.drop(columns = ['listing_id'])

def runClassification(pipeline):
	log_loss_scores = cross_validate(pipeline, X, y, scoring = 'neg_log_loss', cv = 3, return_train_score = True)
	
	print('Mean log_loss score for training data: {}'.format(-1 * np.mean(log_loss_scores['train_score'])))
	print('Mean log_loss score for validation data: {}'.format(-1 * np.mean(log_loss_scores['test_score'])))

	print('Mean train time: {} seconds'.format(np.mean(log_loss_scores['fit_time'])))
	print('Mean validation time: {} seconds'.format(np.mean(log_loss_scores['score_time'])))

	accuracy_scores = cross_validate(pipeline, X, y, scoring = 'accuracy', cv = 5, return_train_score = True)

	print('Mean log_loss score for training data: {}'.format(np.mean(accuracy_scores['train_score'])))
	print('Mean log_loss score for validation data: {}'.format(np.mean(accuracy_scores['test_score'])))

	print('Mean train time: {} seconds'.format(np.mean(accuracy_scores['fit_time'])))
	print('Mean validation time: {} seconds'.format(np.mean(accuracy_scores['score_time'])))

	# Time prediction on test data
	pipeline.fit(X, y)
	start = time.time()
	test_prediction = pd.DataFrame(pipeline.predict_proba(X_test), columns = pipeline.classes_)
	end = time.time()
	print('Predicting test data took {} seconds.'.format(end - start))

	# Save test predictions to csv
	#test_prediction['listing_id'] = test_data['listing_id']
	#test_prediction.to_csv('data/test/prediction.csv', header = True, index = False)



####### VERSION 1 #######

'''
svm_classifier = svm.SVC(gamma = 'auto', probability = True)
pipeline = make_pipeline(svm_classifier)

runClassification(pipeline)
'''

####### - #######


####### VERSION 2 #######


svm_classifier = svm.SVC(gamma = 'auto', probability = True, C = 0.7)
pca = PCA(n_components = 10)
pipeline = make_pipeline(MinMaxScaler(), pca, svm_classifier)

runClassification(pipeline)


####### - #######
