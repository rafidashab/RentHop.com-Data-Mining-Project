import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

data = pd.read_csv('../data/train/consolidated_data.csv')
test_data = pd.read_csv('../data/test/consolidated_data.csv')

X_train = data.drop(columns = ['interest_level'])
X_test = test_data
y = data['interest_level']

#### VERSION 1
# clf = LogisticRegression()

#### VERSION 2
# Adapted from https://scikit-learn.org/stable/modules/cross_validation.html
clf = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(solver = 'liblinear',max_iter=1000, C = 0.7))

scores = cross_val_score(clf, X_train, y, cv=10)
# print('Accuracy: ', scores)
print('Mean Accuracy: ', scores.mean())

scores = cross_val_score(clf, X_train, y, cv=10, scoring = 'neg_log_loss')
# print('Cross Val neg_log_loss Scores: ', -1 * scores)
print('Cross Val neg_log_loss Mean Score: ', -1 * scores.mean())

start1 = time.time()
clf = clf.fit(X_train, y)
stop1 = time.time()
print('\nTime took to train: ', stop1-start1, 'seconds')

prediction_train = clf.predict_proba(X_train)
training_score = log_loss(y, prediction_train)
print('\nPrediction Training log loss: ', training_score)

start2 = time.time()
prediction_test = clf.predict_proba(X_test)
stop2 = time.time()
print('\nTime took to predict ', stop2-start2, 'seconds')

prediction = pd.DataFrame(prediction_test, index = X_test['listing_id'] ,columns=list(['high', 'low', 'medium']))

prediction.to_csv('log_reg_predictions.csv')