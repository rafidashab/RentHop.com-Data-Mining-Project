import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score




# The consolidated data is from Assignment 1
data = pd.read_csv('../data/train/consolidated_data.csv')

# Decision tree is pretty good at selecting the best features
# feature_cols = ['listing_id', 'bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']
feature_cols = ['listing_id', 'bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'apart', 'bedroom', 'build',
                'floor', 'kitchen', 'locat', 'new', 'room', 'thi', 'view', 'elevator', 'hardwood floors',
                'cats allowed', 'dogs allowed', 'laundry in building', 'fitness center']


clf = DecisionTreeClassifier(max_depth=5, min_samples_split=0.1, min_samples_leaf=3)
X_train = data[feature_cols]
y_train = data['interest_level']

# Cross Validation Scores : Tune parameters based on cross validation scores
# min_samples_split= 0.2, max_features=10, min_samples_leaf=0.2
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_log_loss')
print("Cross Validation Scores:", -1 * scores)
print("Cross Validation Score Mean:", -1 * scores.mean())

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print("Accuracy Scores:", -1 * scores)
print("Accuracy Score Mean:", -1 * scores.mean())

start_time = time.time()
# Main Classifier
clf.fit(X_train, y_train)
print("--- Training Time : %s seconds ---" % (time.time() - start_time))

data_test = pd.read_csv('../data/test/consolidated_data.csv')

X_test = data_test[feature_cols]

start_time = time.time()
y_predict = clf.predict_proba(X_test)
print("--- Prediction Time: %s seconds ---" % (time.time() - start_time))

y_train_predict = clf.predict_proba(X_train)
y_predict_accuracy = clf.predict(X_train)
training_score = log_loss(y_train, y_train_predict)
training_score_accuracy = accuracy_score(y_train, y_predict_accuracy)
print("Training Log Loss:", training_score)
print("Training Log Loss:", training_score_accuracy)

output = pd.DataFrame(y_predict, index=X_test['listing_id'], columns=list(['high', 'low', 'medium']))
columns_titles = ["high", "medium", "low"]
output = output.reindex(columns=columns_titles)
output.to_csv('dt_output.csv')
