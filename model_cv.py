### Implement the cross-validation training and test on a model ###
### Use all channels of one epoch as a single training sample ###
### The subset of features is to be defined for each model ###
### The script implements a 5-folds cross-validation ###
# Require the cv train/test datasets to be stored in the folder "subsets".

# Import python libraries
import numpy as np
import pandas as pd
import pylab
import numpy

# Get specific functions from some other python libraries
from os import listdir
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from xgboost.sklearn import XGBClassifier

tot_start = time()

######################################
# Initiate the list of parameter to be tuned
### TO CUSTOMIZE FOR EACH MODEL
param_list = ['n_estimators', 'max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'learning_rate']

# Initiate the list of metrics to be evaluated
metrics_list = ['roc', 'F1', 'recall']

# Initiate the dictionary containing the list of values for parameter tuning
### TO CUSTOMIZE FOR EACH MODEL
param_dic = {
 'max_depth':[6, 7, 8, 9],
 'learning_rate': [0.1, 0.08, 0.06],
 'n_estimators': [100, 500, 600, 800, 1000],
 'gamma':[0, 0.1, 0.2, 0.3],
 'min_child_weight':[4, 6, 8, 10],
 'subsample':[i/10.0 for i in range(5,9)],
 'colsample_bytree':[i/10.0 for i in range(8,10)]
}

# Initiate the dictionary containing the list of values for the parameters
### TO CUSTOMIZE FOR EACH MODEL
best_param_set = {
 'max_depth': 6,
 'learning_rate': 0.1,
 'n_estimators': 100,
 'gamma':0,
 'min_child_weight': 4,
 'subsample':0.5,
 'colsample_bytree':0.8
}


# Implement a 5-folds cross-validation test
def xgb_model_cv(param_set):

	mean_roc_cv = 0
	mean_F1_score_cv = 0
	mean_recall_score_cv = 0

	# Initialize the XGBClassifier with the dictionary parameters
	clf = XGBClassifier(max_depth=param_set['max_depth'], learning_rate=param_set['learning_rate'], n_estimators=param_set['n_estimators'], silent=True, objective='binary:logistic', nthread=-1, gamma=param_set['gamma'], min_child_weight=param_set['min_child_weight'], max_delta_step=0, subsample=param_set['subsample'], colsample_bytree = param_set['colsample_bytree'])

	for i in range(5):

		#create the dataframe that will contain all the features and labels
		### TO CUSTOMIZE FOR EACH MODEL
		train_df = pd.read_csv("/subsets/train_cv_set"+str(i)+".csv")
		test_df = pd.read_csv("/subsets/test_cv_set"+str(i)+".csv")

		train_df.sort_values('index', axis = 0, inplace=True)
		train_df = train_df.set_index('index')
		train_label_df = train_df.pop('class')

		test_df.sort_values('index', axis = 0, inplace=True)
		test_df = test_df.set_index('index')
		test_label_df = test_df.pop('class')

		# Train the classifier on the train set
		clf.fit(train_df, train_label_df)

		# Make predictions on the train set
		train_pred = clf.predict(train_df)
		train_predprob = clf.predict_proba(train_df)[:,1]

		# Make predictions on the test set
		test_pred = clf.predict(test_df)
		test_predprob = clf.predict_proba(test_df)[:,1]

		# Display the metrics score on the train set
		print "ROC score", roc_auc_score(train_label_df, train_pred, average = 'macro')
		print "ROC score proba", roc_auc_score(train_label_df, train_predprob, average = 'macro')
		print "F1 score for training set: {:.4f}.".format(f1_score(train_label_df, train_pred, pos_label=1.0))
		print "Recall score for training set: {:.4f}.".format(recall_score(train_label_df, train_pred, pos_label=1.0, average='binary'))

		# Display the metrics score on the test set
		roc_score_cv = roc_auc_score(test_label_df, test_pred, average = 'macro')
		F1_score_cv = f1_score(test_label_df, test_pred, pos_label=1.0)
		recall_score_cv = recall_score(test_label_df, test_pred, pos_label=1.0, average='binary')
		print "ROC score cv", roc_score_cv
		print "ROC score proba cv", roc_auc_score(test_label_df, test_predprob, average = 'macro')
		print "F1 score for cv set: {:.4f}.".format(F1_score_cv)
		print "Recall score for cv set: {:.4f}.".format(recall_score_cv)

		# Calculate the average metrics scores over the 5-folds cv tests
		mean_roc_cv += roc_score_cv/5
		mean_F1_score_cv += F1_score_cv/5
		mean_recall_score_cv += recall_score_cv/5

	# Display the average metrics scores over the 5-folds cv tests
	print "mean_roc_cv", mean_roc_cv
	print "mean_F1_score_cv", mean_F1_score_cv
	print "mean_recall_score_cv", mean_recall_score_cv

	score = [mean_roc_cv, mean_F1_score_cv, mean_recall_score_cv]
	
	return score


# Initialize the best_score value for F1_score
best_score = 0

# Create the dataframe collecting the parameters and metrics scores for each combination of parameter tested
columns  = param_list + metrics_list
param_search_df = pd.DataFrame(np.zeros((1,len(columns))), columns = columns)

# Implement the selection of the best parameters
for param in range(len(param_list)): # Iterate through the list of parameters
	for p_value in param_dic[param_list[param]]: # For each parameter iterate through the list of values
		param_set = best_param_set.copy()
		#print param_list[param] [debug]
		param_set[param_list[param]] = p_value

		# Calculate the cv metrics score on the 5-folds train/test splits
		score = xgb_model_cv(param_set)

		# modify the dictionary collecting the best parameters if the best score is improved
		if score[1] > best_score:
			best_score = score[1]
			best_param_set[param_list[param]] = p_value

		# Record the parameters combinations and the scores in a dataframe
		temp_param_search_df = pd.DataFrame(np.zeros((1,len(columns))), columns = columns)
		for col in param_list:
			temp_param_search_df[col] = param_set[col]
		for j in range(len(metrics_list)):
			temp_param_search_df[metrics_list[j]] = score[j]

		param_search_df = pd.concat([param_search_df, temp_param_search_df])

print "best model", best_param_set # print the best parameter configuration

# Record the dataframe collecting the metrics scores in a .csv file
### TO CUSTOMIZE FOR EACH MODEL
file_name = "cv_optimization_Model.csv"
param_search_df.to_csv(file_name)

tot_end = time()
print "Total calculation time {:.4f} seconds.".format(tot_end - tot_start)
