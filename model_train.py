### Train classifier model and save the trained model ###
### Use all channels of one 30-second subsample as a single training sample ###
# Require the train dataset to be stored in the folder "subsets" and the folder "trained_model"
# to store the trained model.


# Import python libraries
import numpy as np
import pandas as pd
import pylab
import numpy

# Get specific functions from some other python libraries
from os import listdir
from time import time
from IPython.display import display
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from xgboost.sklearn import XGBClassifier

tot_start = time()

######################################
# Load the train set and create a features and labels dataframe
### TO CUSTOMIZE FOR EACH MODEL
df= pd.read_csv("subsets/train_set.csv")
df.sort_values('index', axis = 0, inplace=True)
df = df.set_index('index')
label_df = df.pop('class')

# Define the XGBoost classifer, parameters should be defined after optimization
### TO CUSTOMIZE FOR EACH MODEL
clf = XGBClassifier(max_depth=9, learning_rate=0.1, n_estimators=1000, silent=True, objective='binary:logistic', nthread=-1, gamma=0.2, min_child_weight=10, max_delta_step=0, subsample=0.6, colsample_bytree = 0.7)
clf.fit(df, label_df)

# Calculate the classifier predictions and print metrics calculations
pred = clf.predict(df)
predprob = clf.predict_proba(df)[:,1]
print "F1 score for training set: {:.4f}.".format(f1_score(label_df, pred, pos_label=1.0))
print "Recall score for training set: {:.4f}.".format(recall_score(label_df, pred, pos_label=1.0, average='binary'))
print "ROC score for training set: {:.4f}.".format(roc_auc_score(label_df, pred, average = 'macro'))
print "ROC score proba for training set: {:.4f}.".format(roc_auc_score(label_df, predprob, average = 'macro'))

# Save the trained model in the folder "trained_model"
### TO CUSTOMIZE FOR EACH MODEL
joblib.dump(clf, 'trained_models/model.pkl', compress=0)

tot_end = time()
print "Total calculation time {:.4f} seconds.".format(tot_end - tot_start)
