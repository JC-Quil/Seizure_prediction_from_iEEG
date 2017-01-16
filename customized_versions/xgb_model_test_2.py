### model_test.py implmentation for the 2nd model ###
### Test the classifier model on the test set ###
### Return the metrics evaluation ###
# Require the test dataset to be stored in the folder "subsets" and the folder "predictions"
# to store the predictions.


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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from xgboost.sklearn import XGBClassifier

tot_start = time()

######################################
# Open the train and test label .csv file listing the filename and their labels
train_data_labels_safe = pd.read_csv("train_and_test_data_labels_safe.csv")

# Create two dataframes, for the features and for the labels
df = pd.read_csv("/subsets/test_set2.csv")
df.sort_values('index', axis = 0, inplace=True)
index_df = pd.DataFrame(df['index']) 
df = df.set_index('index')
label_df = df.pop('class')

# Initialize the classifier using the trained model
clf = joblib.load('/trained_models/model_2.pkl')

# Calculate the classifier predictions for all the test subsamples
y_pred = clf.predict(df)
y_proba = clf.predict_proba(df) 
y_proba = y_proba[:,1]

# Save the predictions and positive prediction probabilities in a dataframe
y_pred_df = pd.DataFrame(y_pred, index = index_df.index, columns = ['pred'])
y_proba_df = pd.DataFrame(y_proba, index = index_df.index, columns = ['proba']) 
y_pred_df = pd.concat([y_pred_df, y_proba_df, index_df], axis = 1) 
y_pred_df.columns = ['pred', 'proba', 'ind'] 
y_pred_df = y_pred_df.sort_values('ind',axis = 0)

# Create the dictionary that will store the name and prediction for each sample file
dic_final = {'File' : [],
            'Class predict2' : np.zeros(train_data_labels_safe.shape[0]),
            'Proba2' : np.zeros(train_data_labels_safe.shape[0]), 
            'Class' : np.zeros(train_data_labels_safe.shape[0])
    }

# Fill the class and probabilities for each sample file in the final dictionary
for i in range(train_data_labels_safe.shape[0]):
    dic_final['File'].append(train_data_labels_safe['image'][i]) # Fill the sample filename
    dic_final['Class predict2'][i] = -1
    dic_final['Class'][i] = train_data_labels_safe['class'][i] # Fill the target label of the sample

    if ((i*16*20+1) >= y_pred_df['ind'].iloc[0]) & ((i*16*20+1) <= y_pred_df['ind'].iloc[-1]):
        a = (i*16*20+1) # Value equal to the index of the first feature row corresponding to the ith sample
        b = (i*16*20+16*20) # Value equal to the index of the last feature row corresponding to the ith sample
        pred_final = y_pred_df[(y_pred_df['ind']>= a) & (y_pred_df['ind'] <= b)]

        if not pred_final.empty: 
            # Attribute to a full sample the maximal probability of being positive of its subsample 
            proba_final = pred_final['proba'][pred_final['proba'].idxmax()] 
            # Attribute a positive class to a full sample if one of its subsample is evaluated positive
            pred_final = pred_final['pred'][pred_final['proba'].idxmax()]
            dic_final['Class predict2'][i] = pred_final
            dic_final['Proba2'][i] = proba_final 


# Record the predictions dataframe in a .csv file
file_name = "/predictions/predictions_model2.csv"
df_final = pd.DataFrame(dic_final, columns = ['File','Class predict2', 'Proba2', 'Class'], index = range(train_data_labels_safe.shape[0]))
df_final = df_final.set_index(['File'])
df_final = df_final.loc[df_final['Class predict2'] != -1]
df_final = df_final.dropna(subset = ['Class predict2'])
df_final.to_csv(file_name)

# Print metrics evaluations
print "F1 score for test set: {:.4f}.".format(f1_score(df_final['Class'], df_final['Class predict2'], pos_label=1.0))
print "Recall score for test set: {:.4f}.".format(recall_score(df_final['Class'], df_final['Class predict2'], pos_label=1.0, average='binary'))
print "ROC score for test set: {:.4f}.".format(roc_auc_score(df_final['Class'], df_final['Class predict2'], average = 'macro'))
print "ROC score proba for test set: {:.4f}.".format(roc_auc_score(df_final['Class'], df_final['Proba2'], average = 'macro'))

tot_end = time()
print "Total calculation time {:.4f} seconds.".format(tot_end - tot_start)
