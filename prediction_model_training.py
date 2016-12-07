### Train classifier models on the training data and save the best model ###

# Import python libraries
import numpy as np
import pandas as pd
import pylab
#import renders as rs

# Get specific functions from some other python libraries
from os import listdir
from time import time
from sklearn import cross_validation
from IPython.display import display # Allows the use of display() for DataFrames
from pandas.tools.plotting import scatter_matrix
from sklearn import svm
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve

######################################
def make_scatter_plot(X, name):    
    """
    Make scatterplot.
    Parameters:
    -----------
    X:a design matrix where each column is a feature and each row is a sample.
    name: the name of the plot.
    """
    pylab.clf()
    df = pd.DataFrame(X)
    axs = scatter_matrix(df, alpha=0.2, figsize=(30, 30))

    for ax in axs[:,0]: # the left boundary
        ax.grid('off', axis='both')
        ax.set_yticks([0, 0.5])

    for ax in axs[-1,:]: # the lower boundary
        ax.grid('off', axis='both')
        ax.set_xticks([0, 0.5])

    pylab.savefig("/Users/jcq/Code/Kaggle/NIH seizure prediction/" + name + ".png")
######################################


tot_start = time()


######################################
# Create a list of all the filenames present in the folder containing the processed training features tables
list_proc_features_tables = sorted(listdir("/Users/jcq/Code/Kaggle/NIH seizure prediction/processed_features_tables"))
print "Processed-features tables filenames", list_proc_features_tables
list_proc_features_tables.remove('.DS_Store')

#create the dataframe that will contain all the features and labels
df = pd.DataFrame()

# open the features .csv file and store them in a DataFrame df
frames = [ pd.read_csv("/Users/jcq/Code/Kaggle/NIH seizure prediction/processed_features_tables/"+f) for f in list_proc_features_tables ]
df = pd.concat(frames)
df = df.set_index('index')
df2 = df


# Create two dataframes to store separately the preictal and the interictal samples
preictal_df = (df2.loc[df2['class'] == 1])
interictal_df = (df2.loc[df2['class'] == 0])

# Create two dataframes to store separately the preictal and the interictal labels
df_pre_labels = preictal_df.pop('class')
df_inter_labels = interictal_df.pop('class')

######################################
# Shuffle and split the preictal- and interictal-dataframe into subsets of training and testing examples
# Use a ratio of .3 for the cross-validation set, random state of 11
df_inter_train, df_inter_test, df_inter_labels_train, df_inter_labels_test = cross_validation.train_test_split(interictal_df, df_inter_labels, test_size=0.3, random_state=20)
df_pre_train, df_pre_test, df_pre_labels_train, df_pre_labels_test = cross_validation.train_test_split(preictal_df, df_pre_labels, test_size=0.3, random_state=20)

# Combine preictal- and interictal-dataframes for both training and testing set to ensure a proper distribution of both categories
df2_train = pd.concat([df_inter_train, df_pre_train])
df2_labels_train = pd.concat([df_inter_labels_train, df_pre_labels_train])
df2_test = pd.concat([df_inter_test, df_pre_test])
df2_labels_test = pd.concat([df_inter_labels_test, df_pre_labels_test])
######################################


##### Modification Optimization
df_labels = df.pop('class')
df_train = df
#print "df_train", df_train.describe()
#print "df2_train", df2_train.describe()
#print "df2_labels_train", df2_labels_train.describe()
#print df2_labels_train.head()
#print "df2_labels_test", df2_labels_test.describe()
#print "df2_test", df2_test.describe()


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label=1)


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    return clf


# Initialize the parameters and models

######### cv = StratifiedShuffleSplit(df2_labels_train)
######### MODIFIE pour se passer de GridSearch
#clf = ensemble.AdaBoostClassifier(random_state=5)
###base_estimator = tree.DecisionTreeClassifier(max_depth= 3)
###clf = ensemble.AdaBoostClassifier(base_estimator = base_estimator, n_estimators = 400, learning_rate = 0.0025, random_state=5)
clf = ensemble.GradientBoostingClassifier(n_estimators = 600, learning_rate = 0.2, max_depth = 5, subsample = 0.5, random_state = 5)

#parameters = {'learning_rate': [0.1, 0.5, 0.7,1],
#             'n_estimators': [50, 100, 200, 400]
             #'n_estimators': [300]
#             }

##########f1_scorer = make_scorer(f1_score, pos_label=1)
########## A CHANGER POUR OPTIMISER
##########clf_grid = GridSearchCV(clf, param_grid = parameters, scoring=f1_scorer, fit_params=None, cv = cv)
##########clf_grid.fit(df2_train, df2_labels_train)
clf.fit(df2_train, df2_labels_train)
##########clf = clf_grid.best_estimator_

#print clf
#print "ROC curve", roc_curve(, df2_labels_train, pos_label = 1)
print "F1 score for training set: {:.4f}.".format(predict_labels(clf, df2_train, df2_labels_train))
print "F1 score for test set: {:.4f}.".format(predict_labels(clf, df2_test, df2_labels_test))

joblib.dump(clf, '/Users/jcq/Code/Kaggle/NIH seizure prediction/optimized_models/non_compressed_model.pkl', compress=0)
#joblib.dump(clf, '/Users/jcq/Code/Kaggle/NIH seizure prediction/optimized_models/compressed_model.pkl')



second_features = clf.predict(df_train)

mean_second_features = []
second_labels = [] # Create the reference list for labels
second_pred = [] # Create the list for next predictions

for i in range(len(second_features)/16):
    mean_second_features.append(float(sum(second_features[(i*16):((i+1)*16)]/len(second_features[(i*16):((i+1)*16)]))))
    second_labels.append(df_labels.iloc[i*16+1])

print "calc", mean_second_features[:5], mean_second_features[-5:], sum(mean_second_features)/len(mean_second_features)
print "lab", second_labels[:5], second_labels[-5:], sum(second_labels)/len(second_labels)

for j in range(0,16,1):
    tresh = j/16.
    second_pred = []
    for k in range(len(mean_second_features)):
        if mean_second_features[k] > tresh:
            second_pred.append(1)
        else:
            second_pred.append(0)
    #print "second_pred", second_pred[:5], second_pred[50:55], second_pred[-5:]
    print "F1 score for treshold {}: {:.4f}.".format(tresh, f1_score(second_labels, second_pred, pos_label=1))



tot_end = time()
print "Total calculation time {:.4f} seconds.".format(tot_end - tot_start)
