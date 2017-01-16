### train_test_cv_split.py implementation for the project 1st model ###
### Split the labelled dataset between train and test set, and create a 5-Fold cross validation split ###
### Organize each subsamples 16 channels features into one feature row ###
### Kfold functions from libraries are not appropriate due to the dataset structure ###
# Require the processed features table to be stored in the folder "processed_features_tables" and the folder "subsets"
# to store train/test sets.

# Import python libraries
import numpy as np
import pandas as pd
import pylab
import numpy

# Get specific functions from some other python libraries
from os import listdir
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.externals import joblib
from sklearn.cross_validation import KFold

tot_start = time()

######################################
# Create a list of all the filenames present in the folder containing the processed features tables
list_proc_features_tables = sorted(listdir("processed_features_tables"))
if '.DS_Store' in list_proc_features_tables:
    list_proc_features_tables.remove('.DS_Store')

#create the dataframe that will contain all the features and labels
df = pd.DataFrame()

# Create the list of features to drop from dataset
functions_1 = ['mean', 'std dev','spectral edge frequency', 'correlation matrix (channel)', 'hjorth activity', 'hjorth mobility',
             'hjorth complexity', 'skewness', 'kurtosis', 'Katz FD', 'Higuchi FD']
functions_7 = ['power spectral density','PSD wavelet', 'correlation matrix (frequency)']

functions_list = []
for function in functions_7:
    for feat_no in range(0,7,1):
        functions_list.append(function+str(feat_no))
        functions_list.append(function+str(feat_no)+"_pow")
for function in functions_1:
    functions_list.append(function)
    functions_list.append(function+"_pow")
#print len(functions_list) #[debug]

# Drop the features selected above from the dataset
# Assemble each subsamples 16 channels features into one feature row 
for i in range(len(list_proc_features_tables)):
    f = list_proc_features_tables[i]
    df_temp = pd.read_csv("processed_features_tables/"+f)
    if i == 0:
        df = df_temp.drop(labels = functions_list, axis = 1)
    else:
        df_temp = df_temp.drop(labels = functions_list, axis = 1 )
        df_temp = df_temp.drop(labels = ['index', 'class'], axis = 1 )
        col_names = list(df_temp.columns.values)
        columns_names = []
        for k in range(0,df_temp.shape[1],1):
            a = str(i)+col_names[k]
            columns_names.append(a)
        df_temp.columns = columns_names
        df = pd.concat([df, df_temp], axis = 1)

df.sort_values('index', axis = 0, inplace=True)
df.set_index([range(df.shape[0])], inplace = True)


# Split the dataset into 3 parts: interictal samples, sorted preictal samples and unsorted preictal samples
# Create the dataframe containing the unsorted preictal samples features
ix_1 = 1990401 # Index of the first sample of the category
second_preictal_index = df[df['index'] >= ix_1].index.tolist()
#print "second_preictal_list", len(second_preictal_index) [debug]

# Create two dataframes to store separately the preictal and the interictal samples
preictal_df = df.loc[(df['class'] == 1) & (df['index'] < ix_1)]
preictal_df = preictal_df.sort_values('index', axis = 0)
# print "preictal_df", preictal_df.shape [debug]

interictal_df = (df.loc[df['class'] == 0])
interictal_df = interictal_df.sort_values('index',axis = 0)
# print "interictal_df", interictal_df.shape [debug]

# Create the train test splits for the preictal and the interictal samples
preictal_folds = list(KFold(n = preictal_df.shape[0], n_folds=25, shuffle=False))
interictal_folds = list(KFold(n = interictal_df.shape[0], n_folds=24, shuffle=False))


# Create the test set
test_i = [] # create the list for temporary storage of interictal sample indices for the test set
test_p = [] # create the list for temporary storage of preictal sample indices for the test set

# Compose the list of indices for the test set
for i in range(25): 
    if ((i+1)%5) == 0: 
        tr_p, tt_p = preictal_folds[i]
        test_p = test_p + list(tt_p+interictal_df.shape[0])
    if ((i+1)%6) == 0 and i < 24: 
        tr_i, tt_i = interictal_folds[i] 
        test_i = test_i + list(tt_i) 

# Create the test set dataframe and record it in a .csv file 
test_df = pd.concat([df.iloc[test_p], interictal_df.iloc[test_i]], axis = 0)
test_file_name = "subsets/test_set1.csv" 
test_df.set_index('index', inplace = True) 
test_df.to_csv(test_file_name) 
#print "test_df", test_df.shape [debug]


# Create the train set, and train/test cross-validation sets
# Create list of list for indices for the train/test cross-validation sets
index_lists = [[], [], [], [], [], [], [], [], [], []]

for j in range(25): 
    if ((j+1)%5) != 0: 
        k = int((j+1)%5-1) 
        tr_p, tt_p = preictal_folds[j]
        index_lists[k] = index_lists[k] + list(tt_p+interictal_df.shape[0])
        a = [0, 1, 2, 3, 4]
        a.remove(k)
        for m in a:
            index_lists[m+5] = index_lists[m+5] + list(tt_p+interictal_df.shape[0])   

    if j == 4:
        k = int(4)
        index_lists[k] = index_lists[k] + second_preictal_index
        a = [0, 1, 2, 3, 4]
        a.remove(4)
        for m in a:
            index_lists[m+5] = index_lists[m+5] + second_preictal_index   

    if ((j+1)%6) != 0 and j < 24:
        k = int((j+1)%6-1) 
        tr_i, tt_i = interictal_folds[j] 
        index_lists[k] = index_lists[k] + list(tt_i)
        a = [0, 1, 2, 3, 4]
        a.remove(k)
        for m in a:
            index_lists[m+5] = index_lists[m+5] + list(tt_i)

train_set_index_list = []
for n in range(5):
    train_set_index_list += index_lists[n]

# Create the train set dataframe and record it in a .csv file 
train_file_name = "subsets/train_set1.csv" 
train_df = df.iloc[list(set(train_set_index_list))]
train_df.set_index('index', inplace = True) 
train_df.to_csv(train_file_name) 
#print "train_df", train_df.shape [debug]


# Create the train/test cross-validation sets dataframe and record them in .csv files (5-Fold split)
for l in range(5):
    cv_df = df.iloc[list(set(index_lists[l]))]
    train_df = df.iloc[list(set(index_lists[l+5]))]

    cv_df = cv_df.set_index('index')
    train_df = train_df.set_index('index')

    train_cv_file_name = "subsets/train_cv_set1"+ str(l)+".csv"
    test_cv_file_name = "subsets/test_cv_set1"+ str(l)+".csv"
    train_df.to_csv(train_cv_file_name)
    test_df.to_csv(test_cv_file_name)


tot_end = time()
print "Total calculation time {:.4f} seconds.".format(tot_end - tot_start)
