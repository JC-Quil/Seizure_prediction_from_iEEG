### Process the training samples features through scaling, PCA is optional ###

# Import python libraries
import numpy as np
import pandas as pd
import pylab
#import renders as rs

# Get specific functions from some other python libraries
from os import listdir
from sklearn.decomposition import PCA
from sklearn import cross_validation
from math import isnan
from IPython.display import display # Allows the use of display() for DataFrames
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler

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


######################################
# Create a list of all the filenames present in the folder containing the features tables
list_features_tables = sorted(listdir("/Users/jcq/Code/Kaggle/NIH seizure prediction/features_tables"))
list_features_tables.remove('.DS_Store')
#print "Features tables filenames", list_features_tables

#create the dataframe that will contain all the features and lables
df = pd.DataFrame()

# open the features .csv file and store them in a DataFrame df
frames = [ pd.read_csv("/Users/jcq/Code/Kaggle/NIH seizure prediction/features_tables/"+f) for f in list_features_tables ]
df = pd.concat(frames, axis = 0)
df.set_index('index', inplace = True)
#df_labels = df.pop(['0'])
#print "feature table", df.head(n=20)
#print "feature", df.shape
# Remove rows where value is null or NaN
col = list(df.columns.values)
#print "list_columns", len(col), col
col.remove('class')
#print "list_columns", len(col), col
df = df.dropna(subset = col)
#print "feature", df.shape
df = df[(df.filter(items=col).T != 0).any()]
#@print "feature", df.shape
# Store the labels in the DataFrame df_labels
df_labels = df.pop('class')


print "feature not null", df.shape
#print "ShannonEntropy min", df['shannon entropy'].min()

# Display a description of the dataset
display(df.describe())
######################################

######################################
# Outlier detection
# For each feature find the data points with extreme high or low values
df_outliers = pd.DataFrame()

for feature in df.keys():
    if feature != 'class':
        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(df[feature], 25)
        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(df[feature], 75)
        # Use the interquartile range to calculate an outlier step (2 times the interquartile range)
        step = 4 * (Q3 - Q1)
    
        # Display the outliers
        print "Data points considered outliers for the feature '{}':".format(feature)
        print "Lower limit {}, higher limit {}.".format(Q1 - step, Q3 + step)

        #df_outliers = df_outliers.append(df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))])
        df_outliers = df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))]
        #result = df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))]
        print "Outliers", feature, df_outliers.shape
######################################

######################################
# Scale the features using Min_Max method
min_max_scaler = MinMaxScaler()
df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), index = df.index, columns=df.columns)
display(df_scaled.describe())
######################################

######################################
# Recombine the class with the dataframe
df = pd.concat([df_scaled, df_labels], axis = 1)
######################################

#### make_scatter_plot(df_scaled, "complete_scaled") #####

######################################
# Features removed due to the large quantity of outliers
# df_scaled.drop(['correlation matrix (frequency)', 'Hjorth FD'], axis = 1)
######################################

#### make_scatter_plot(df_scaled, "dropped_scaled") #####

######################################
# PCA with multiple components number
#n_components = [8, 9, 10, 11, 12]  ### Chose the number of components to use

# Code de remplacement pour effectuer la recherche sur plusieurs valeurs de PCs
#pca_ = []
#pca_df_ = []

#for components in range(len(n_components)):

#  pca_.append(PCA(n_components=int(n_components[components]), whiten=True).fit(df_scaled))
#  pca = pca_[components]

#  print "Components", components
#  print "Explained Variance ratio", pca.explained_variance_ratio_
#  print "Total Explained Variance", pca.explained_variance_ratio_.cumsum()
###########################

###########################
# Save the pandas DataFrame containing the processed features in a .CSV file
file_name = "/Users/jcq/Code/Kaggle/NIH seizure prediction/processed_features_tables/processed_features.csv"
print df.shape
df.to_csv(file_name)
###########################