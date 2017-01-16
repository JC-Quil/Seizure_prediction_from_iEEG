### Process the features vectors through scaling ###
### Calculate the squared features ###
### Note: feature scaling should be done per patient and per channel ###
# Require the feature tables to be stored in the folder "features_tables" and the folder "processed_features_tables"
# to store the tables containing the processed features.

# Import python libraries
import numpy as np
import pandas as pd
import pylab

# Get specific functions from some other python libraries
from os import listdir
from sklearn.decomposition import PCA
from sklearn import cross_validation
from math import isnan
from IPython.display import display # Allows the use of display() for DataFrames
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler

######################################
# Function for scatter plot visualization of the features
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

    pylab.savefig(name + ".png")
######################################


######################################
# Create a list of all the filenames present in the folder containing the features tables
list_features_tables = sorted(listdir("/features_tables"))
if '.DS_Store' in list_features_tables:
    list_features_tables.remove('.DS_Store')
#print "list", list_features_tables [debug]

# open the features .csv file and store them in a DataFrame df
frames = [ pd.read_csv("/features_tables/"+f) for f in list_features_tables ]
df = pd.concat(frames, axis = 0)
df.sort_values('index', axis = 0, inplace=True)

# Remove rows where value is NaN
col = list(df.columns.values)
col.remove('class')
col.remove('index')
df = df.dropna(subset = col)
######################################


######################################
# This code add the squared features as features
df_pow = df.drop(['class', 'index'], axis = 1)
pow2 = lambda x: x**2
df_pow.applymap(pow2)
for i in range(len(col)):
    col[i] = col[i]+"_pow"
df_pow.columns = col
df = pd.concat([df, df_pow], axis = 1)
######################################


######################################
# Initiate th MinMaxScaler
min_max_scaler = MinMaxScaler()

# Apply the MinMaxScaler() function to all the features for each channel
for i in range(16):
    df_channel = df[((df['index']-1)%320)%16 == i]
    print "df_channel", df_channel[df_channel['class'] != 0].shape
    df_channel.set_index('index', inplace = True)
    df_labels = df_channel.pop('class')
    
    df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df_channel), index = df_channel.index, columns=df_channel.columns)
    #display(df_scaled.describe()) #[debug]
    
    df_channel = pd.concat([df_scaled, df_labels], axis = 1) # Recombine the class with the dataframe
    
    # Save the pandas DataFrame containing the processed features in a .CSV file for each channels
    file_name = "/processed_features_tables/processed_features_ch"+ str(i+1) +".csv"
    df_channel.to_csv(file_name)
######################################