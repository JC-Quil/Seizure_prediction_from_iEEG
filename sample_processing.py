### This script command the processing of the labeled samples and the feature recording. ###
### Use the converter script to load samples, discard non-valid subsamples and to compute the features. ###
# Require the dataset to be stored in the folder "dataset" and the folder "features_tables"
# to store the feature tables.

# Import python libraries
import numpy as np
import pandas as pd

# Get specific functions from some other python libraries
from scipy.io import loadmat   # For loading MATLAB data (.dat) files
from converter import calculate_features
from os import listdir
from time import time


# open the train data label safe .csv file (list provided with the dataset)
train_data_list_df = pd.read_csv("train_and_test_data_labels_safe.csv")

# Create a list of all the filenames present in the folder "train" containing the labeled samples
filenames = sorted(listdir("dataset"))

# Create the list of features for the data processing based on the converter functions list, plus
# the index and class categories
functions_7 = [ 'shannon entropy', 'power spectral density', 'PSD wavelet', 'correlation matrix (frequency)']
functions_1 = ['mean', 'std dev','spectral edge frequency', 'correlation matrix (channel)', 'hjorth activity', 'hjorth mobility', 'hjorth complexity', 'skewness', 'kurtosis', 'Katz FD', 'Higuchi FD']
functions_list = ['index','class']
for function in functions_7: # Those features are calculated on 7 frequencys bands
	for feat_no in range(0,7,1):
		functions_list.append(function+str(feat_no))
for function in functions_1:
	functions_list.append(function)
print len(functions_list) #[debug]


# Create the dataframe that will store the features for all the subsamples analyzed
feat_dataframe = pd.DataFrame(np.zeros((1,len(functions_list))), index = ['a'], columns=functions_list)

# Create the dataframe to store temporarily the features from one sample analyzed
temp_feat_dataframe = pd.DataFrame(np.zeros((16,len(functions_list))), columns=functions_list)

# Create a dictionary to collect the features dictionary from function calculate_features
feat_dic = dict()

# Record the processing start time
start = time()

j = 0 # count the number of safe files treated
k = 0 # count the progress through the file list

for i in range(3648,6042,1): # Allow to subdivize the sample processing

	k += 1

	if k == len(filenames):
		print "Complete"

	else:

		if train_data_list_df['safe'][i] == 1:
			print "Essai concluant", train_data_list_df['image'][i]

			feat = dict() # Reinitialize the dict feat for each .mat files

			# Calculate the features from sample i using the converter code			
			feat = calculate_features("train"+train_data_list_df['image'][i])
			
			# Iterate through the subsamples and channels to record each feature in a dataframe
			for epoch in range(int(feat['mean'].size/16)): 
				for channel in range(16):
					for item in functions_list:
						if item == 'class':
							temp_feat_dataframe[item][channel] = train_data_list_df['class'][i]
						elif item == 'index':
							temp_feat_dataframe[item][channel] = int(i * (16*20) + (channel+1) + 16 * (epoch))
						else:
							temp_feat_dataframe[item][channel] = feat[item].iloc[epoch].loc[channel]


				feat_dataframe = pd.concat([feat_dataframe, temp_feat_dataframe])			
			j+=1

# Save the pandas DataFrame containing the features in a .CSV file
file_name = "features_tables/feat_dataframe_" + str(i) + ".csv"
feat_dataframe = feat_dataframe.drop('a', axis = 0)
feat_dataframe = feat_dataframe.set_index('index')
#print feat_dataframe
feat_dataframe.to_csv(file_name)

print "Number of safe examples", j

end = time()
print "Transformed the files in {:.4f} seconds.".format(end - start)
print feat_dataframe.shape
