### Load the safe training samples and record in a single file the features calculated

# Import python libraries
import numpy as np
import pandas as pd

# Get specific functions from some other python libraries
from math import floor, log
from scipy.io import loadmat   # For loading MATLAB data (.dat) files
from converter  import calculate_features
from os import listdir
from time import time


# Open the train_and_test_label .csv file listing the files safe to use
train_and_test_labels = pd.read_csv("/Users/jcq/Code/Kaggle/NIH seizure prediction/train_data_labels_safe.csv")

# Create a list of all the filenames present in the folder train
filenames = sorted(listdir("/Users/jcq/Code/Kaggle/NIH seizure prediction/train"))

# Create a list of the features for the analysis, equal to the functions list
functions = [ 'shannon entropy', 'correlation matrix (channel)', 'correlation matrix (frequency)', 'shannon entropy (dyad)', 'hjorth activity', 'hjorth mobility', 'hjorth complexity', 'skewness', 'kurtosis', 'Katz FD', 'Higuchi FD']
functions2 = [ 'SpectralEdgeFreq', 'PSD_Band', 'Autocorr matrix (epoch)','Ent rate of Change']
feat_stats = ['absMax', 'absMin', 'Max', 'Min', 'avg', 'var']
functions_list = ['index','class']
for function in functions:
	for feat_stat in feat_stats:
		functions_list.append(function+feat_stat)
for function2 in functions2:
	if function2 == 'SpectralEdgeFreq' or function2 == 'Ent rate of Change':
		functions_list.append(function2+"1")
	if function2 == 'Autocorr matrix (epoch)':
		functions_list.append(function2+"1")
		functions_list.append(function2+"2")
	if function2 == 'PSD_Band':
		for l in range(7):
			functions_list.append(function2+str(l+1))
#print functions_list [debug]


# Create the dataFrame model that will store the features for all the epochs analyzed
feat_dataframe = pd.DataFrame(np.zeros((1,79)), index = ['a'], columns=functions_list)
# Create the dataFrame model to store temporarily the features for all the epochs analyzed
temp_feat_dataframe = pd.DataFrame(np.zeros((16,79)), columns=functions_list)

# Create a dictionary to collect the modified features
modif_feat = dict()

start = time()

j = 0 # count the number of safe files trated
k = 0 # count the progress through the file list

for i in range(2431,2631,1): # full is range 1 to 1301 for the file 1, should be len(filenames) 1407

	k = k+1

	if k == len(filenames):
		print "Complete"

	else:
		k += 1

		if train_and_test_labels['safe'][i] == 1:
			print "Essai concluant", train_and_test_labels['image'][i]

			feat = dict() # Reinitialize the dict feat for each .mat files

			###################################### MODIFIED for drop treatment
			# Verify whether the sample contain sufficient data without drop-off
			valid = True
			feat, valid = calculate_features("/Users/jcq/Code/Kaggle/NIH seizure prediction/train/"+train_and_test_labels['image'][i])
			
			# Calculate the features using the converter code
			if valid: ###################################### MODIFIED for drop treatment
				# Calculate the statistics values on the features listed in "functions"
				for item in functions:
					for feat_stat1 in feat_stats:
						if feat_stat1 == 'avg':
							dfavg = (pd.DataFrame(feat[item].mean(axis = 0, numeric_only = float).append(pd.Series([feat_stat1], index = ['feat_stat']))).T).set_index('feat_stat')
						if feat_stat1 == 'absMax':
							dfabsMax = (pd.DataFrame(feat[item].mean(axis = 0, numeric_only = float).append(pd.Series([feat_stat1], index = ['feat_stat']))).T).set_index('feat_stat')
						if feat_stat1 == 'absMin':
							dfabsMin = (pd.DataFrame(feat[item].mean(axis = 0, numeric_only = float).append(pd.Series([feat_stat1], index = ['feat_stat']))).T).set_index('feat_stat')
						if feat_stat1 == 'Max':
							dfMax = (pd.DataFrame(feat[item].mean(axis = 0, numeric_only = float).append(pd.Series([feat_stat1], index = ['feat_stat']))).T).set_index('feat_stat')
						if feat_stat1 == 'Min':
							dfMin = (pd.DataFrame(feat[item].mean(axis = 0, numeric_only = float).append(pd.Series([feat_stat1], index = ['feat_stat']))).T).set_index('feat_stat')
						if feat_stat1 == 'var':
							dfvar = (pd.DataFrame(feat[item].mean(axis = 0, numeric_only = float).append(pd.Series([feat_stat1], index = ['feat_stat']))).T).set_index('feat_stat')
					modif_feat[item] = pd.concat([dfavg, dfabsMax, dfabsMin, dfMax, dfMin, dfvar])

				# Register the features calculated for each channel of each epoch in a DataFrame
				for channel in range(16): # iterate through the number of channels
					# Record the position of the sample in the dataframe as a 
					temp_feat_dataframe['class'][channel] = train_and_test_labels['class'][i]
					
					for key in functions: # iterate through all the features
						for feat_stat2 in feat_stats:
							temp_feat_dataframe[key+feat_stat2][channel] = modif_feat[key].loc[feat_stat2].loc[channel]
					for key2 in functions2:
						for m in range(feat[key2].shape[0]):
							temp_feat_dataframe[key2+str(m+1)][channel] = feat[key2].iloc[m].iloc[channel]
					temp_feat_dataframe['class'][channel] = train_and_test_labels['class'][i]

					temp_feat_dataframe['index'][channel] = int(j * (16) + channel + 1)
				feat_dataframe = pd.concat([feat_dataframe, temp_feat_dataframe])			
				j+=1

	if j == 100 or j == 200 or j ==300 or j == 400 or j == 500 or j == 600 or j == 700 or j == 800 or j == 900 or k == len(filenames): # Condition with j controls the number of test in each saved .csv files, should be 100
		file_name = "/Users/jcq/Code/Kaggle/NIH seizure prediction/feat_dataframe_" + str(i) + ".csv"
		feat_dataframe = feat_dataframe.drop('a', axis = 0)
		feat_dataframe = feat_dataframe.set_index('index')
		#print feat_dataframe
		feat_dataframe.to_csv(file_name)
		# Reinitialize the feat dataframe each time a file of 100 examples is recorded or when the last sample has been treated
		feat_dataframe = pd.DataFrame(np.zeros((1,79)), index = ['a'], columns=functions_list)

# Save the pandas DataFrame containing the features in a .CSV file in the case that the limit range is inferiof to the filenames lenght 
file_name = "/Users/jcq/Code/Kaggle/NIH seizure prediction/non_comp_feat_dataframe_" + str(i) + ".csv"
feat_dataframe = feat_dataframe.drop('a', axis = 0)
feat_dataframe = feat_dataframe.set_index('index')
feat_dataframe.to_csv(file_name)
end = time()
print "Number of safe examples", j
print "Transformed the files in {:.4f} seconds.".format(end - start)
