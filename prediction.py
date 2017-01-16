### Implement bagging for 3 classifier models ###
### Combine the prediction by averaging the probalities from each model ###
# Require the models predictions to be stored in the folder "predictions".

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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

tot_start = time()

######################################
# open the predictions .csv file from 3 models
frames = [ pd.read_csv("/predictions/predictions_model"+str(f)+".csv") for f in range(1,4,1) ]
frames[1].drop(['File', 'Class'], axis = 1, inplace = True)
frames[2].drop(['File', 'Class'], axis = 1, inplace = True)

#create the dataframe that will contain all the features and labels
df = pd.concat(frames, axis = 1)

# Create the dictionary that will store the name and prediction for each sample file
dic_final = {'File' : [],
            'Class' : np.zeros(df.shape[0])
    }

# Fill the class for each file in the final dataframe
for i in range(df.shape[0]):
    dic_final['File'].append(df['File'][i])
    if df['Proba1'][i] != 0:
    	# Average the prediction probabilities for the 3 model, treshold is set to 0.5
        dic_final['Class'][i] = round(np.mean([df['Proba2'][i], df['Proba2'][i], df['Proba3'][i]]))

# Record the final prediction dataframe in a .csv file
file_name = "/predictions/predictions.csv"
df_final = pd.DataFrame(dic_final, columns = ['File', 'Class'], index = range(df.shape[0]))
df_final = df_final.set_index(['File'])
df_final.to_csv(file_name)

#print "ROC score", roc_auc_score(df['Class'], dic_final['Class'], average = 'macro')
print "ROC score proba", roc_auc_score(df['Class'], df_final['Class'], average = 'macro')
print "F1 score for test set: {:.4f}.".format(f1_score(df['Class'], df_final['Class'], pos_label=1.0))
print "Recall score for test set: {:.4f}.".format(recall_score(df['Class'], df_final['Class'], pos_label=1.0, average='binary'))

tot_end = time()
print "Total calculation time {:.4f} seconds.".format(tot_end - tot_start)
