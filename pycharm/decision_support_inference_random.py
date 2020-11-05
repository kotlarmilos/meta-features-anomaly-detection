from sklearn.preprocessing import StandardScaler
from database import Database
from preprocessing import *
import utilities
import numpy as np
import os
import time
from itertools import combinations
from algorithms.kmeans import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold


from random import choices
population = ['Gaussian', 'Linear', 'RPCA', 'KMeans', 'AutoencoderModel']
weights = [0.024, 0.208, 0.16, 0.44, 0.168]

db = Database("127.0.0.1","root","","anomaly_detection_decision_support")
# db = Database("anomaly-detection-mysql.ch1ih3mzagsi.eu-central-1.rds.amazonaws.com","admin","6xy4AMtnhkFJWfIWHsuu","anomaly_detection_decision_support")
global_datasets, evaluation, global_features = db.get_datasets()

# for dataset in utilities.get_datasets('/Users/miloskotlar/GoogleDrive/Academic/PhD/III/linear_datasets/'):
#     db.update_characterization_user_defined_data(dataset)

# exit(1)
characterization_columns = np.array(global_features.columns)
characterization_columns.shape = (1, global_features.shape[1])

user_defined_characterization_columns = ['high_dimensional',
'nominal',
'spatial',
'temporal',
'graphs_and_networks',
'high-dimensional',
'manufacturing',
'transport',
'finance',
'medicine',
'images',
'text',
'software',
'social',
'local',
'global',
'cluster',
'anomaly_space',
'anomaly_ratio',]
predefined_characterization_columns = np.setdiff1d(characterization_columns,user_defined_characterization_columns)

characterization_filters=['all', 'temporal', 'high_dimensional', 'manufacturing', 'tranport','finance',
'medicine',
'images',
'text',
'software',
'social',
'local',
'global',
'cluster']

characterization_attributes=['all', 'prefedined', 'user_defined']


global_features = global_features.loc[global_features['high-dimensional'] == 1]
datasets = global_datasets.loc[global_datasets['id'].isin(global_features.index.to_numpy())]


# datasets = global_datasets
features = global_features.to_numpy()

total = len(datasets)

features = standardize_data(features)

print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for testing:', len(features))

# plt.plot()
m = eval('KMeans()')
f = []
t = []

method_match = 0

for i in range(len(datasets)):
    test_features = features[i]
    test_features.shape = (1, len(features[0]))
    train_features = features

    try:
        actual_score, actual_method, actual_params = m.crossval(i, datasets, evaluation)
        # for method in np.unique(actual_method):
        #     f.append(test_features)
        #     t.append(method)

        random_method = choices(population, weights)[0]

        if random_method in actual_method:
            method_match = method_match + 1
    except:
        print('Error')

print(method_match/total)
print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))