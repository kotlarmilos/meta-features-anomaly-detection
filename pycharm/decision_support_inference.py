from sklearn.preprocessing import StandardScaler

from database import Database
from preprocessing import *
import utilities
import numpy as np
import tensorflow as tf
import os
import algorithm as alg
import tensorflow_probability as tfp
from sklearn.metrics import f1_score

import math
from algorithms.gaussian import Gaussian
from algorithms.linear import Linear
from algorithms.kmeans import KMeans
from algorithms.autoencoder import AutoencoderModel
from algorithms.rpca import RPCA

from scipy.stats import multivariate_normal
import time

from scipy.stats import normaltest

from sklearn import datasets as dd




from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

db = Database("127.0.0.1","root","","anomaly_detection_decision_support")
# db = Database("35.234.97.110","root","6g8HBIy0F8atEKtb","anomaly_detection_decision_support")
datasets, evaluation, features = db.get_datasets()

features = features.to_numpy()
features = standardize_data(features)
features = dimension_reduction(features, 2)

# fig = plt.figure(figsize=(12, 12))
# plt.subplots_adjust(hspace=0.5)


temporal = np.array(datasets.loc[datasets['type_of_data'] == '\'temporal\''].axes[0])
temporal_features = features[temporal]
high = np.array(datasets.loc[datasets['type_of_data'] == '\'high-dimensional\''].axes[0])
high_features = features[high]
spatial = np.array(datasets.loc[datasets['type_of_data'].str.contains('\'spatial\'')].axes[0])
spatial_features = features[spatial]

# ax = fig.add_subplot(1, 2, 1)
# ax.scatter(high_features[:,0], high_features[:,1], c='r', s=50, label='high-dimensional')


# for i in range(len(high_features)):
#     text = datasets.loc[datasets['type_of_data'] == '\'high-dimensional\''].iloc[i]['name']
#     ax.annotate(text, (high_features[i][0], high_features[i][1]))
# # ax.scatter(temporal_features[:,0], temporal_features[:,1], c='b', s=50, label='temporal')
# # ax.scatter(spatial_features[:,0], spatial_features[:,1], c='g', s=50, label='spatial')
# plt.legend()
# plt.plot()

# features = temporal_features
features = high_features


print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for testing:', len(datasets))

methods = utilities.get_methods()
results = pd.DataFrame(columns=['Dataset_id','Dataset name','Best method','Score','Closest dataset','Closest best method','Closest best score','Proposed best method','Proposed best score','Method match','Precision'])

for i, dataset in datasets.iterrows():
    if i>=len(features):
        break

    # if i<10 or i>62:
    #     continue
    # test_features = features[i-10]

    # if i>62:
    #     continue
    test_features = features[i]
    test_features.shape = (1,2)
    train_features = features
    for method in methods:
        if method['name'] in ['AutoencoderModel']:#KMeans#Gaussian#Linear#RPCA
            m = eval(method['name']+'()')
            try:
                print('Finding closest model to using %s data for %s...' % (method['name'], dataset['name']))
                best_score, best_method, best_params, closest_dataset_name = m.distance(test_features, train_features, datasets, evaluation)
                print('Best f1 score for closest dataset is %f with %s and parameters %s' % (best_score, best_method, best_params))
                actual_score, actual_method, actual_params = m.crossval(i, datasets, evaluation)
                print('Best f1 score for test dataset is %f with %s and parameters %s' % (actual_score, actual_method, actual_params))
                expected_score, expected_method, expected_params = m.predict(i, datasets, evaluation, best_method, best_params)
                print('Expected f1 score would be %f with the proposed method %s ' % (expected_score, expected_method)) # and parameters %s , expected_params
                df = pd.DataFrame(
                    np.array([[dataset['id'], dataset['name'], actual_method,actual_score,closest_dataset_name, best_method, best_score, expected_method, expected_score, actual_method==expected_method, 1-(actual_score-expected_score) ]]),
                    columns=['Dataset_id','Dataset name','Best method','Score','Closest dataset','Closest best method','Closest best score','Proposed best method','Proposed best score','Method match','Precision'])
                results = results.append(df)
            except:
                print('Can\'t find evaluation')
results.to_csv('test.csv')
print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))