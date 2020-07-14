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
datasets, evaluation, features = db.get_datasets()

features = features.to_numpy()
features = standardize_data(features)
features = dimension_reduction(features, 2)

fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

ax = fig.add_subplot(1, 1, 1)
ax.scatter(features[:,0], features[:,1], c='b', s=50)

print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for testing:', len(datasets))

methods = utilities.get_methods()

for i, dataset in datasets.iterrows():
    test_features = features[i]
    train_features = features
    for method in methods:
        m = eval(method['name']+'()')
        print('Finding closest model to data for %s...' % dataset['name'])
        best_score, best_method, best_params = m.distance(test_features, train_features, datasets, evaluation)
        print('Best f1 score is %f with %s and parameters %s' % (best_score, best_method, best_params))
        actual_score, actual_method, actual_params = m.crossval(i, datasets, evaluation)
        print('Actual f1 score is %f with %s and parameters %s' % (actual_score, actual_method, actual_params))

        expected_score, expected_method, expected_params = m.predict(i, datasets, evaluation, best_method, best_params)
        print('Expected f1 score would be %f with %s and parameters %s' % (expected_score, expected_method, expected_params))


print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))