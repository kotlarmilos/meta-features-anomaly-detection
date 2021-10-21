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

# db = Database("34.68.13.182","root","6g8HBIy0F8atEKtb","anomaly_detection_decision_support")
db = Database("127.0.0.1", "root", "", "anomaly_detection_decision_support")
# db.truncate_database()
datasets = utilities.get_datasets('/Users/miloskotlar/GoogleDrive/Academic/PhD/III/datasets/')
devices = utilities.get_devices()
methods = utilities.get_methods()

print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for evaluation:', len(datasets))

i = 0
for dataset in datasets:
    i += 1
    if (dataset['name']!='Breast Cancer Diagnostic'):
        continue
    print('*********************************************')
    print('Dataset: %s' % dataset['name'])

    print('Loading data...')
    features, target, anomaly_ratio = load_data(dataset)
    # dataset['id'] = db.get_dataset_id(dataset)
    # dataset['anomaly_entropy'] = str(anomaly_ratio)
    # if not dataset['id']:
    #     ft = characterize_data(dataset, features, target)
    #     dataset['id'] = db.insert_data_info(dataset, ft)
    # continue

    print('Size: %dx%d' % (features.shape[0], features.shape[1]))
    print('Anomaly ration: %f%%' % anomaly_ratio)

    print('Data standardization...')
    features = standardize_data(features)

    for method in methods:
        if method['name'] != 'AutoencoderModel':
            continue
        m = eval(method['name'] + '()')
        params, headers = m.get_params(features)
        for p in params:
            # try:
            dim = 2
            if features.shape[1] > dim:
                print('Data dimension reduction from %d to %d...' % (features.shape[1], dim))
                r_features = dimension_reduction(features, dim)
            else:
                dim = features.shape[1]
                r_features = features

            print('Fitting model to data...')
            best_scores, probs = m.evaluate(r_features, target, anomaly_ratio, p)
            m.visualize_2d(dataset, r_features, target, probs, best_scores, 2)
            # except:
            #     print("An error occurred for dataset %s and method %s and parameters %s"
            #           % (dataset['name'], method['name'], np.concatenate((headers, p))))
            break

print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
