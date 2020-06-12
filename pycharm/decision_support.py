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

from scipy.stats import multivariate_normal

from scipy.stats import normaltest

from sklearn import datasets as dd




from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

db = Database("34.68.13.182","root","6g8HBIy0F8atEKtb","anomaly_detection_decision_support")
datasets = utilities.get_datasets('/Users/miloskotlar/GoogleDrive/Academic/PhD/III/linear_datasets/')
devices = utilities.get_devices()
methods = utilities.get_methods()

print('*** Datasets for evaluation:', len(datasets))

i = 0
for dataset in datasets:
    i+=1
    if (i<2):
        continue
    print('*********************************************')
    print('Dataset: %s' % dataset['name'])

    print('Loading data...')
    features, target, anomaly_ratio = load_data(dataset)

    print('Size: %dx%d' % (features.shape[0], features.shape[1]))
    print('Anomaly ration: %f%%' % anomaly_ratio)

    print('Data standardization...')
    features = standardize_data(features)

    reduced = True
    visualise = True

    min = 1
    max = features.shape[1]
    if reduced:
        max = 2
    for d in range(max, min, -1):
        if features.shape[1] > d:
            print('Data dimension reduction from %d to %d...' % (features.shape[1], d))
            r_features = dimension_reduction(features, d)
        else:
            r_features = features

        m = Linear()
        print('Fitting model to data...')
        best_scores, probs, test_predictions = m.evaluate(r_features, target, anomaly_ratio)

        print('++++++++Performance for %d dimensions++++++++' % d)
        print('     Threshold of %.2E gives the best accuracy %s' % (best_scores['acc']['epsilon'], best_scores['acc']['scores']))
        print('     Threshold of %.2E gives the best precision %s' % (best_scores['prec']['epsilon'], best_scores['prec']['scores']))
        print('     Threshold of %.2E gives the best recall %s' % (best_scores['recall']['epsilon'], best_scores['recall']['scores']))
        print('     Threshold of %.2E gives the best f1 %s' % (best_scores['f1']['epsilon'], best_scores['f1']['scores']))
        print('     Manually calculated threshold of %.2E gives %s f1 score' % (best_scores['manual']['epsilon'], best_scores['manual']['scores']))
        print('+++++++++++++++++++++++++++++++++++++++++++++')
        if d <= 2 and visualise:
            m.visualize_2d(dataset, r_features, target, probs, best_scores, test_predictions)