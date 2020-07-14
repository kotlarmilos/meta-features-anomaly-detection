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
db = Database("127.0.0.1","root","","anomaly_detection_decision_support")
datasets = utilities.get_datasets('/Users/miloskotlar/GoogleDrive/Academic/PhD/III/datasets/')
devices = utilities.get_devices()
methods = utilities.get_methods()

print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for evaluation:', len(datasets))

i = 0
for dataset in datasets:
    i+=1
    if (i!=3):
        continue
    print('*********************************************')
    print('Dataset: %s' % dataset['name'])

    print('Loading data...')
    features, target, anomaly_ratio = load_data(dataset)
    dataset['id'] = db.get_dataset_id(dataset)
    dataset['anomaly_entropy'] = str(anomaly_ratio)
    if not dataset['id']:
        ft = characterize_data(dataset, features, target)
        # dataset['id'] = db.insert_data_info(dataset, ft)
    # continue

    print('Size: %dx%d' % (features.shape[0], features.shape[1]))
    print('Anomaly ration: %f%%' % anomaly_ratio)

    print('Data standardization...')
    features = standardize_data(features)

    for method in methods:
        m = eval(method['name']+'()')
        params, headers = m.get_params(features)
        for p in params:
            try:
                dim = p[0]
                if features.shape[1] > dim:
                    print('Data dimension reduction from %d to %d...' % (features.shape[1], dim))
                    r_features = dimension_reduction(features, dim)
                else:
                    dim = features.shape[1]
                    r_features = features

                print('Fitting model to data...')
                # t1_start = time.perf_counter()
                # result = m.evaluate(r_features, target, anomaly_ratio, p)
                # t1_stop = time.perf_counter()
                # db.insert_evaluation_info('CPU', method,dataset, p, headers, t1_stop-t1_start, result[0])
            except:
                print("An error occurred for dataset %s and method %s and parameters %s"
                      % (dataset['name'], method['name'], np.concatenate((headers, p))))
            # break

print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# df = pd.read_csv('~/Desktop/brest cancer evaluation.csv', sep=';', error_bad_lines=False, warn_bad_lines=False, index_col=None)
# # data = df[df['name'] == 'Gaussian']
# # plt.plot(data['pca'],data['f1'], c='g', label='Gaussian method')
# # data = df[df['name'] == 'Linear']
# # plt.plot(data['pca'],data['f1'], c='r',  label='Linear regression')
# # data = df[df['name'] == 'RPCA']
# # plt.plot(data['pca'],data['f1'], c='b',  label='rPCA')
# # plt.legend()
# # plt.xlabel('dimensions')
# # plt.ylabel('f1 score')
# # plt.show()
# for k in range(2, 50, 5):
#     data = df[(df['name'] == 'KMeans') & (df['k'] == k)]
#     plt.plot(data['pca'],data['f1'], label='KMeans(%d)' % k)
#     plt.xlabel('dimensions')
#     plt.ylabel('f1 score')
#
#
# plt.legend()
# plt.show()
#
# # from r_pca import R_pca
#
#
#
# #
# # # generate low rank synthetic data
# # N = 100
# # num_groups = 3
# # num_values_per_group = 40
# # p_missing = 0.2
# #
# # Ds = []
# # for k in range(num_groups):
# #     d = np.ones((N, num_values_per_group)) * (k + 1) * 10
# #     Ds.append(d)
# #
# # D = np.hstack(Ds)
# #
# # # decimate 20% of data
# # n1, n2 = D.shape
# # S = np.random.rand(n1, n2)
# # D[S < 0.2] = 0
# #
# # # use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
# # rpca = R_pca(D)
# # L, S = rpca.fit(max_iter=10000, iter_print=100)
# #
# # # visually inspect results (requires matplotlib)
# # rpca.plot_fit()
# # plt.show()
#
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn import datasets, linear_model
# # from sklearn.metrics import mean_squared_error, r2_score
# #
# #
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# #
# # # Load the diabetes dataset
# # diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# #
# # # Use only one feature
# # diabetes_X = diabetes_X[:, np.newaxis, 2]
# #
# # # Split the data into training/testing sets
# # diabetes_X_train = diabetes_X[:-20]
# # diabetes_X_test = diabetes_X[-20:]
# #
# # # Split the targets into training/testing sets
# # diabetes_y_train = diabetes_y[:-20]
# # diabetes_y_test = diabetes_y[-20:]
# #
# # # Create linear regression object
# # regr = linear_model.LinearRegression()
# #
# # # Train the model using the training sets
# # regr.fit(diabetes_X_train, diabetes_y_train)
# #
# # # Make predictions using the testing set
# # diabetes_y_pred = regr.predict(diabetes_X_test)
# #
# # features = tf.constant(diabetes_X_train,dtype=tf.float32)
# # target = tf.constant(diabetes_y_train,dtype=tf.float32)
# # ftest = tf.constant(diabetes_X_test,dtype=tf.float32)
# # ttest = tf.constant(diabetes_y_test,dtype=tf.float32)
# #
# #
# # model = Sequential([
# #     Dense(1, activation='relu', input_shape=[features.shape[1]]),
# #   ])
# # # model = Sequential()
# # # model.add(Dense(1, input_dim=features.shape[1], activation='relu'))
# #
# # model.compile(loss='mean_squared_error',
# #               optimizer=tf.keras.optimizers.RMSprop(0.3),
# #               metrics=['mean_absolute_error', 'mean_squared_error'])
# #
# # history = model.fit(
# #   features, target,
# #   epochs=500)
# #
# # # model.fit(features, target)
# # weights = tf.transpose(model.get_weights()[0])
# # bias = model.get_weights()[1].flatten()
# # test_predictions = model.predict(features).flatten()
# # probs = tf.math.divide(
# #     tf.math.abs(tf.math.subtract(tf.math.add(tf.reduce_sum(tf.multiply(weights, features), axis=1), bias), target)),
# #     tf.math.sqrt(tf.reduce_sum(tf.multiply(weights, weights), axis=1)))
# #
# # # predictions = weights[0][0] * features[0] + weights[0][1]
# # print(probs)
# # plt.scatter(features.numpy().flatten(), target.numpy().flatten(), c=probs)
# # plt.plot(features.numpy().flatten(), test_predictions, color='blue', linewidth=3)
# # #
# # # target = target.numpy().flatten()
# # # stepsize = (max(probs) - min(probs)) / 1000
# # # epsilons = np.arange(min(probs), max(probs), stepsize)
# # # for epsilon in np.nditer(epsilons):
# # #     prediction = (probs < epsilon)
# # #     if len(set(prediction)) == 1:
# # #         continue
# # #     acc = accuracy_score(target, prediction)
# # #     prec = precision_score(target, prediction, labels=[0,1])
# # #     recall = recall_score(target, prediction, labels=[0,1])
# # #     f1 = f1_score(target, prediction, labels=[0,1])
# # #     # roc_auc = roc_auc_score(target, prediction)
# # #     if acc > best_scores['acc']['scores']['acc']:
# # #         best_scores['acc']['scores']['acc'] = acc
# # #         best_scores['acc']['scores']['prec'] = prec
# # #         best_scores['acc']['scores']['recall'] = recall
# # #         best_scores['acc']['scores']['f1'] = f1
# # #         best_scores['acc']['epsilon'] = epsilon
# # #     if prec > best_scores['prec']['scores']['prec']:
# # #         best_scores['prec']['scores']['acc'] = acc
# # #         best_scores['prec']['scores']['prec'] = prec
# # #         best_scores['prec']['scores']['recall'] = recall
# # #         best_scores['prec']['scores']['f1'] = f1
# # #         best_scores['prec']['epsilon'] = epsilon
# # #     if recall > best_scores['recall']['scores']['recall']:
# # #         best_scores['recall']['scores']['acc'] = acc
# # #         best_scores['recall']['scores']['prec'] = prec
# # #         best_scores['recall']['scores']['recall'] = recall
# # #         best_scores['recall']['scores']['f1'] = f1
# # #         best_scores['recall']['epsilon'] = epsilon
# # #     if f1 > best_scores['f1']['scores']['f1']:
# # #         best_scores['f1']['scores']['acc'] = acc
# # #         best_scores['f1']['scores']['prec'] = prec
# # #         best_scores['f1']['scores']['recall'] = recall
# # #         best_scores['f1']['scores']['f1'] = f1
# # #         best_scores['f1']['epsilon'] = epsilon
# # #
# # # # find metrics and for estimated epsilon based on anomaly percentage
# # # outliers = np.argpartition(probs, math.ceil(len(target) * anomaly_ratio))[:math.ceil(len(target) * anomaly_ratio)]
# # # prediction = np.zeros(len(target))
# # # for x, y in zip(outliers, prediction):
# # #     prediction[x] = 1
# # #
# # # acc = accuracy_score(target, prediction)
# # # prec = precision_score(target, prediction, labels=[0, 1])
# # # recall = recall_score(target, prediction, labels=[0, 1])
# # # f1 = f1_score(target, prediction, labels=[0, 1])
# # #
# # # best_scores['manual'] = {'epsilon': max(probs[outliers]), 'scores':{'acc':acc, 'prec':prec, 'recall':recall, 'f1':f1}}
# # # return best_scores
# #
# # # probs = []
# # # for item in features.numpy().flatten():
# # #     probs.append(weights[0] * item + b)
# #
# # # #
# # # # probs = tf.math.divide(
# # # #     tf.math.abs(tf.math.add(tf.reduce_sum(tf.multiply(weights, features), axis=1), model.get_weights()[1])),
# # # #     tf.math.sqrt(tf.reduce_sum(tf.multiply(weights, weights), axis=1)))
# # # #
# # # # predictions = weights[0][0] * features[0] + weights[0][1]
# # #
# # #
# # #
# # # # The coefficients
# # # print('Coefficients: \n', regr.coef_)
# # # # The mean squared error
# # # print('Mean squared error: %.2f'
# # #       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # # # The coefficient of determination: 1 is perfect prediction
# # # print('Coefficient of determination: %.2f'
# # #       % r2_score(diabetes_y_test, diabetes_y_pred))
# # #
# # # # Plot outputs
# # # plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# # # plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
# # #
# # # plt.xticks(())
# # # plt.yticks(())
# #
# # plt.show()