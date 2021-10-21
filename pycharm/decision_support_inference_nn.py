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

datasets = global_datasets
features = global_features.to_numpy()

total = len(datasets)

features = standardize_data(features)

print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for testing:', len(features))

# plt.plot()
m = eval('KMeans()')
f = []
t = []


for i in range(len(datasets)):
    test_features = features[i]
    test_features.shape = (1, len(features[0]))
    train_features = features

    try:
        actual_score, actual_method, actual_params = m.crossval(i, datasets, evaluation)
        # for method in np.unique(actual_method):
        #     f.append(test_features)
        #     t.append(method)


        f.append(test_features[0])
        t.append(actual_method[0])
    except:
        print('Error')

print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

features = np.array(f)
target = np.array(t)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

X = features
Y = target
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

acc_per_fold = []
loss_per_fold = []
fold_no = 1
kfold = KFold(10)
for train, test in kfold.split(X, encoded_Y):
    model = Sequential()
    model.add(Dense(94, input_dim=94, activation='relu'))
    Dense(64, activation='relu'),
    model.add(Dense(1, activation='softmax'))#sigmoid

    model.compile(
        optimizer='adam',
        loss='mse',#categorical_crossentropy
        metrics=['accuracy'],
    )

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')


    # Train the model.
    history = model.fit(
        X[train], encoded_Y[train],
      epochs=5,
      batch_size=32,
    )

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
    #     tf.keras.metrics.MeanSquaredError(),
    #     tf.keras.metrics.AUC(),
    #     tf.keras.metrics.Precision(),
    #     tf.keras.metrics.Recall(),
    #     tf.keras.metrics.Accuracy()
    # ])

    # Generate generalization metrics
    scores = model.evaluate(X[test], encoded_Y[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1


# model.fit(X, encoded_Y)
# results = model.evaluate(X, encoded_Y)
# results = estimator.score(X, Y)
# results = estimator.predict(X)
# results = estimator.get_params()

# test_loss, test_acc = model.evaluate(X_test, y_test)
# print('Test accuracy:', test_acc)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(model, X, encoded_Y, cv=kfold)
# print(results)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')