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
datasets, evaluation, features = db.get_datasets()

characterization_columns = np.array(features.columns)
features = features.to_numpy()
features = standardize_data(features)


temporal = np.array(datasets.loc[datasets['type_of_data'] == '\'temporal\''].axes[0])
temporal_features = features[temporal]
# spatial = np.array(datasets.loc[datasets['type_of_data'].str.contains('\'spatial\'')].axes[0])
# spatial_features = features[spatial]
high = np.array(datasets.loc[datasets['type_of_data'] == '\'high-dimensional\''].axes[0])
high_features = features[high]


total = 52 #9


# features = high_features
features = temporal_features
print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for testing:', len(features))

# plt.plot()
m = eval('KMeans()')
f = []
t = []

for i, dataset in datasets.iterrows():
    # if i>=len(features):
    #     break

    if i<10 or i>62:
        continue
    test_features = features[i-10]
    #
    # if i>62:
    #     continue

#
# # test_features = features[i]
# test_features.shape = (1,len(features[0])) #2
# train_features = features
    f.append(test_features)
    actual_score, actual_method, actual_params = m.crossval(i, datasets, evaluation)
    # for method in np.unique(actual_method):
    #     f.append(test_features)
    #     t.append(method)

    t.append(actual_method[0])

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
    model.add(Dense(93, input_dim=93, activation='relu'))
    Dense(64, activation='relu'),
    model.add(Dense(1, activation='softmax'))#sigmoid

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
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