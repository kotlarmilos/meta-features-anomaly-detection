import math
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import utilities

class Linear:
    def __init__(self):
        pass

    def get_params(self, features):
        pca = list(reversed(range(2, features.shape[1]+1)))
        return utilities.make_cartesian([pca]), ['pca']

    def select_threshold(self, probs, target, anomaly_ratio):
        best_scores = {}
        best_scores['acc'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}
        best_scores['prec'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}
        best_scores['recall'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}
        best_scores['f1'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}

        # find best metrics and epsilons using test data
        stepsize = (max(probs) - min(probs)) / 1000
        epsilons = np.arange(min(probs), max(probs), stepsize)
        epsilons = epsilons[::-1]
        for epsilon in np.nditer(epsilons, order='C'):
            prediction = (probs > epsilon)
            if len(set(prediction)) == 1:
                continue
            acc = accuracy_score(target, prediction)
            prec = precision_score(target, prediction, labels=[0,1])
            recall = recall_score(target, prediction, labels=[0,1])
            f1 = f1_score(target, prediction, labels=[0,1])
            # roc_auc = roc_auc_score(target, prediction)
            if acc > best_scores['acc']['scores']['acc']:
                best_scores['acc']['scores']['acc'] = acc
                best_scores['acc']['scores']['prec'] = prec
                best_scores['acc']['scores']['recall'] = recall
                best_scores['acc']['scores']['f1'] = f1
                best_scores['acc']['epsilon'] = epsilon
            if prec > best_scores['prec']['scores']['prec']:
                best_scores['prec']['scores']['acc'] = acc
                best_scores['prec']['scores']['prec'] = prec
                best_scores['prec']['scores']['recall'] = recall
                best_scores['prec']['scores']['f1'] = f1
                best_scores['prec']['epsilon'] = epsilon
            if recall > best_scores['recall']['scores']['recall']:
                best_scores['recall']['scores']['acc'] = acc
                best_scores['recall']['scores']['prec'] = prec
                best_scores['recall']['scores']['recall'] = recall
                best_scores['recall']['scores']['f1'] = f1
                best_scores['recall']['epsilon'] = epsilon
            if f1 > best_scores['f1']['scores']['f1']:
                best_scores['f1']['scores']['acc'] = acc
                best_scores['f1']['scores']['prec'] = prec
                best_scores['f1']['scores']['recall'] = recall
                best_scores['f1']['scores']['f1'] = f1
                best_scores['f1']['epsilon'] = epsilon

        # find metrics and for estimated epsilon based on anomaly percentage
        outliers = np.argsort(probs)[-math.ceil(len(target) * anomaly_ratio):]
        outliers = outliers[::-1]

        # outliers = np.argpartition(probs, math.ceil(len(target) * anomaly_ratio))[:math.ceil(len(target) * anomaly_ratio)]
        prediction = np.zeros(len(target))
        for x, y in zip(outliers, prediction):
            prediction[x] = 1

        acc = accuracy_score(target, prediction)
        prec = precision_score(target, prediction, labels=[0, 1])
        recall = recall_score(target, prediction, labels=[0, 1])
        f1 = f1_score(target, prediction, labels=[0, 1])

        best_scores['manual'] = {'epsilon': min(probs[outliers]), 'scores':{'acc':acc, 'prec':prec, 'recall':recall, 'f1':f1}}
        return best_scores

    # def find_nearest(self, array, value, datasets, evaluation):
    #     array = np.asarray(array)
    #     # idx = (np.abs(array - value)).argmin()
    #     idx = np.argsort(np.abs(array-value))[1]
    #     closest_dataset_id = datasets.iloc[idx]['id']
    #     closest_dataset_name = datasets.iloc[idx]['name']
    #     print('Closest dataset is %s...' % closest_dataset_name)
    #     es = evaluation[evaluation['dataset_id'] == closest_dataset_id]
    #     best_score = es['f1'].max()
    #     method = es[es['f1'] == best_score]['method'].to_numpy()[0]
    #     params = {'pca': es[es['f1'] == best_score]['pca'].to_numpy()[0], 'k': es[es['f1'] == best_score]['k'].to_numpy()[0]}
    #     return best_score, method, params, closest_dataset_name
    #
    # def crossval(self, idx, datasets, evaluation):
    #     closest_dataset_id = datasets.iloc[idx]['id']
    #     es = evaluation[evaluation['dataset_id'] == closest_dataset_id]
    #     best_score = es['f1'].max()
    #     method = es[es['f1'] == best_score]['method'].to_numpy()[0]
    #     params = {'pca': es[es['f1'] == best_score]['pca'].to_numpy()[0], 'k': es[es['f1'] == best_score]['k'].to_numpy()[0]}
    #     return best_score, method, params
    #
    # def predict(self, idx, datasets, evaluation, best_method, best_params):
    #     # closest_dataset_id = datasets.iloc[idx]['id']
    #     # es = evaluation[(evaluation['dataset_id'] == closest_dataset_id)
    #     # & (evaluation['method'] == best_method)
    #     # & (evaluation['pca'] == best_params['pca'])
    #     # & (evaluation['k'] == best_params['k'])]
    #     #
    #     # score = es['f1'].to_numpy()[0]
    #     # return score, best_method, best_params
    #     closest_dataset_id = datasets.iloc[idx]['id']
    #     es = evaluation[(evaluation['dataset_id'] == closest_dataset_id)
    #                     & (evaluation['method'] == best_method)]
    #     # & (evaluation['pca'] == best_params['pca'])
    #     # & (evaluation['k'] == best_params['k'])
    #
    #     score = es['f1'].max()  # .to_numpy()[0]
    #     return score, best_method, best_params

    def find_nearest(self, k, array, value, datasets, evaluation):
        array = np.asarray(array)
        # idx = (np.abs(array - value)).argmin()
        idxs = np.argsort(np.abs(array - value))[1:k+1]
        # print(list(datasets.iloc[np.argsort(np.abs(array - value))]['name'].to_numpy()))
        # print(list(array[np.argsort(np.abs(array - value))]))
        # arr.append(datasets.iloc[np.argsort(np.abs(array - value))]['name'].to_numpy())
        # arr.append(array[np.argsort(np.abs(array - value))])
        # return 0
        result = []
        temp_array = []
        for idx in idxs:
            closest_dataset_id = datasets.iloc[idx]['id']
            closest_dataset_name = datasets.iloc[idx]['name']
            # print('Closest dataset is %s...' % closest_dataset_name)
            es = evaluation[evaluation['dataset_id'] == closest_dataset_id]
            best_score = es['f1'].max()
            method = es[es['f1'] == best_score]['method'].to_numpy()[0]
            result.append({'method': method, 'dataset_name': closest_dataset_name, 'distance': np.abs(array - value)[idx], 'best_score':best_score })
            temp_array.append(method)
            params = {'pca': es[es['f1'] == best_score]['pca'].to_numpy()[0],
                      'k': es[es['f1'] == best_score]['k'].to_numpy()[0]}


        # m, c = np.unique(result, return_counts=True)
        # idx = np.argmax(c)
        # return m[idx]
        return result, temp_array

    def crossval(self, idx, datasets, evaluation):
        closest_dataset_id = datasets.iloc[idx]['id']
        es = evaluation[evaluation['dataset_id'] == closest_dataset_id]
        best_score = es['f1'].max()
        method = es[es['f1'] == best_score]['method'].to_numpy()
        params = {'pca': es[es['f1'] == best_score]['pca'].to_numpy()[0],
                  'k': es[es['f1'] == best_score]['k'].to_numpy()[0]}
        return best_score, method, params

    def predict(self, idx, datasets, evaluation, best_method, best_params):
        closest_dataset_id = datasets.iloc[idx]['id']
        es = evaluation[(evaluation['dataset_id'] == closest_dataset_id)
                        & (evaluation['method'] == best_method)]
        # & (evaluation['pca'] == best_params['pca'])
        # & (evaluation['k'] == best_params['k'])

        score = es['f1'].max()  # .to_numpy()[0]
        return score, best_method, best_params

    def distance(self,k, test_features, train_features, datasets, evaluation):
        test_target_feature = tf.constant(test_features[:, test_features.shape[1]-1])
        test_features = tf.constant(np.delete(test_features, test_features.shape[1]-1, 1), dtype=tf.float32)

        train_target_feature = tf.constant(train_features[:, train_features.shape[1]-1])
        train_features = tf.constant(np.delete(train_features, train_features.shape[1]-1, 1), dtype=tf.float32)

        model = Sequential([
            Dense(1, activation='linear', input_shape=[train_features.shape[1]]),  # linear activation
        ])

        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.RMSprop(0.3),
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        model.fit(train_features, train_target_feature, epochs=50)
        weights = tf.transpose(model.get_weights()[0])
        bias = model.get_weights()[1].flatten()

        train_target_feature = train_target_feature.numpy()
        test_target_feature = test_target_feature.numpy()
        # print('y = x * (%f) + (%f)' % (weights[0][0], bias))
        # for idx, val in enumerate(features):
        #     print('x: %f y: %f class: %f pred_y: %f diff: %f' %(val, target_feature[idx], target[idx], val*weights[0][0]+bias, math.fabs(target_feature[idx]-val*weights[0][0]+bias)))

        train_predictions = model.predict(train_features).flatten()
        train_probs = np.square(np.subtract(train_target_feature, train_predictions))

        test_predictions = model.predict(test_features).flatten()
        test_probs = np.square(np.subtract(test_target_feature, test_predictions))

        return self.find_nearest(k, train_probs, test_probs, datasets, evaluation)



    def evaluate(self, features, target, anomaly_ratio, p):
        target_feature = tf.constant(features[:, features.shape[1]-1])
        features = tf.constant(np.delete(features, features.shape[1]-1, 1), dtype=tf.float32)
        model = Sequential([
            Dense(1, activation='linear', input_shape=[features.shape[1]]), #linear activation
        ])

        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.RMSprop(0.3),
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        model.fit(features, target_feature, epochs=50)
        weights = tf.transpose(model.get_weights()[0])
        bias = model.get_weights()[1].flatten()

        target_feature = target_feature.numpy()
        # print('y = x * (%f) + (%f)' % (weights[0][0], bias))
        # for idx, val in enumerate(features):
        #     print('x: %f y: %f class: %f pred_y: %f diff: %f' %(val, target_feature[idx], target[idx], val*weights[0][0]+bias, math.fabs(target_feature[idx]-val*weights[0][0]+bias)))

        test_predictions = model.predict(features).flatten()
        probs = np.square(np.subtract(target_feature,test_predictions))
        # probs = tf.math.divide(
        #     tf.math.abs(
        #         tf.math.subtract(tf.math.add(tf.reduce_sum(tf.multiply(weights, features), axis=1), bias), tf.constant(1, dtype=tf.float32))),#target
        #     tf.math.sqrt(tf.math.add(tf.multiply(weights, weights), tf.multiply(bias, bias)))).numpy().flatten() #tf.math.sqrt(tf.reduce_sum(tf.multiply(weights, weights), axis=1))

        print('Selecting threshold...')
        best_scores = self.select_threshold(probs, target, anomaly_ratio)

        return best_scores, probs, test_predictions


    def visualize_2d(self, dataset, features, target, probs, best_scores, test_predictions):
        performance = ['acc', 'prec', 'recall', 'f1', 'manual']
        fig = plt.figure(figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('%s, PCA = %d' % (dataset['name'], 2), fontsize=15)

        df = pd.concat(
            [pd.DataFrame(data=features, columns=['pca1', 'pca2']),
             pd.DataFrame(data=target, columns=['target'])],
            axis=1)

        for idx, val in enumerate(performance):
            outliers = np.array(np.where(probs >= best_scores[val]['epsilon'])).flatten()

            ax = fig.add_subplot(3, 3, idx+1)
            if val == 'manual':
                ax.set_title('%s (manual): %f%%, epsilon: %.2E' % (val, best_scores[val]['scores']['f1'], best_scores[val]['epsilon']), fontsize=12)
            else:
                ax.set_title('%s: %f%%, epsilon: %.2E' % (val, best_scores[val]['scores'][val], best_scores[val]['epsilon']), fontsize=12)

            for cls, color in zip([0, 1], ['g', 'r']):
                indicesToKeep = df['target'] == cls
                ax.scatter(df.loc[indicesToKeep, 'pca1']
                           , df.loc[indicesToKeep, 'pca2']
                           , c=color
                           , s=50)

            ax.scatter(df.loc[outliers, 'pca1']
                       , df.loc[outliers, 'pca2']
                       , c='w'
                       , s=10)

            test_features = np.delete(features, features.shape[1] - 1, 1)
            plt.plot(test_features, test_predictions, color='blue', linewidth=2)

        ax = fig.add_subplot(3, 3, 6)
        ax.set_title('Probabilities', fontsize=12)
        ax.scatter(df['pca1'], df['pca2'], c=probs, s=50)

        test_features = np.delete(features, features.shape[1] - 1, 1)
        plt.plot(test_features, test_predictions, color='blue', linewidth=1)

        fig.legend(['regression', 'normal', 'anomaly', 'detected'], facecolor="#B6B6B6")
        plt.show()
