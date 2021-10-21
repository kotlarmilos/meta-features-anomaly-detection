import math
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt

import tensorflow as tf
from random import randint

import utilities

arr = []


class Eucledian:
    def __init__(self):
        pass

    def get_params(self, features):
        pca = list(reversed(range(2, features.shape[1] + 1)))
        k = list(range(2, 50))
        return utilities.make_cartesian([pca, k]), ['pca', 'k']

    def estimate_gaussian(self, features):
        mu = tf.reduce_mean(features, axis=0)
        mu = tf.reshape(mu, [1, features.shape[1]])
        mx = tf.matmul(tf.transpose(mu), mu)
        vx = tf.matmul(tf.transpose(features), features) / tf.cast(tf.shape(features)[0], tf.float64)
        sigma = vx - mx
        return mu, sigma

    def select_threshold(self, probs, target, anomaly_ratio):
        best_scores = {}
        best_scores['acc'] = {'epsilon': 0, 'scores': {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0}}
        best_scores['prec'] = {'epsilon': 0, 'scores': {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0}}
        best_scores['recall'] = {'epsilon': 0, 'scores': {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0}}
        best_scores['f1'] = {'epsilon': 0, 'scores': {'acc': 0, 'prec': 0, 'recall': 0, 'f1': 0}}

        # find best metrics and epsilons using test data
        stepsize = (max(probs) - min(probs)) / 100
        epsilons = np.arange(min(probs), max(probs), stepsize)
        epsilons = epsilons[::-1]
        for epsilon in np.nditer(epsilons, order='C'):
            prediction = (probs > epsilon)
            if len(set(prediction)) == 1:
                continue
            acc = accuracy_score(target, prediction)
            prec = precision_score(target, prediction, labels=[0, 1])
            recall = recall_score(target, prediction, labels=[0, 1])
            f1 = f1_score(target, prediction, labels=[0, 1])
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

        best_scores['manual'] = {'epsilon': min(probs[outliers]),
                                 'scores': {'acc': acc, 'prec': prec, 'recall': recall, 'f1': f1}}
        return best_scores

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

    def distance(self, k, test_features, train_features, datasets, evaluation):
        test_features = tf.constant(test_features)
        train_features = tf.constant(train_features)


        l2_norm = tf.norm(tf.subtract(test_features, train_features) , axis = 1, ord='euclidean')

        # distances = tf.reduce_sum(tf.abs(tf.subtract(test_features, train_features)), 1)
        train_probs = l2_norm.numpy()
        test_probs = 0

        return self.find_nearest(k, train_probs, test_probs, datasets, evaluation)

    def input_fn(self):
        return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(self.features, dtype=tf.float32), num_epochs=1)

    def evaluate(self, features, target, anomaly_ratio, p):
        # self.features = tf.constant(features)
        k = p[1]
        self.features = features
        kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=int(k), use_mini_batch=False)

        # train
        num_iterations = 10
        previous_centers = None
        cluster_centers = None
        for _ in range(num_iterations):
            kmeans.train(self.input_fn)
        #     cluster_centers = kmeans.cluster_centers()
        #     if previous_centers is not None:
        #         print ('delta:', cluster_centers - previous_centers)
        #     previous_centers = cluster_centers
        #     print('score:', kmeans.score(self.input_fn))
        # print ('cluster centers:', cluster_centers)

        # map the input points to their clusters
        probs = []
        clusters = []
        for item in list(kmeans.predict(self.input_fn)):
            probs.append(item['all_distances'][item['cluster_index']])
            clusters.append(item['cluster_index'])

        probs = np.array(probs)
        clusters = np.array(clusters)
        # for i, point in enumerate(features):
        #     cluster_index = cluster_indices[i]
        #     center = cluster_centers[cluster_index]
        #     print('point:', point, 'is in cluster', cluster_index, 'centered at', center)
        # model = tf.compat.v1.estimator.experimental.KMeans(k)
        # # train a model
        # model.train(self.input_fn)
        #
        # # test a model
        # probs = model.predict_cluster_index(self.input_fn)
        #
        # print('Selecting threshold...')
        best_scores = self.select_threshold(probs, target, anomaly_ratio)
        #
        return best_scores, probs, clusters

    def visualize_2d(self, dataset, features, target, probs, best_scores, clusters, n):
        performance = ['f1']
        fig = plt.figure(figsize=(8, 6))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('%s, KMeans PCA = %d, K = %d' % (dataset['name'], n, len(np.unique(clusters))), fontsize=15)

        df = pd.concat(
            [pd.DataFrame(data=features, columns=['pca1', 'pca2']),
             pd.DataFrame(data=target, columns=['target']),
             pd.DataFrame(data=clusters, columns=['cluster'])],
            axis=1)

        colors = []
        for i in range(len(np.unique(clusters))):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        for idx, val in enumerate(performance):
            outliers = np.array(np.where(probs >= best_scores[val]['epsilon'])).flatten()

            ax = fig.add_subplot(1, 1, 1)
            if val == 'manual':
                ax.set_title('%s (manual): %f%%, epsilon: %.2E' % (
                val, best_scores[val]['scores']['f1'], best_scores[val]['epsilon']), fontsize=12)
            else:
                ax.set_title(
                    '%s: %f%%, epsilon: %.2E' % (val, best_scores[val]['scores'][val], best_scores[val]['epsilon']),
                    fontsize=12)

            # for cls, color in zip(range(0, len(np.unique(clusters))), colors):
            #     indicesToKeep = df['cluster'] == cls
            #     ax.scatter(df.loc[indicesToKeep, 'pca1']
            #                , df.loc[indicesToKeep, 'pca2']
            #                , c=color
            #                , s=50)

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

        # ax = fig.add_subplot(3, 3, 6)
        # ax.set_title('Probabilities', fontsize=12)
        # ax.scatter(df['pca1'], df['pca2'], c=probs, s=50)
        fig.legend(['normal', 'anomaly', 'detected'], facecolor="#B6B6B6")

        plt.show()

    # def visualize_2d(self, dataset, features, target, probs, best_scores, clusters, n):
    #     performance = ['acc', 'prec', 'recall', 'f1', 'manual']
    #     fig = plt.figure(figsize=(12, 12))
    #     plt.subplots_adjust(hspace=0.5)
    #     fig.suptitle('%s, PCA = %d, K = %d' % (dataset['name'], n, len(np.unique(clusters))), fontsize=15)
    #
    #     df = pd.concat(
    #         [pd.DataFrame(data=features, columns=['pca1', 'pca2']),
    #          pd.DataFrame(data=target, columns=['target']),
    #          pd.DataFrame(data=clusters, columns=['cluster'])],
    #         axis=1)
    #
    #     colors = []
    #     for i in range(len(np.unique(clusters))):
    #         colors.append('#%06X' % randint(0, 0xFFFFFF))
    #
    #     for idx, val in enumerate(performance):
    #         outliers = np.array(np.where(probs >= best_scores[val]['epsilon'])).flatten()
    #
    #         ax = fig.add_subplot(3, 3, idx + 1)
    #         if val == 'manual':
    #             ax.set_title('%s (manual): %f%%, epsilon: %.2E' % (
    #             val, best_scores[val]['scores']['f1'], best_scores[val]['epsilon']), fontsize=12)
    #         else:
    #             ax.set_title(
    #                 '%s: %f%%, epsilon: %.2E' % (val, best_scores[val]['scores'][val], best_scores[val]['epsilon']),
    #                 fontsize=12)
    #
    #         for cls, color in zip(range(0, len(np.unique(clusters))), colors):
    #             indicesToKeep = df['cluster'] == cls
    #             ax.scatter(df.loc[indicesToKeep, 'pca1']
    #                        , df.loc[indicesToKeep, 'pca2']
    #                        , c=color
    #                        , s=50)
    #
    #         # for cls, color in zip([0, 1], ['g', 'r']):
    #         #     indicesToKeep = df['target'] == cls
    #         #     ax.scatter(df.loc[indicesToKeep, 'pca1']
    #         #                , df.loc[indicesToKeep, 'pca2']
    #         #                , c=color
    #         #                , s=50)
    #         #
    #         # ax.scatter(df.loc[outliers, 'pca1']
    #         #            , df.loc[outliers, 'pca2']
    #         #            , c='w'
    #         #            , s=10)
    #
    #     ax = fig.add_subplot(3, 3, 6)
    #     ax.set_title('Probabilities', fontsize=12)
    #     ax.scatter(df['pca1'], df['pca2'], c=probs, s=50)
    #     fig.legend(['normal', 'anomaly', 'detected'], facecolor="#B6B6B6")
    #
    #     plt.show()