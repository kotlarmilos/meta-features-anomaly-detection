import math
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp


class Gaussian:
    def __init__(self):
        pass

    def estimate_gaussian(self, features):
        mu = tf.reduce_mean(features, axis=0)
        mu = tf.reshape(mu, [1, features.shape[1]])
        mx = tf.matmul(tf.transpose(mu), mu)
        vx = tf.matmul(tf.transpose(features), features) / tf.cast(tf.shape(features)[0], tf.float64)
        sigma = vx - mx
        return mu, sigma

    def select_threshold(self, probs, target, anomaly_ratio):
        best_scores = {}
        best_scores['acc'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}
        best_scores['prec'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}
        best_scores['recall'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}
        best_scores['f1'] = {'epsilon': 0, 'scores':{'acc':0, 'prec':0, 'recall':0, 'f1':0}}

        # find best metrics and epsilons using test data
        stepsize = (max(probs) - min(probs)) / 1000
        epsilons = np.arange(min(probs), max(probs), stepsize)
        for epsilon in np.nditer(epsilons):
            prediction = (probs < epsilon)
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
        outliers = np.argpartition(probs, math.ceil(len(target) * anomaly_ratio))[:math.ceil(len(target) * anomaly_ratio)]
        prediction = np.zeros(len(target))
        for x, y in zip(outliers, prediction):
            prediction[x] = 1

        acc = accuracy_score(target, prediction)
        prec = precision_score(target, prediction, labels=[0, 1])
        recall = recall_score(target, prediction, labels=[0, 1])
        f1 = f1_score(target, prediction, labels=[0, 1])

        best_scores['manual'] = {'epsilon': max(probs[outliers]), 'scores':{'acc':acc, 'prec':prec, 'recall':recall, 'f1':f1}}
        return best_scores


    def evaluate(self, features, target, anomaly_ratio):
        features_normal = tf.constant(np.delete(features, np.where(target == 1),  axis=0))
        features = tf.constant(features)
        mu, sigma = self.estimate_gaussian(features_normal)

        mvn = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tf.linalg.cholesky(sigma))
        probs = mvn.prob(features).numpy()

        print('Selecting threshold...')
        best_scores = self.select_threshold(probs, target, anomaly_ratio)

        return best_scores, probs


    def visualize_2d(self, dataset, features, target, probs, best_scores):
        performance = ['acc', 'prec', 'recall', 'f1', 'manual']
        fig = plt.figure(figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('%s, PCA = %d' % (dataset['name'], 2), fontsize=15)

        df = pd.concat(
            [pd.DataFrame(data=features, columns=['pca1', 'pca2']),
             pd.DataFrame(data=target, columns=['target'])],
            axis=1)

        for idx, val in enumerate(performance):
            outliers = np.array(np.where(probs < best_scores[val]['epsilon'])).flatten()

            ax = fig.add_subplot(3, 3, idx+1)
            if val == 'manual':
                val = 'f1'
                ax.set_title('%s (manual): %f%%, epsilon: %.2E' % (val, best_scores[val]['scores'][val], best_scores[val]['epsilon']), fontsize=12)
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


        ax = fig.add_subplot(3, 3, 6)
        ax.set_title('Probabilities', fontsize=12)
        ax.scatter(df['pca1'], df['pca2'], c=probs, s=50)
        fig.legend(['normal', 'anomaly', 'detected'], facecolor="#B6B6B6")

        plt.show()
