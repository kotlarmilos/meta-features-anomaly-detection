import math
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

import utilities

class RPCA:
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
        for epsilon in np.nditer(epsilons):
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
        # outliers = np.argpartition(probs, math.ceil(len(target) * anomaly_ratio))[:math.ceil(len(target) * anomaly_ratio)]
        outliers = np.argsort(probs)[-math.ceil(len(target) * anomaly_ratio):]
        outliers = outliers[::-1]

        prediction = np.zeros(len(target))
        for x, y in zip(outliers, prediction):
            prediction[x] = 1

        acc = accuracy_score(target, prediction)
        prec = precision_score(target, prediction, labels=[0, 1])
        recall = recall_score(target, prediction, labels=[0, 1])
        f1 = f1_score(target, prediction, labels=[0, 1])

        best_scores['manual'] = {'epsilon': min(probs[outliers]), 'scores':{'acc':acc, 'prec':prec, 'recall':recall, 'f1':f1}}
        return best_scores

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def frobenius_norm_tf(M):
        return tf.norm(M, ord='fro', axis=(0,1))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    @staticmethod
    def shrink_tf(M, tau):
        return tf.multiply(tf.sign(M), tf.maximum(tf.subtract(tf.math.abs(M), tau), tf.zeros(M.shape, dtype=tf.float64)))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def svd_threshold_tf(self, M, tau):
        s, u, v = tf.linalg.svd(M)
        return tf.matmul(u, tf.matmul(tf.linalg.diag(self.shrink_tf(s, tau)), v, adjoint_b=True))
        #
        # U, S, V = tf.linalg.svd(M, full_matrices=False)
        # return tf.multiply(U, tf.multiply(tf.linalg.diag(self.shrink_tf(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        err_tf = tf.constant(np.Inf, dtype=tf.float64)
        Sk_tf = self.S_tf
        Yk_tf = self.Y_tf
        Lk_tf = tf.zeros(self.D.shape, dtype=tf.float64)

        _tol = 1E-7 * self.frobenius_norm(self.D)
        _tol_tf = tf.constant(1E-7, dtype=tf.float64) * self.frobenius_norm_tf(self.D_tf)

        while (err_tf.numpy() > _tol_tf.numpy()) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)

            Lk_tf = self.svd_threshold_tf(tf.add(tf.subtract(self.D_tf,Sk_tf),tf.multiply(self.mu_inv_tf,Yk_tf)), self.mu_inv_tf)
            # Lk_tf = tf.constant(Lk, dtype=tf.float64)


            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Sk_tf = self.shrink_tf(
                tf.add(tf.subtract(self.D_tf,Lk_tf), tf.math.multiply(self.mu_inv_tf, Yk_tf)), tf.math.multiply(self.mu_inv_tf, self.lmbda_tf))

            # Sk_tf = tf.constant(Sk, dtype=tf.float64)

            Yk = Yk + self.mu * (self.D - Lk - Sk)
            Yk_tf = tf.add(Yk_tf, tf.math.multiply(self.mu_tf, tf.subtract(self.D_tf, tf.add(Lk_tf, Sk_tf))))

            err = self.frobenius_norm(self.D - Lk - Sk)
            err_tf = self.frobenius_norm_tf(tf.subtract(self.D_tf, tf.add(Lk_tf, Sk_tf)))
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err_tf.numpy() <= _tol_tf.numpy():
                print('iteration: {0}, error: {1}'.format(iter, err_tf.numpy()))

        self.L = Lk
        self.S = Sk

        self.L_tf = Lk_tf
        self.S_tf = Sk_tf
        return Lk_tf.numpy(), Sk_tf.numpy()

    def evaluate(self, features, target, anomaly_ratio, p):
        self.D = features
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.mu = np.prod(self.D.shape) / (4 * self.frobenius_norm(self.D))
        self.mu_inv = 1 / self.mu
        self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

        self.D_tf = tf.constant(features, dtype=tf.float64)
        self.S_tf = tf.zeros(self.D_tf.shape, dtype=tf.float64)
        self.Y_tf = tf.zeros(self.D_tf.shape, dtype=tf.float64)
        self.mu_tf = tf.divide(tf.math.reduce_prod(tf.constant(self.D_tf.shape, dtype=tf.float64)),
                               tf.multiply(tf.constant(4, dtype=tf.float64),self.frobenius_norm_tf(self.D_tf)))

        self.mu_inv_tf = tf.divide(tf.constant(1, dtype=tf.float64), self.mu_tf)
        self.lmbda_tf = tf.divide(tf.constant(1, dtype=tf.float64),
                                  tf.math.sqrt(tf.reduce_max(tf.constant(self.D_tf.shape, dtype=tf.float64))))



        # features_normal = tf.constant(np.delete(features, np.where(target == 1),  axis=0))
        # features = tf.constant(features)
        # X = features

        # from r_pca import R_pca
        # use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
        # rpca = R_pca(X)
        L, S = self.fit(max_iter=10000, iter_print=100)
        # print(L, S)
        # from sklearn.decomposition import PCA

        probs = []
        for s in S:
            probs.append(np.sum(np.abs(s)))

        probs = np.array(probs)
        # pca = PCA(n_components=features.shape[1])
        # f = pca.fit_transform(features)

        # from sklearn.metrics import r2_score
        # from sklearn.metrics import mean_squared_error
        # from math import sqrt
        # import numpy as np
        #
        # r2 = r2_score(X, f)
        # rmse = sqrt(mean_squared_error(X, f))
        #
        # # RMSE normalised by mean:
        # nrmse = rmse / sqrt(np.mean(X ** 2))

        print('Selecting threshold...')
        best_scores = self.select_threshold(probs, target, anomaly_ratio)

        return best_scores, probs

    def find_nearest(self, array, value, datasets, evaluation):
        array = np.asarray(array)
        # idx = (np.abs(array - value)).argmin()
        idx = np.argsort(np.abs(array-value))[1]
        closest_dataset_id = datasets.iloc[idx]['id']
        closest_dataset_name = datasets.iloc[idx]['name']
        print('Closest dataset is %s...' % closest_dataset_name)
        es = evaluation[evaluation['dataset_id'] == closest_dataset_id]
        best_score = es['f1'].max()
        method = es[es['f1'] == best_score]['method'].to_numpy()[0]
        params = {'pca': es[es['f1'] == best_score]['pca'].to_numpy()[0], 'k': es[es['f1'] == best_score]['k'].to_numpy()[0]}
        return best_score, method, params, closest_dataset_name

    def crossval(self, idx, datasets, evaluation):
        closest_dataset_id = datasets.iloc[idx]['id']
        es = evaluation[evaluation['dataset_id'] == closest_dataset_id]
        best_score = es['f1'].max()
        method = es[es['f1'] == best_score]['method'].to_numpy()[0]
        params = {'pca': es[es['f1'] == best_score]['pca'].to_numpy()[0], 'k': es[es['f1'] == best_score]['k'].to_numpy()[0]}
        return best_score, method, params

    def predict(self, idx, datasets, evaluation, best_method, best_params):
        # closest_dataset_id = datasets.iloc[idx]['id']
        # es = evaluation[(evaluation['dataset_id'] == closest_dataset_id)
        # & (evaluation['method'] == best_method)
        # & (evaluation['pca'] == best_params['pca'])
        # & (evaluation['k'] == best_params['k'])]
        #
        # score = es['f1'].to_numpy()[0]
        # return score, best_method, best_params
        closest_dataset_id = datasets.iloc[idx]['id']
        es = evaluation[(evaluation['dataset_id'] == closest_dataset_id)
                        & (evaluation['method'] == best_method)]
        # & (evaluation['pca'] == best_params['pca'])
        # & (evaluation['k'] == best_params['k'])

        score = es['f1'].max()  # .to_numpy()[0]
        return score, best_method, best_params

    def distance(self, test_features, train_features, datasets, evaluation):
        # features = np.concatenate((test_features, train_features))
        index = np.where(train_features == test_features[0])[0][0]
        features = train_features
        self.D = features
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.mu = np.prod(self.D.shape) / (4 * self.frobenius_norm(self.D))
        self.mu_inv = 1 / self.mu
        self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

        self.D_tf = tf.constant(features, dtype=tf.float64)
        self.S_tf = tf.zeros(self.D_tf.shape, dtype=tf.float64)
        self.Y_tf = tf.zeros(self.D_tf.shape, dtype=tf.float64)
        self.mu_tf = tf.divide(tf.math.reduce_prod(tf.constant(self.D_tf.shape, dtype=tf.float64)),
                               tf.multiply(tf.constant(4, dtype=tf.float64), self.frobenius_norm_tf(self.D_tf)))

        self.mu_inv_tf = tf.divide(tf.constant(1, dtype=tf.float64), self.mu_tf)
        self.lmbda_tf = tf.divide(tf.constant(1, dtype=tf.float64),
                                  tf.math.sqrt(tf.reduce_max(tf.constant(self.D_tf.shape, dtype=tf.float64))))

        # features_normal = tf.constant(np.delete(features, np.where(target == 1),  axis=0))
        # features = tf.constant(features)
        # X = features

        # from r_pca import R_pca
        # use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
        # rpca = R_pca(X)
        L, S = self.fit(max_iter=10000, iter_print=100)
        # print(L, S)
        # from sklearn.decomposition import PCA

        probs = []
        for s in S:
            probs.append(np.sum(np.abs(s)))

        train_probs = np.array(probs)
        test_probs =  train_probs[index]

        return self.find_nearest(train_probs, test_probs, datasets, evaluation)


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
            outliers = np.array(np.where(probs > best_scores[val]['epsilon'])).flatten()

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


        ax = fig.add_subplot(3, 3, 6)
        ax.set_title('Probabilities', fontsize=12)
        ax.scatter(df['pca1'], df['pca2'], c=probs, s=50)
        fig.legend(['normal', 'anomaly', 'detected'], facecolor="#B6B6B6")

        plt.show()
