import math
import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import utilities


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed


class AutoencoderModel:
    def __init__(self):
        pass

    def get_params(self, features):
        pca = list(reversed(range(2, features.shape[1] + 1)))
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
        # epsilons = epsilons[::-1]
        for epsilon in np.nditer(epsilons):#, order='C'
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

    def loss(self, model, original):
        reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
        return reconstruction_error

    def train(self, loss, model, opt, original):
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss(model, original), model.trainable_variables)
            gradient_variables = zip(gradients, model.trainable_variables)
            opt.apply_gradients(gradient_variables)

    def evaluate(self, features, target, anomaly_ratio, p):
        features = features.astype('float32')
        features_normal = tf.constant(np.delete(features, np.where(target == 1),  axis=0))

        # MNIST input 28 rows * 28 columns = 784 pixels
        input_img = Input(shape=(features_normal.shape[1],))
        # encoder
        encoder1 = Dense(128, activation='relu')(input_img)
        encoder2 = Dense(32, activation='sigmoid')(encoder1)
        # decoder
        decoder1 = Dense(128, activation='relu')(encoder2)
        decoder2 = Dense(features_normal.shape[1], activation='sigmoid')(decoder1)

        # this model maps an input to its reconstruction
        autoencoder = Model(inputs=input_img, outputs=decoder2)

        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(features_normal, features_normal,
                        epochs=50,
                        batch_size=32,
                        shuffle=True)

        # create encoder model
        encoder = Model(inputs=input_img, outputs=encoder2)
        # create decoder model
        encoded_input = Input(shape=(32,))
        decoder_layer1 = autoencoder.layers[-2]
        decoder_layer2 = autoencoder.layers[-1]
        decoder = Model(inputs=encoded_input, outputs=decoder_layer2(decoder_layer1(encoded_input)))

        latent_vector = encoder.predict(features)
        # get decoder output to visualize reconstructed image
        reconstructed_imgs = decoder.predict(latent_vector)

        probs = (tf.keras.losses.MSE(features, reconstructed_imgs)).numpy()
        # probs2 = np.square(features - reconstructed_imgs)
        # autoencoder = Autoencoder(intermediate_dim=2, original_dim=features_normal.shape[1])
        # opt = tf.optimizers.Adam(learning_rate=0.2)
        #
        # training_dataset = tf.data.Dataset.from_tensor_slices(features)
        # training_dataset = training_dataset.batch(100)
        # training_dataset = training_dataset.shuffle(features.shape[0])
        # training_dataset = training_dataset.prefetch(100 * 4)
        #
        # for epoch in range(100):
        #     for step, batch_features in enumerate(training_dataset):
        #         self.train(self.loss, autoencoder, opt, batch_features)
        #         loss_values = self.loss(autoencoder, batch_features)
        #
        # result = autoencoder(tf.constant(features))

        best_scores = self.select_threshold(probs, target, anomaly_ratio)

        return best_scores, probs


    def visualize_2d(self, dataset, features, target, probs, best_scores, pca):
        performance = ['acc', 'prec', 'recall', 'f1', 'manual']
        fig = plt.figure(figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle('%s, PCA = %d' % (dataset['name'], pca), fontsize=15)

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





# (training_features, _), (test_features, _) = tf.keras.datasets.mnist.load_data()
# training_features = training_features / np.max(training_features)
# training_features = training_features.reshape(training_features.shape[0],
#                                               training_features.shape[1] * training_features.shape[2])
# training_features = training_features.astype('float32')
# training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
# training_dataset = training_dataset.batch(100)
# training_dataset = training_dataset.shuffle(training_features.shape[0])
# training_dataset = training_dataset.prefetch(100 * 4)
#
# writer = tf.summary.create_file_writer('tmp')
#
# with writer.as_default():
#     with tf.summary.record_if(True):
#         for epoch in range(1):
#             for step, batch_features in enumerate(training_dataset):
#                 train(loss, autoencoder, opt, batch_features)
#                 loss_values = loss(autoencoder, batch_features)
#                 original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
#                 reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
#                                            (batch_features.shape[0], 28, 28, 1))
#                 tf.summary.scalar('loss', loss_values, step=step)
#                 tf.summary.image('original', original, max_outputs=10, step=step)
#                 tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
#
# result = autoencoder.predict(training_features)
# print(result)