# import tensorflow_probability as tfp
import tensorflow as tf
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
#
# @tf.function
# def nn(features, target):
#     features = tf.constant(features)
#     target = tf.constant(target)
#     # tf.profiler.experimental.start('logdir')
#     # create a NN
#     model = Sequential()
#     model.add(Dense(features.shape[1], input_dim=features.shape[1], activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#
#     # build a model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
#         tf.keras.metrics.MeanSquaredError(),
#         tf.keras.metrics.AUC(),
#         tf.keras.metrics.Precision(),
#         tf.keras.metrics.Recall(),
#         tf.keras.metrics.Accuracy()
#     ])
#
#     # measure training time
#     start_time = tf.timestamp()
#     model.fit(features, target)
#     train_time = tf.timestamp() - start_time
#
#     # measure test time
#     start_time = tf.timestamp()
#     results = model.predict(features)
#     test_time = (tf.timestamp() - start_time) / features.shape[0]
#
#     # cross validation
#     validation = model.evaluate(features, target)
#
#     # tf.profiler.experimental.stop()
#
#     return train_time.numpy(), test_time.numpy(), validation
#
#
def gaussian(features):
    # measure training time
    start_time = tf.timestamp()
    features = tf.constant(features)
    mu = tf.reduce_mean(features, axis=0)
    mu = tf.reshape(mu, [1, features.shape[1]])
    mx = tf.matmul(tf.transpose(mu), mu)
    vx = tf.matmul(tf.transpose(features), features) / tf.cast(tf.shape(features)[0], tf.float64)
    sigma = vx - mx
    #     mvn = tfp.distributions.MultivariateNormalTriL(loc=mu,scale_tril=tf.linalg.cholesky(sigma))
    train_time = tf.timestamp() - start_time
    # measure test time
    start_time = tf.timestamp()
    #     mvn.prob(tf.constant(features))
    test_time = (tf.timestamp() - start_time) / features.shape[0]

    return train_time.numpy(), test_time.numpy()
#
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
#
# def linear(features):
#     # create linear regression
#     model = Sequential()
#     model.add(Dense(1, input_dim=features.shape[1], activation='linear'))
#
#     model.compile(loss='mean_squared_error',
#                   optimizer=tf.keras.optimizers.RMSprop(0.001),
#                   metrics=['mean_absolute_error', 'mean_squared_error'])
#
#     # measure training time
#     start_time = tf.timestamp()
#     model.fit(features, features)
#     train_time = tf.timestamp() - start_time
#
#     weights = tf.transpose(model.get_weights()[0])
#
#     # measure test time
#     start_time = tf.timestamp()
#     distance = tf.math.divide(
#         tf.math.abs(tf.math.add(tf.reduce_sum(tf.multiply(weights, features), axis=1), model.get_weights()[1])),
#         tf.math.sqrt(tf.reduce_sum(tf.multiply(weights, weights), axis=1)))
#     test_time = (tf.timestamp() - start_time) / features.shape[0]
#
#     return train_time.numpy(), test_time.numpy()
#
#
# def pca(features):
#     features -= tf.reduce_mean(features, axis=0)
#
#     start_time = tf.timestamp()
#     result = tf.linalg.svd(features)
#     train_time = test_time = (tf.timestamp() - start_time)
#
#     return train_time.numpy(), test_time.numpy()
#
#
# # function which returns data (necessary for tensorflow v1)
# def input_fn():
#     return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(features, dtype=tf.float32), num_epochs=1)
#
#
# def kmeans(features):
#     # build a model
#     model = tf.compat.v1.estimator.experimental.KMeans(5)
#
#     # train a model
#     start_time = tf.timestamp()
#     model.train(input_fn)
#     train_time = (tf.timestamp() - start_time)
#
#     # test a model
#     start_time = tf.timestamp()
#     model.predict_cluster_index(input_fn)
#     test_time = (tf.timestamp() - start_time)
#
#     return train_time.numpy(), test_time.numpy()



