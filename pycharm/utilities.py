import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import json
from sklearn.utils.extmath import cartesian

strategy = None
if (os.environ.get('COLAB_TPU_ADDR') != None):
    os.environ['TPU_ADDR'] = os.environ['COLAB_TPU_ADDR']

if (os.environ.get('TPU_ADDR') != None):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

def get_datasets(path):
    datasets = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            with open(os.path.join(root, dir, 'metadata.json')) as json_file:
                metadata = json.load(json_file)
                metadata['files'] = [os.path.join(root, dir, filename) for filename in metadata['files']]
                datasets.append(metadata)
    return datasets

def get_devices():
    if strategy:
        devices = ['ASIC']
    else:
        devices = [d.name for d in device_lib.list_local_devices() if 'XLA' not in d.name]
    return devices

def get_methods():
    methods = [
        {
            'name': 'Gaussian',
            'isSupervised': False,
        },
        {
            'name': 'Eucledian',
            'isSupervised': False,
        },
        {
            'name': 'Manhattan',
            'isSupervised': False,
        },
        {
            'name': 'Linear',
            'isSupervised': False,
        },
        {
            'name': 'RPCA',
            'isSupervised': False,
        },
        {
            'name': 'KMeans',
            'isSupervised': False,
        },
        {
            'name': 'AutoencoderModel',
            'isSupervised': True,
        }
    ];

    return methods

def make_cartesian(x):
    return cartesian(x)

# def evaluate(device, methods, dataset, features, target):
#     isSupervised = target is not None
#     for method in methods:
#         if method['isSupervised'] == isSupervised:
#             # if method['name'] == 'gaussian':
#             #     result = gaussian(features)
#             # elif method['name'] == 'linear_regression':
#             #     result = linear(features)
#             # elif method['name'] == 'pca':
#             #     result = pca(features)
#             # elif method['name'] == 'kmeans':
#             #     result = kmeans(features)
#             # elif method['name'] == 'neural_network':
#             #     result = nn(features, target)
#
#             # print('Running', method['name'])
#             # print(device, result)
#             #
#             # insert_evaluation_info(device, method, dataset, result)
