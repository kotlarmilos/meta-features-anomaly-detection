from sklearn.preprocessing import StandardScaler

from database import Database
from preprocessing import *
import utilities
import numpy as np
import os
import time
from itertools import combinations
from algorithms.kmeans import KMeans
from algorithms.linear import Linear
from algorithms.rpca import RPCA
from algorithms.autoencoder import AutoencoderModel
from algorithms.gaussian import Gaussian

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

db = Database("127.0.0.1","root","","anomaly_detection_decision_support")
# db = Database("anomaly-detection-mysql.ch1ih3mzagsi.eu-central-1.rds.amazonaws.com","admin","6xy4AMtnhkFJWfIWHsuu","anomaly_detection_decision_support")
global_datasets, evaluation, global_features = db.get_datasets()

datasets = global_datasets
summary_results = []
total = len(datasets)
ammortization = 0

features = []

# characterizationModel = AutoencoderModel()
# for i in range(len(datasets)):
#     if datasets.iloc[i]['id'] not in [236, 237]:
#         temp_path = datasets.iloc[i]['files'].replace('\'', '')
#         datasets.at[i, 'files'] = [temp_path]
#         temp_features, target, anomaly_ratio = load_data(datasets.iloc[i])
#
#         temp_features = standardize_data(temp_features)
#
#         latentVector = characterizationModel.getEncodedData(temp_features, 32)
#         latentVector = np.transpose(latentVector)
#         latentVector = characterizationModel.getEncodedData(latentVector, 1)
#         features.append(np.hstack(latentVector))
#
# features = np.array(features)
# np.savetxt('autoencoders_characterization.txt', features, fmt='%f')
features = np.loadtxt('autoencoders_characterization.txt', dtype=float)


print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for testing:', len(datasets))


best_method_name_for = ''
best_method_score_for = 0
methods = utilities.get_methods()
for eval_method in methods:
    try:
        if eval_method['name'] in ['Gaussian']:#AutoencoderModel#Gaussian#Linear##KMeans#'Linear''Gaussian'
            continue
        m = eval(eval_method['name'] + '()')

        ks = np.zeros((10,), dtype=float)
        method_match = np.zeros((10,), dtype=float)

        results = pd.DataFrame(columns=['Dataset','Distance','Best method', 'F1 with the best method','K', 'F1 method', 'F1 with the proposed method', 'Accuracy'])

        for i in range(len(datasets)):
            if datasets.iloc[i]['id'] in [236, 237]:
                total -= 1
                ammortization += 1
                continue
            test_features = features[i-ammortization]
            test_features.shape = (1,len(features[0]))
            train_features = features

            try:
                result, array = m.distance(10, test_features, train_features, datasets, evaluation)
                actual_score, actual_method, actual_params = m.crossval(i, datasets, evaluation)

                results = results.append(pd.DataFrame(
                        np.array([[datasets.iloc[i]['name'], 0, actual_method[0],actual_score, 0,actual_method[0],actual_score, 1]]),
                        columns=['Dataset','Distance','Best method', 'F1 with the best method','K', 'F1 method', 'F1 with the proposed method', 'Accuracy']))

                j = 0
                for r in result:
                    j+=1
                    mm, c = np.unique(array[:j], return_counts=True)
                    idx = np.argmax(c)
                    k_method = mm[idx]
                    best_score, method, params = m.predict(i, datasets, evaluation, k_method, None)
                    if not np.isnan(best_score):
                        ks[j-1]+=1-(actual_score-best_score)
                    if k_method in actual_method:
                        method_match[j-1]+=1

                    results = results.append(pd.DataFrame(
                        np.array([[r['dataset_name'],r['distance'],r['method'],r['best_score'],j,k_method, best_score, 1-(actual_score-best_score)]]),
                        columns=['Dataset','Distance','Best method', 'F1 with the best method','K', 'F1 method', 'F1 with the proposed method', 'Accuracy']))

                results = results.append(pd.Series(), ignore_index=True)

            except:
                print('Error')
        ks /= float(total)
        method_match /= float(total)

        if best_method_score_for<method_match[0]:
            best_method_score_for = method_match[0]
            best_method_name_for = eval_method['name']

        results = results.append(pd.DataFrame(
                np.array([['K', 'Method match', 'Accuracy']]),
                columns=[ 'K', 'F1 with the proposed method', 'Accuracy']))

        for index, item in enumerate(ks):
            results = results.append(pd.DataFrame(
                np.array([[int(index+1), method_match[index], item]]),
                columns=['K', 'F1 with the proposed method', 'Accuracy']))

        results.to_csv('results/%s.csv' % eval_method['name'])
        print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    except:
        print('Except error')
summary_results.append((best_method_name_for, best_method_score_for, total))
print(summary_results)