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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

db = Database("127.0.0.1","root","","anomaly_detection_decision_support")
# db = Database("35.234.97.110","root","6g8HBIy0F8atEKtb","anomaly_detection_decision_support")
datasets, evaluation, features = db.get_datasets()

# for dataset in utilities.get_datasets('/Users/miloskotlar/GoogleDrive/Academic/PhD/III/datasets/'):
#     db.update_characterization_user_defined_data(dataset)

characterization_columns = np.array(features.columns)
features = features.to_numpy()
features = standardize_data(features)

### change per purpose
# features = dimension_reduction(features, 2)



# temporal = np.array(datasets.loc[datasets['type_of_data'] == '\'temporal\''].axes[0])
# temporal_features = features[temporal]
# spatial = np.array(datasets.loc[datasets['type_of_data'].str.contains('\'spatial\'')].axes[0])
# spatial_features = features[spatial]
high = np.array(datasets.loc[datasets['type_of_data'] == '\'high-dimensional\''].axes[0])
high_features = features[high]


features = high_features

print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print('*** Datasets for testing:', len(features))

methods = utilities.get_methods()
# results = pd.DataFrame(columns=['Dataset_id','Dataset name','Best method','Score','Closest dataset','Closest best method','Closest best score','Proposed best method','Proposed best score','Method match','Precision'])
# results = pd.DataFrame(columns=['K','combs', 'Precision'])
# results = pd.DataFrame(columns=['c1','c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'])
results = pd.DataFrame(columns=['Dataset','Distance','Best method', 'F1 with the best method','K', 'F1 method', 'F1 with the proposed method', 'Accuracy'])
global_features = features

characterization_columns = ['anomaly_space', 'attr_ent.mean', 'attr_ent.sd', 'can_cor.sd', 'cat_to_num'
        , 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd', 'finance'
        , 'freq_class.mean', 'g_mean.mean', 'g_mean.sd', 'graphs_and_networks'
        , 'gravity', 'h_mean.mean', 'h_mean.sd', 'high_dimensional', 'images'
        , 'iq_range.mean', 'iq_range.sd', 'joint_ent.mean', 'joint_ent.sd'
        , 'kurtosis.sd', 'mad.mean', 'mad.sd', 'manufacturing', 'max.mean', 'max.sd'
        , 'mean.mean', 'mean.sd', 'median.mean', 'median.sd', 'medicine', 'min.mean'
        , 'min.sd', 'nr_cat', 'nr_class', 'nr_disc', 'nr_inst', 'num_to_cat'
        , 'range.mean', 'range.sd', 'row_count', 'sd.mean', 'sd.sd', 'skewness.mean'
        , 'skewness.sd', 'social', 'spatial', 't_mean.mean', 't_mean.sd', 'temporal'
        , 'transport', 'var.mean', 'var.sd']

# for comb_k in range(3, 4):
combs = combinations(characterization_columns, 3)
for comb in combs:
    features = global_features
    comb = np.array(comb)
# var = ['anomaly_space', 'attr_ent.mean', 'attr_ent.sd', 'can_cor.sd', 'cat_to_num'
#         , 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd', 'finance'
#         , 'freq_class.mean', 'g_mean.mean', 'g_mean.sd', 'graphs_and_networks'
#         , 'gravity', 'h_mean.mean', 'h_mean.sd', 'high_dimensional', 'images'
#         , 'iq_range.mean', 'iq_range.sd', 'joint_ent.mean', 'joint_ent.sd'
#         , 'kurtosis.sd', 'mad.mean', 'mad.sd', 'manufacturing', 'max.mean', 'max.sd'
#         , 'mean.mean', 'mean.sd', 'median.mean', 'median.sd', 'medicine', 'min.mean'
#         , 'min.sd', 'nr_cat', 'nr_class', 'nr_disc', 'nr_inst', 'num_to_cat'
#         , 'range.mean', 'range.sd', 'row_count', 'sd.mean', 'sd.sd', 'skewness.mean'
#         , 'skewness.sd', 'social', 'spatial', 't_mean.mean', 't_mean.sd', 'temporal'
#         , 'transport', 'var.mean', 'var.sd']
# temp = np.unique(temp)
# print(temp
# comb = temp
# comb = ['anomaly_space', 'kurtosis.sd']
#comb_k = 17#3
    idx = np.in1d(characterization_columns, comb).nonzero()[0]
    features = features[:, idx]

    # features = dimension_reduction(features, 2)

    ks = np.zeros((9,), dtype=float)
    method_match = np.zeros((9,), dtype=float)

    # fig = plt.figure(figsize=(12, 12))
    # plt.subplots_adjust(hspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(features[:,0], features[:,1], c='r', s=50, label='high-dimensional')
    #
    #
    # for i in range(len(features)):
    #     text = datasets.loc[datasets['type_of_data'] == '\'high-dimensional\''].iloc[i]['name']
    #     ax.annotate(text, (features[i][0], features[i][1]))
    # # # ax.scatter(temporal_features[:,0], temporal_features[:,1], c='b', s=50, label='temporal')
    # # # ax.scatter(spatial_features[:,0], spatial_features[:,1], c='g', s=50, label='spatial')
    # plt.legend()
    # plt.plot()

    for i, dataset in datasets.iterrows():
        if i>=len(features):
            break

        # if i<10 or i>62:
        #     continue
        # test_features = features[i-10]

        # if i>62:
        #     continue
        test_features = features[i]
        test_features.shape = (1,len(features[0])) #2
        train_features = features


        for method in methods:
            if method['name'] in ['KMeans']:#AutoencoderModel#Gaussian#Linear#RPCA
                m = eval(method['name']+'()')
                # try:
                result, array = m.distance(10 ,test_features, train_features, datasets, evaluation)
                actual_score, actual_method, actual_params = m.crossval(i, datasets, evaluation)

                results = results.append(pd.DataFrame(
                        np.array([[dataset['name'], 0, actual_method,actual_score, 0,actual_method,actual_score, 1]]),
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
                    if k_method == actual_method:
                        method_match[j-1]+=1

                    results = results.append(pd.DataFrame(
                        np.array([[r['dataset_name'],r['distance'],r['method'],r['best_score'],j,k_method, best_score, 1-(actual_score-best_score)]]),
                        columns=['Dataset','Distance','Best method', 'F1 with the best method','K', 'F1 method', 'F1 with the proposed method', 'Accuracy']))

                results = results.append(pd.Series(), ignore_index=True)

                # if best_method == actual_method:
                #     sum += 1
            # expected_score, expected_method, expected_params = m.predict(i, datasets, evaluation, best_method, best_params)

            # except:
            #     print('Error')

# results = results.append(df)
# if sum/total >0.5:
# print('For combs %s iteration %s K=%s precision is %s' % (comb, counter, k,sum/total))

    ks /= 10.
    method_match /= 10.

    if (method_match>0.7).any():
        print(comb, method_match)
    # results = results.append(pd.DataFrame(
    #         np.array([['K', 'Method match', 'Accuracy']]),
    #         columns=[ 'K', 'F1 with the proposed method', 'Accuracy']))
    #
    # for index, item in enumerate(ks):
    #     results = results.append(pd.DataFrame(
    #         np.array([[int(index+1), method_match[index], item]]),
    #         columns=['K', 'F1 with the proposed method', 'Accuracy']))
    #
    # results.to_csv('test.csv')
print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))