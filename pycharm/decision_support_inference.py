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
from algorithms.eucledian import Eucledian
from algorithms.manhattan import Manhattan

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

db = Database("127.0.0.1","root","","anomaly_detection_decision_support")
# db = Database("anomaly-detection-mysql.ch1ih3mzagsi.eu-central-1.rds.amazonaws.com","admin","6xy4AMtnhkFJWfIWHsuu","anomaly_detection_decision_support")

#global_datasets - only for datasets info
#evaluations - evaluations of datasets and algorithms
# global_features - meta-features for datasets
global_datasets, evaluation, global_features = db.get_datasets()

# for dataset in utilities.get_datasets('/Users/miloskotlar/GoogleDrive/Academic/PhD/III/linear_datasets/'):
#     db.update_characterization_user_defined_data(dataset)

# exit(1)
characterization_columns = np.array(global_features.columns)
characterization_columns.shape = (1, global_features.shape[1])

# used for excluding them from the list
user_defined_characterization_columns = ['high-dimensional',
'nominal',
'spatial',
'temporal',

'graphs_and_networks',
'manufacturing',
'transport',
'finance',
'medicine',
'images',
'text',
'software',
'social',

'local',
'global',
'cluster',

'anomaly_space',

'anomaly_ratio',
                                         ]
predefined_characterization_columns = np.setdiff1d(characterization_columns,user_defined_characterization_columns)

general = ['attr_to_inst',
'cat_to_num',
'freq_class',
'inst_to_attr',
'nr_attr',
'nr_bin',
'nr_cat',
'nr_class',
'nr_inst',
'nr_num',
'num_to_cat',
'attr_to_inst',
'cat_to_num',
'freq_class',
'inst_to_attr',
'nr_attr',
'nr_bin',
'nr_cat',
'nr_class',
'nr_inst',
'nr_num',
'num_to_cat']
statistical = [
'can_cor',
'cor',
'cov',
'eigenvalues',
'g_mean',
'gravity',
'h_mean',
'iq_range',
'kurtosis',
'lh_trace',
'mad',
'max',
'mean',
'median',
'min',
'nr_cor_attr',
'nr_disc',
'nr_norm',
'nr_outliers',
'p_trace',
'range',
'roy_root',
'sd',
'sd_ratio',
'skewness',
'sparsity',
't_mean',
'var',
'w_lambda',
'can_cor',
'cor',
'cov',
'eigenvalues',
'g_mean',
'gravity',
'h_mean',
'iq_range',
'kurtosis',
'lh_trace',
'mad',
'max',
'mean',
'median',
'min',
'nr_cor_attr',
'nr_disc',
'nr_norm',
'nr_outliers',
'p_trace',
'range',
'roy_root',
'sd',
'sd_ratio',
'skewness',
'sparsity',
't_mean',
'var',
'w_lambda'
]
theoretical_info = [
'attr_conc',
'attr_ent',
'class_conc',
'class_ent',
'eq_num_attr',
'joint_ent',
'mut_inf',
'ns_ratio',
'attr_conc',
'attr_ent',
'class_conc',
'class_ent',
'eq_num_attr',
'joint_ent',
'mut_inf',
'ns_ratio'
]

characterization_filters=[
'all',
'temporal',
'high-dimensional',
'manufacturing',
'transport',
'finance',
'medicine',
'text',
'software',
'social',
'local',
'global',
'cluster'
]

# used for testing
user_defined_characterization_columns = ['high-dimensional',
'nominal',
'spatial',
'temporal',

# 'graphs_and_networks',
# 'manufacturing',
# 'transport',
# 'finance',
# 'medicine',
# 'images',
# 'text',
# 'software',
# 'social',

'local',
'global',
'cluster',
'anomaly_space',
'anomaly_ratio',]

# characterization_attributes=['all', 'prefedined', 'user_defined']
# characterization_attributes=['general', 'statistical', 'theoretical_info']
# characterization_attributes=['user_defined']

# characterization_attributes=['prefedined', 'user_defined','general', 'statistical', 'theoretical_info']
characterization_attributes=['user_defined']

# custom = ['anomaly_space', 'attr_ent.mean', 'attr_ent.sd', 'can_cor.sd', 'cat_to_num'
#                 , 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd', 'finance'
#                 , 'freq_class.mean', 'g_mean.mean', 'g_mean.sd', 'graphs_and_networks'
#                 , 'gravity', 'h_mean.mean', 'h_mean.sd', 'high_dimensional', 'images'
#                 , 'iq_range.mean', 'iq_range.sd', 'joint_ent.mean', 'joint_ent.sd'
#                 , 'kurtosis.sd', 'mad.mean', 'mad.sd', 'manufacturing', 'max.mean', 'max.sd'
#                 , 'mean.mean', 'mean.sd', 'median.mean', 'median.sd', 'medicine', 'min.mean'
#                 , 'min.sd', 'nr_cat', 'nr_class', 'nr_disc', 'nr_inst', 'num_to_cat'
#                 , 'range.mean', 'range.sd', 'row_count', 'sd.mean', 'sd.sd', 'skewness.mean'
#                 , 'skewness.sd', 'social', 'spatial', 't_mean.mean', 't_mean.sd', 'temporal'
#                 , 'transport', 'var.mean', 'var.sd']

summary_results = []

# for all aspects of interest
for characterization_filter in characterization_filters:
    if characterization_filter == 'all':
        mid_features = global_features
        datasets = global_datasets
    else:
        mid_features = global_features.loc[global_features[characterization_filter] == 1]
        datasets = global_datasets.loc[global_datasets['id'].isin(mid_features.index.to_numpy())]

    # combs = combinations(characterization_columns[0], 2)
    # for comb in combs:
    #     print(comb)
    #     idx = np.in1d(characterization_columns, np.array(comb)).nonzero()[0]
    #     features = mid_features.to_numpy()[:, idx]

    for characterization_attribute in characterization_attributes:
        if characterization_attribute == 'prefedined':
            idx = np.in1d(characterization_columns, predefined_characterization_columns).nonzero()[0]
            features = mid_features.to_numpy()[:, idx]
        # elif characterization_attribute == 'user_defined':
        #     idx = np.in1d(characterization_columns, user_defined_characterization_columns).nonzero()[0]
        #     features = mid_features.to_numpy()[:, idx]
        # else:
        #     features = mid_features.to_numpy()
        elif characterization_attribute == 'user_defined':
            idx = np.in1d(characterization_columns, user_defined_characterization_columns).nonzero()[0]
            features = mid_features.to_numpy()[:, idx]
        elif characterization_attribute == 'general':
            idx = np.in1d(characterization_columns, general).nonzero()[0]
            features = mid_features.to_numpy()[:, idx]
        elif characterization_attribute == 'statistical':
            idx = np.in1d(characterization_columns, statistical).nonzero()[0]
            features = mid_features.to_numpy()[:, idx]
        elif characterization_attribute == 'theoretical_info':
            idx = np.in1d(characterization_columns, theoretical_info).nonzero()[0]
            features = mid_features.to_numpy()[:, idx]

        features = standardize_data(features)

        # temporal = np.array(datasets.loc[datasets['type_of_data'] == '\'temporal\''].axes[0])
        # temporal_features = features[temporal]
        # # spatial = np.array(datasets.loc[datasets['type_of_data'].str.contains('\'spatial\'')].axes[0])
        # # spatial_features = features[spatial]
    #high = np.array(datasets.loc[datasets['type_of_data'] == '\'high-dimensional\''].axes[0])
        #high_features = features[high]


        total = len(datasets)


        # features = high_features
        # features = temporal_features
        print('*** Start:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        print('*** Datasets for testing:', len(features))
        print('*** Environment: ',characterization_filter, characterization_attribute)


        # characterization_columns = ['anomaly_space', 'attr_ent.mean', 'attr_ent.sd', 'can_cor.sd', 'cat_to_num'
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

        # for comb_k in range(2, 3):
        # combs = combinations(characterization_columns, 2)
        # for comb in combs:
        #     features = global_features
        #     comb = np.array(comb)
        # comb = ['anomaly_space', 'attr_ent.mean', 'attr_ent.sd', 'can_cor.sd', 'cat_to_num'
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
        # temp = ['can_cor.sd', 'global',
        # 'can_cor.sd', 'local',
        # 'cat_to_num', 'global',
        # 'cat_to_num', 'local',
        # 'cluster', 'global',
        # 'cluster', 'local',
        # 'anomaly_space', 'global',
        # 'anomaly_space', 'local',
        # 'attr_count', 'global',
        # 'attr_count', 'local',
        # 'can_cor.sd', 'global']
        # temp = np.unique(temp)
        # comb = temp
        # comb = ['anomaly_space', 'kurtosis.sd']
        #comb_k = 17#3
        # idx = np.in1d(characterization_columns, comb).nonzero()[0]
        # features = features[:, idx]

            # features = dimension_reduction(features, 2)




        ### change per purpose
        # features = dimension_reduction(features, 2)

        #
        # fig = plt.figure(figsize=(12, 12))
        # plt.subplots_adjust(hspace=0.5)
        # ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(features[:,0], features[:,1], c='r', s=50, label='temporal')
        #
        #
        # for i in range(len(features)):
        #     text = datasets.loc[datasets['type_of_data'] == '\'temporal\''].iloc[i]['name']
        #     ax.annotate(text, (features[i][0], features[i][1]))
        # # # ax.scatter(temporal_features[:,0], temporal_features[:,1], c='b', s=50, label='temporal')
        # # # ax.scatter(spatial_features[:,0], spatial_features[:,1], c='g', s=50, label='spatial')
        # plt.legend()
        # plt.plot()

        best_method_name_for = ''
        best_method_score_for = 0
        methods = utilities.get_methods()

        # changing meta features and retrieving the results
        for change in np.linspace(0,1,11):
            for eval_method in methods:
                try:
                    if eval_method['name'] in ['Gaussian']:#''#'',,'AutoencoderModel', 'Gaussian', 'Linear', 'Manhattan', 'RPCA', 'KMeans'
                        continue
                    m = eval(eval_method['name'] + '()')

                    ks = np.zeros((10,), dtype=float)
                    method_match = np.zeros((10,), dtype=float)

                    results = pd.DataFrame(columns=['Dataset','Distance','Best method', 'F1 with the best method','K', 'F1 method', 'F1 with the proposed method', 'Accuracy'])


                    for i in range(len(datasets)):
                        print('******')

                        test_features = features[i]
                        test_features.shape = (1,len(features[0]))
                        train_features = features

                        # changing meta features and retrieving the results
                        test_features[0][0] += test_features[0][0]*change
                        test_features[0][2] += test_features[0][2]*change
                        test_features[0][3] += test_features[0][3]*change
                        test_features[0][5] += test_features[0][5]*change

                        try:
                            result, array = m.distance(1, test_features, train_features, datasets, evaluation)
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
                                    # if j==1:
                                    #     print(method, change)
                                if k_method in actual_method:
                                    method_match[j-1]+=1

                                results = results.append(pd.DataFrame(
                                    np.array([[r['dataset_name'],r['distance'],r['method'],r['best_score'],j,k_method, best_score, 1-(actual_score-best_score), change]]),
                                    columns=['Dataset','Distance','Best method', 'F1 with the best method','K', 'F1 method', 'F1 with the proposed method', 'Accuracy', 'Change']))

                            results = results.append(pd.Series(), ignore_index=True)
                            # if best_method == actual_method:
                            #     sum += 1
                        # expected_score, expected_method, expected_params = m.predict(i, datasets, evaluation, best_method, best_params)
                        except:
                            print('Error')
                    # results = results.append(df)
                    # if sum/total >0.5:
                    # print('For combs %s iteration %s K=%s precision is %s' % (comb, counter, k,sum/total))

                    ks /= float(total)
                    method_match /= float(total)

                    if best_method_score_for<method_match[0]:
                        best_method_score_for = method_match[0]
                        best_method_name_for = eval_method['name']




                    # if (method_match[0]>0.9):
                    #     print(comb, method_match)


                    results = results.append(pd.DataFrame(
                            np.array([['K', 'Method match', 'Accuracy']]),
                            columns=[ 'K', 'F1 with the proposed method', 'Accuracy']))

                    for index, item in enumerate(ks):
                        results = results.append(pd.DataFrame(
                            np.array([[int(index+1), method_match[index], item]]),
                            columns=['K', 'F1 with the proposed method', 'Accuracy']))

                    results.to_csv('results/%s_%s_%s.csv' % (characterization_filter, characterization_attribute, eval_method['name']))
                    print('*** End:', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                except:
                    print('Except error')
            summary_results.append((characterization_filter, characterization_attribute, best_method_name_for, best_method_score_for, total, change))
            print(summary_results)
print(summary_results)