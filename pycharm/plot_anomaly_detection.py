import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(font_scale=0.5)

df = pd.read_excel(r'~/Desktop/AnomalyDetection.xlsx', sheet_name='Anomaly detection results',
                   engine='openpyxl')

df['f1'] = df['f1'].apply(lambda x: x*100)
df = df.sort_values(by=['f1'], ascending=False)

g = sns.factorplot(
    data=df, kind="bar",
    x="dataset", y="f1",
    ci="sd", color="blue"
)
g.despine(left=True)
g.set_axis_labels("Datasets", "Best F1 scores")
g.set_xticklabels(rotation=30)
plt.show()

# df_filtered = df.loc[df['method'] == 'RPCA']
# df_filtered = df.loc[df['method'] == 'Linear']
# df_filtered = df.loc[df['method'] == 'Gaussian']
# df_filtered = df.loc[df['method'] == 'AutoencoderModel']
# df_filtered = df.loc[df['method'] == 'KMeans']
#
# g = sns.factorplot(
#     data=df_filtered, kind="bar",
#     x="dataset", y="f1",
#     ci="sd", color="blue"
# )
# g.despine(left=True)
# g.set_axis_labels("Datasets", "Best F1 scores")
# g.set_xticklabels(rotation=30)
# plt.show()

#
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(10,8))
#
# sns.barplot(
#     data=df.loc[df['method'] == 'RPCA'],
#     x="dataset", y="f1",
#     ci="sd", color="blue", ax=ax1
# )
# sns.barplot(
#     data=df.loc[df['method'] == 'Linear'],
#     x="dataset", y="f1",
#     ci="sd", color="blue", ax=ax2
# )
# sns.barplot(
#     data=df.loc[df['method'] == 'Gaussian'],
#     x="dataset", y="f1",
#     ci="sd", color="blue", ax=ax3
# )
# sns.barplot(
#     data=df.loc[df['method'] == 'AutoencoderModel'],
#     x="dataset", y="f1",
#     ci="sd", color="blue", ax=ax4
# )
# sns.barplot(
#     data=df.loc[df['method'] == 'KMeans'],
#     x="dataset", y="f1",
#     ci="sd", color="blue", ax=ax5
# )
# ax1.tick_params(axis='x', rotation=30)
# ax2.tick_params(axis='x', rotation=30)
# ax3.tick_params(axis='x', rotation=30)
# ax4.tick_params(axis='x', rotation=30)
# ax5.tick_params(axis='x', rotation=30)
#
# ax1.set_xlabel("Datasets with the best RPCA algorithm")
# ax2.set_xlabel("Datasets with the best Linear algorithm")
# ax3.set_xlabel("Datasets with the best Gaussian algorithm")
# ax4.set_xlabel("Datasets with the best Autoencoder algorithm")
# ax5.set_xlabel("Datasets with the best KMeans algorithm")
#
# ax1.set_ylabel("f1 scores")
# ax2.set_ylabel("f1 scores")
# ax3.set_ylabel("f1 scores")
# ax4.set_ylabel("f1 scores")
# ax5.set_ylabel("f1 scores")
# plt.show()



# df_overall = df.groupby(['method']).count()
# df_overall['method'] = df_overall.index
# # print(df_overall)
# g = sns.factorplot(
#     data=df_overall, kind="bar",
#     x="method", y="f1",
#     ci="sd", color="blue"
# )
# g.despine(left=True)
# g.set_axis_labels("Datasets", "Best methods count")
# plt.show()

# df_type_of_data = pd.DataFrame(np.array([
#     ['AutoencoderModel', 0, 'high_dimensional'],
#     ['AutoencoderModel', 21, 'temporal'],
#     ['AutoencoderModel', 0, 'spatial'],
#     ['AutoencoderModel', 0, 'nominal'],
#
#     ['Gaussian', 1, 'high_dimensional'],
#     ['Gaussian', 2, 'temporal'],
#     ['Gaussian', 1, 'spatial'],
#     ['Gaussian', 0, 'nominal'],
#
#     ['KMeans', 5, 'high_dimensional'],
#     ['KMeans', 50, 'temporal'],
#     ['KMeans', 0, 'spatial'],
#     ['KMeans', 1, 'nominal'],
#
#     ['Linear', 2, 'high_dimensional'],
#     ['Linear', 24, 'temporal'],
#     ['Linear', 0, 'spatial'],
#     ['Linear', 0, 'nominal'],
#
#     ['RPCA', 1, 'high_dimensional'],
#     ['RPCA', 20, 'temporal'],
#     ['RPCA', 0, 'spatial'],
#     ['RPCA', 0, 'nominal']]), columns=['method','count', 'type'])
# #
# # g = sns.factorplot(
# #     data=df_type_of_data, kind="bar",
# #     x="method", y="count", hue="type",
# #     ci="sd"
# # )
# # g.despine(left=True)
# # g.set_axis_labels("Datasets", "Best methods count")
# # plt.show()
# #
# df_anomaly_type = pd.DataFrame(np.array([
#     ['AutoencoderModel', 0, 'cluster'],
#     ['AutoencoderModel', 8, 'global'],
#     ['AutoencoderModel', 13, 'local'],
#
#     ['Gaussian', 0, 'cluster'],
#     ['Gaussian', 0, 'global'],
#     ['Gaussian', 3, 'local'],
#
#     ['KMeans', 2, 'cluster'],
#     ['KMeans', 15, 'global'],
#     ['KMeans', 39, 'local'],
#
#     ['Linear', 1, 'cluster'],
#     ['Linear', 9, 'global'],
#     ['Linear', 17, 'local'],
#
#     ['RPCA', 0, 'cluster'],
#     ['RPCA', 8, 'global'],
#     ['RPCA', 13, 'local']]), columns=['method','count', 'type'])
#
# # g = sns.factorplot(
# #     data=df_anomaly_type, kind="bar",
# #     x="method", y="count", hue="type",
# #     ci="sd"
# # )
# # g.despine(left=True)
# # g.set_axis_labels("Datasets", "Best methods count")
# # plt.show()
# #
# df_anomaly_space = pd.DataFrame(np.array([
#     ['AutoencoderModel', 21, 'univariate'],
#     ['AutoencoderModel', 0, 'multivariate'],
#
#     ['Gaussian', 2, 'univariate'],
#     ['Gaussian', 1, 'multivariate'],
#
#     ['KMeans', 49, 'univariate'],
#     ['KMeans', 6, 'multivariate'],
#
#     ['Linear', 24, 'univariate'],
#     ['Linear', 2, 'multivariate'],
#
#     ['RPCA', 19, 'univariate'],
#     ['RPCA', 1, 'multivariate']]), columns=['method','count', 'type'])
#
# # g = sns.factorplot(
# #     data=df_anomaly_space, kind="bar",
# #     x="method", y="count", hue="type",
# #     ci="sd"
# # )
# # g.despine(left=True)
# # g.set_axis_labels("Datasets", "Best methods count")
# # plt.show()
# #
#
# df_anomaly_ratio = pd.DataFrame(np.array([
#     ['AutoencoderModel', 21, 'high'],
#     ['AutoencoderModel', 0, 'low'],
#
#     ['Gaussian', 1, 'high'],
#     ['Gaussian', 2, 'low'],
#
#     ['KMeans', 5, 'high'],
#     ['KMeans', 50, 'low'],
#
#     ['Linear', 2, 'high'],
#     ['Linear', 24, 'low'],
#
#     ['RPCA', 0, 'high'],
#     ['RPCA', 20, 'low']]), columns=['method','count', 'type'])
#
# # g = sns.factorplot(
# #     data=df_anomaly_space, kind="bar",
# #     x="method", y="count", hue="type",
# #     ci="sd"
# # )
# # g.despine(left=True)
# # g.set_axis_labels("Datasets", "Best methods count")
# # plt.show()
#
# df_data_domain = pd.DataFrame(np.array([
#     ['AutoencoderModel', 0, 'manufacturing'],
#     ['AutoencoderModel', 3, 'transport'],
#     ['AutoencoderModel', 3, 'finance'],
#     ['AutoencoderModel', 0, 'medicine'],
#     ['AutoencoderModel', 0, 'text'],
#     ['AutoencoderModel', 8, 'software'],
#     ['AutoencoderModel', 7, 'social'],
#
#     ['Gaussian', 2, 'manufacturing'],
#     ['Gaussian', 0, 'transport'],
#     ['Gaussian', 0, 'finance'],
#     ['Gaussian', 0, 'medicine'],
#     ['Gaussian', 0, 'text'],
#     ['Gaussian', 1, 'software'],
#     ['Gaussian', 0, 'social'],
#
#     ['KMeans', 0, 'manufacturing'],
#     ['KMeans', 7, 'transport'],
#     ['KMeans', 6, 'finance'],
#     ['KMeans', 2, 'medicine'],
#     ['KMeans', 1, 'text'],
#     ['KMeans', 28, 'software'],
#     ['KMeans', 9, 'social'],
#
#     ['Linear', 1, 'manufacturing'],
#     ['Linear', 5, 'transport'],
#     ['Linear', 3, 'finance'],
#     ['Linear', 0, 'medicine'],
#     ['Linear', 1, 'text'],
#     ['Linear', 9, 'software'],
#     ['Linear', 7, 'social'],
#
#     ['RPCA', 0, 'manufacturing'],
#     ['RPCA', 2, 'transport'],
#     ['RPCA', 3, 'finance'],
#     ['RPCA', 0, 'medicine'],
#     ['RPCA', 0, 'text'],
#     ['RPCA', 9, 'software'],
#     ['RPCA', 6, 'social']]), columns=['method','count', 'type'])
#
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(10,8))
#
# sns.barplot(
#     data=df_type_of_data,
#     x="method", y="count", hue="type",
#     ci="sd", ax=ax1
# )
# sns.barplot(
#     data=df_anomaly_type,
#     x="method", y="count", hue="type",
#     ci="sd", ax=ax2
# )
# sns.barplot(
#     data=df_anomaly_space,
#     x="method", y="count", hue="type",
#     ci="sd", ax=ax3
# )
# sns.barplot(
#     data=df_data_domain,
#     x="method", y="count", hue="type",
#     ci="sd", ax=ax4
# )
# sns.barplot(
#     data=df_anomaly_ratio,
#     x="method", y="count", hue="type",
#     ci="sd", ax=ax5
# )
#
# ax1.tick_params(axis='x', rotation=30)
# ax2.tick_params(axis='x', rotation=30)
# ax3.tick_params(axis='x', rotation=30)
# ax4.tick_params(axis='x', rotation=30)
# ax5.tick_params(axis='x', rotation=30)
#
# ax1.set_xlabel("The best algorithms for different data types")
# ax2.set_xlabel("The best algorithms for different anomaly types")
# ax3.set_xlabel("The best algorithms for different anomaly spaces")
# ax4.set_xlabel("The best algorithms for different data domains")
# ax4.set_xlabel("The best algorithms for different anomaly ratio ranges")
#
# ax1.set_ylabel("count")
# ax2.set_ylabel("count")
# ax3.set_ylabel("count")
# ax4.set_ylabel("count")
# ax5.set_ylabel("count")
# plt.show()



#
# g = sns.factorplot(
#     data=df_anomaly_space, kind="bar",
#     x="method", y="count", hue="type",
#     ci="sd"
# )
# g.despine(left=True)
# g.set_axis_labels("Datasets", "Best methods count")
# plt.show()



# df_hd = pd.read_excel(r'~/Desktop/AnomalyDetection.xlsx', sheet_name='High-dimensional overview',
#                    engine='openpyxl')
#
# print(df_hd.columns)
# # df_hd['f1'] = df_hd['f1'].apply(lambda x: x*100)
# df_hd = df_hd.sort_values(by=['f1'], ascending=False)
#
#
# g = sns.factorplot(
#     data=df_hd, kind="bar",
#     x="dataset", y="f1", hue="method",
# )
# g.despine(left=True)
# g.set_axis_labels("Datasets", "Best F1 scores")
# g.set_xticklabels(rotation=10)
# plt.show()

#
#
# random = pd.DataFrame(np.array([
#     ['Gaussian', 0.024],
#     ['Linear', 0.208],
#     ['RPCA', 0.16],
#     ['KMeans', 0.44],
#     ['AutoencoderModel', 0.168],
#     ]), columns=['meta-feature','Algorithm match'])
#
# g = sns.factorplot(
#     data=random, kind="bar",
#     x="meta-feature", y="Algorithm match",
#     color="#618ad5"
# )
# g.despine(left=True)
# g.set_axis_labels("Algorithms", "The best algorithms for datasets from the repository (percentage)")
# plt.show()
# #
f1_dist = pd.DataFrame(np.array([
    ['<=0.2', 13],
    ['>0.2 && <=0.4', 10],
    ['>0.4 && <=0.6', 12],
    ['>0.6 && <=0.8', 17],
    ['>0.8 && <=1', 11],
    ]), columns=['F1 score','Number of datasets'])

g = sns.factorplot(
    data=f1_dist, kind="bar",
    x="F1 score", y="Number of datasets",
    color="#618ad5"
)
g.despine(left=True)
# g.set_axis_labels("Meta-feature", "Algorithm match")
plt.show()