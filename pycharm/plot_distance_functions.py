import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(font_scale=0.5)

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(10,8))

## 93 meta features
# df93 = pd.DataFrame(np.array([
#     [1, 0.4, 0.880923523],
#     [2, 0.6, 0.931109521],
#     [3, 0.5, 0.8798707]
#     ]), columns=['K','Algorithm match', 'Difference between the best F1 scores and obtained F1 scores'])
#
# # ## 2 meta features
# df2 = pd.DataFrame(np.array([
#     [1, 0.8, 0.92253545],
#     [2, 0.5, 0.894614156],
#     [3, 0.6, 0.931109521]
#     ]), columns=['K','Algorithm match', 'Difference between the best F1 scores and obtained F1 scores'])
#
# # ## 56 meta features
# df56 = pd.DataFrame(np.array([
#     [1, 0.7, 0.893640883],
#     [2, 0.5, 0.880983161],
#     [3, 0.5, 0.880983161]
#     ]), columns=['K','Algorithm match', 'Difference between the best F1 scores and obtained F1 scores'])
#
# # ## 7 meta features
# df7 = pd.DataFrame(np.array([
#     [1, 0.7, 0.893640883],
#     [2, 0.5, 0.880983161],
#     [3, 0.4, 0.880923523]
#     ]), columns=['K','Algorithm match', 'Difference between the best F1 scores and obtained F1 scores'])
#
#
# sns.barplot(
#     data=df93,
#     x="K", y="Algorithm match",
#     ci="sd", ax=ax1
# )
# sns.barplot(
#     data=df2,
#     x="K", y="Algorithm match",
#     ci="sd", ax=ax2
# )
# sns.barplot(
#     data=df56,
#     x="K", y="Algorithm match",
#     ci="sd", ax=ax3
# )
# sns.barplot(
#     data=df7,
#     x="K", y="Algorithm match",
#     ci="sd", ax=ax4
# )
#
# ax1.set_xlabel("K")
# ax2.set_xlabel("K")
# ax3.set_xlabel("K")
# ax4.set_xlabel("K")
#
# ax1.set_ylabel("Difference between the best F1 scores and obtained F1 scores")
# ax2.set_ylabel("Difference between the best F1 scores and obtained F1 scores")
# ax3.set_ylabel("Difference between the best F1 scores and obtained F1 scores")
# ax4.set_ylabel("Difference between the best F1 scores and obtained F1 scores")
# plt.show()
#
# g = sns.factorplot(
#     data=df_anomaly_space, kind="bar",
#     x="K", y="Algorithm match",
#     ci="sd"
# )
# g.despine(left=True)
# g.set_axis_labels("K", "Algorithm match")

# g = sns.factorplot(
#     data=df_anomaly_space, kind="bar",
#     x="K", y="Difference between the best F1 scores and obtained F1 scores",
#     ci="sd"
# )
# g.despine(left=True)
# g.set_axis_labels("K", "Difference between the best F1 scores and obtained F1 scores")
# plt.show()
#
## all_predefined
df_distance_functions = pd.DataFrame(np.array([
    ['KMeans', 0.707692308],
    ['Linear', 0.523076923],
    ['RPCA', 0.4],
    ['Autoencoder', 0.4],
    ['Gaussian', 0.12]
    ]), columns=['algorithm','performance'])

g = sns.factorplot(
    data=df_distance_functions, kind="bar",
    x="algorithm", y="performance",
    color="#618ad5"
)
g.despine(left=True)
g.set_axis_labels("Distance function", "Performance of model selection using pyMFE meta-features")
plt.show()

