import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(font_scale=0.8)

df_meta_features = pd.DataFrame(np.array([
    ['General (Linear distance function)', 0.723076923],
    ['Statistical (KMeans distance function)', 0.661538462],
    ['Information theory (RPCA distance function)', 0.676923077],
    ['pyMFE (KMeans distance function)', 0.707692308],
    ['Autoencoder (KMeans distance function)', 0.7846153846],
    ['NN', 0.65],
    ['Random', 0.5],
    ['Domain-specific (RPCA distance function)', 0.846153846],
    ]), columns=['meta-feature','Algorithm match'])

g = sns.factorplot(
    data=df_meta_features, kind="bar",
    x="meta-feature", y="Algorithm match",
    color="#618ad5"
)
g.despine(left=True)
g.set_axis_labels("Meta-features", "Performance of model selection using different meta-features")
g.set_xticklabels(rotation=20)
plt.show()


