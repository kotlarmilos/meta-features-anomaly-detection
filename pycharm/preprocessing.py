import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn import preprocessing
from sklearn.decomposition import PCA


def load_data(dataset):
    df = pd.DataFrame()
    for filename in dataset['files']:
        df_temp = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False, index_col=None)
        has_header = any(str(cell).isdigit() for cell in df_temp.iloc[0])
        if df.columns.size != 0:
            df_temp.columns = df.columns
        elif not has_header:
            header = []
            for i in range(0, df_temp.shape[1]):
                header.append('p%d' % i)
            header[df_temp.shape[1]-1] = 'label'
            df_temp.columns = header
        df = pd.concat([df, df_temp], axis=0)

    # convert objects to category codes
    for col_name in df.columns:
        if df[col_name].dtype == 'object':
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes.astype('int64')

    # split dataset into features and target arrays
    features = target = None
    if dataset['label']:
        features = np.nan_to_num(df.drop(columns=[dataset['label']]).to_numpy())
        target = df[dataset['label']].to_numpy()
    else:
        features = df.to_numpy()

    anomaly_ratio = np.count_nonzero(target)/len(target)

    return features, target, anomaly_ratio


def standardize_data(features):
    scaler = preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    return np.nan_to_num(features)


def dimension_reduction(features, n):
    pca = PCA(n_components=n)
    return pca.fit_transform(features)


def characterize_data(dataset, features, target):
    mfe = MFE(groups=["general", "statistical", "info-theory"], suppress_warnings=True)
    if dataset['label']:
        mfe.fit(features, target)
    else:
        mfe.fit(features)
    ft = mfe.extract()
    return np.nan_to_num(ft)

    # def evaluate(self, device, methods, dataset):
    #     isSupervised = self.target is not None
    #     for method in methods:
    #         if method['isSupervised'] == isSupervised:
    #             print('Running', method['name'])
    #             if method['name'] == 'gaussian':
    #                 result = gaussian(features)
    #             elif method['name'] == 'linear_regression':
    #                 result = linear(features)
    #             elif method['name'] == 'pca':
    #                 result = pca(features)
    #             elif method['name'] == 'kmeans':
    #                 result = kmeans(features)
    #             elif method['name'] == 'neural_network':
    #                 result = nn(features, target)

    # insert_evaluation_info(device, method, dataset, result)
