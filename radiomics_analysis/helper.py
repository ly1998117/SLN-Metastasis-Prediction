# -*- coding: utf-8 -*- 
"""
@Time : 2024/9/3 17:23 
@Author :   liuyang 
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
@File :     helprt.py 
"""
import os
import warnings

import mpire as mpi

from matplotlib import pyplot as plt
from inspect import isclass
from sklearn.manifold import TSNE
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


def _bootstrap_train_once(X_train, X_test, y_train, y_test, cls, return_probs=False):
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_resampled = X_train.iloc[indices]
    y_resampled = y_train.iloc[indices]
    return RadiomicsHelper.train_fold(X_resampled, X_test, y_resampled, y_test, cls, return_probs)


def _grid_search_fn(seed, fold, X_train, X_test, y_train, y_test, cls):
    result = RadiomicsHelper.train_fold(X_train, X_test, y_train, y_test, cls)
    result.update({
        'seed': seed,
        'fold': fold
    })
    return result


class RadiomicsHelper:
    def __init__(self, filepath, params='/data_smr/liuy/Project/BreastCancer/radiomics_analysis/params.yaml'):
        self.dir_path = os.path.dirname(os.path.dirname(filepath))
        self.filename = os.path.basename(filepath).split('.')[0]
        self.params = params
        self.data = pd.read_csv(filepath)
        self.data = self.data.rename(columns={'SLN状态（0_无转移，1_转移）': 'label'})

    @staticmethod
    def featname_split(features, feat_list=None):
        # Since some features are not used, we only select first order and texture features for now
        if feat_list is None:
            feat_list = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
        exclude_list = []
        rest_cname_list = []
        for feature_type in feat_list:
            for c_names in features.columns.to_list():
                if feature_type in c_names:
                    rest_cname_list.append(c_names)
                else:
                    exclude_list.append(c_names)
        return rest_cname_list, exclude_list

    @staticmethod
    def normalize(X):
        colNames = X.columns  # to read the feature's name
        X = X.astype(np.float64)
        X = StandardScaler().fit_transform(X)
        X = pd.DataFrame(X)
        X.columns = colNames
        return X

    @staticmethod
    def standardization(features):
        features_data = np.array(features)
        process_data = MinMaxScaler().fit_transform(features_data)
        return process_data

    @staticmethod
    def _catch_features(data, params, keys=('path_image', 'path_mask'), deep=1):
        image = data[keys[0]]
        mask = data[keys[1]]
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(params)
            # extractor.enableAllFeatures()
            print(f'Extracting {os.path.basename(image)}')
            result = dict(extractor.execute(image, mask))
            data.update(result)
            return data
        except Exception as e:
            print(f'FEATURE EXTRACTION FAILED {e}: Image {image}, Mask {mask}')
            # data_array = nibabel.load(image).get_fdata()
            # mask_array = nibabel.load(mask).get_fdata()
            # affine = nibabel.load(image).affine
            # affine[:3, :3] = np.eye(3)
            # nibabel.save(nibabel.Nifti1Image(data_array, affine), image)
            # nibabel.save(nibabel.Nifti1Image(mask_array, affine), mask)
            return {}

    def multi_catch_features(self, keys=('path_image', 'path_mask')):
        for k in keys:
            self.data[k] = self.data[k].map(lambda x: os.path.join(
                os.path.dirname(os.path.dirname(self.dir_path)), x
            ))
        with mpi.WorkerPool(n_jobs=32) as pool:
            results = pool.map(self._catch_features,
                               zip(self.data.to_dict(orient='records'), [self.params] * len(self.data),
                                   [keys] * len(self.data)))
        # results = []
        # for data, mask in zip(datas, masks):
        #     results.append(self._catch_features(data, mask))
        return pd.DataFrame(results)

    def read_features(self, keys=None):
        if os.path.exists(f'{self.dir_path}/CSV/{self.filename}_features.csv'):
            features = pd.read_csv(f'{self.dir_path}/CSV/{self.filename}_features.csv')
        else:
            features = self.multi_catch_features(keys).reset_index(drop=True)
            features.to_csv(f'{self.dir_path}/CSV/{self.filename}_features.csv', index=False)
        # features = features.merge(data, on='path_image').fillna(0).reset_index(drop=True)
        # path, label = features['path_image'], features['label']
        # features = features.drop(columns='label')
        return features

    @staticmethod
    def bootstrap_train(X_train, X_test, y_train, y_test, cls, n_iterations=1000, ci_percentile=95, return_probs=False):
        args_list = [(X_train, X_test, y_train, y_test, cls, return_probs) for _ in range(n_iterations)]
        with mpi.WorkerPool(n_jobs=32, start_method='spawn', ) as pool:
            metrics_results = pool.map(_bootstrap_train_once, args_list, progress_bar=True)
        if return_probs:
            return metrics_results
        metrics_results = pd.DataFrame(metrics_results)
        # 计算置信区间
        lower_percentile = (100 - ci_percentile) / 2
        upper_percentile = 100 - lower_percentile
        lower_bound = metrics_results.quantile(lower_percentile / 100, axis=0)
        upper_bound = metrics_results.quantile(upper_percentile / 100, axis=0)
        return pd.DataFrame({
            'Mean': metrics_results.mean(),
            f'{ci_percentile}% Lower Bound': lower_bound,
            f'{ci_percentile}% Upper Bound': upper_bound
        })

    @staticmethod
    def bootstrap_test(X_train, X_test, y_train, y_test, cls, n_iterations=1000, ci_percentile=95):
        cls = cls()
        cls.start_train(X_train, y_train)
        metrics_results = []
        for _ in tqdm(range(n_iterations), desc='Bootstrap testing'):
            indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
            X_test_resampled, y_test_resampled = X_test.iloc[indices], y_test.iloc[indices]
            # 计算评估结果
            pred_test = cls.predict(X_test_resampled)
            result = cls.metrics(y_test_resampled, pred_test)
            metrics_results.append(result)
        metrics_results = pd.DataFrame(metrics_results)
        # 计算置信区间
        lower_percentile = (100 - ci_percentile) / 2
        upper_percentile = 100 - lower_percentile
        lower_bound = metrics_results.quantile(lower_percentile / 100, axis=0)
        upper_bound = metrics_results.quantile(upper_percentile / 100, axis=0)
        return pd.DataFrame({
            'Mean': metrics_results.mean(),
            f'{ci_percentile}% Lower Bound': lower_bound,
            f'{ci_percentile}% Upper Bound': upper_bound
        })

    @staticmethod
    def train_fold(X_train, X_test, y_train, y_test, cls=None, return_probs=False):
        # 使用 SMOTE 算法进行数据过采样
        # print("Applying SMOTE...")
        # smote = SMOTE(random_state=42)
        # X_train, y_train = smote.fit_resample(X_train, y_train)
        #
        # # 输出过采样后的类别分布
        # print(f"Resampled class distribution: {Counter(y_train)}")
        if isclass(cls):
            cls = cls()
        cls.start_train(X_train, y_train)
        pred_train = cls.predict(X_train)
        pred_test = cls.predict(X_test)
        if return_probs:
            return pred_test[:, -1].reshape(1, -1)
        # print(f'Training result: {cls.metrics(y_train, pred_train)}')
        result = {**cls.metrics(y_train, pred_train, 'train'), **cls.metrics(y_test, pred_test, 'test')}
        # print(f'Testing result: {result}')
        return result

    @staticmethod
    def select_features(select_fn, features, label):
        # PCA 提取
        select_features = features.columns
        if select_fn is not None:
            for fn in select_fn:
                select_features = fn(features[select_features], label)
        return list(select_features)

    @staticmethod
    def visualize_features(features, label, description=None):
        print("Performing t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)
        tsne_df = pd.DataFrame({
            'Dim1': features_tsne[:, 0],
            'Dim2': features_tsne[:, 1],
            'Label': label
        })
        plt.figure(figsize=(8, 6))
        sns.set_theme(style='white', font_scale=1.5)
        sns.scatterplot(
            x='Dim1', y='Dim2',
            hue='Label',
            palette='Set1',
            data=tsne_df,
            alpha=0.7
        )
        if description is not None:
            plt.title(description)
        else:
            plt.title('t-SNE Visualization of Selected Features')
        plt.legend(title='Label')
        plt.show()

    def __call__(self):
        features = self.read_features()
        path, label = features['path_image'], features['label']
        included_columns = [x for i, x in enumerate(features.columns) if type(features.iat[1, i]) != str]
        features = self.normalize(features[included_columns].apply(pd.to_numeric, errors='ignore'))
        self.visualize_features(features, label, description='t-SNE Visualization of All Features')
        print(f'Read features {features.shape}')
        select_features = self.select_features(features, label)
        process_data = features[select_features].reset_index(drop=True)
        print(f'Selected features {process_data.shape} : {select_features}')
        self.visualize_features(process_data, label)
        self.train(process_data, label)
