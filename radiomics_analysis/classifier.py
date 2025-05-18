# -*- coding: utf-8 -*- 
"""
@Time : 2024/9/3 17:20 
@Author :   liuyang 
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
@File :     random_forest.py 
"""
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as _LogisticRegression


class _Dataset(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class MLP(nn.Module):
    def __init__(self, in_features, reduction, device='cpu'):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_features // reduction, out_features=1)
        )
        self.clf.to(device)
        self.optim = torch.optim.Adam(params=self.parameters(), lr=0.001)
        self.device = device

    def start_train(self, x, y):
        dataloader = DataLoader(_Dataset(x, y), batch_size=128, num_workers=0)
        loss = torch.nn.BCEWithLogitsLoss()
        for e in tqdm(range(1000)):
            for data in dataloader:
                x, y = data
                x = x / x.norm(dim=0)
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.clf(x).flatten()
                loss(pred, y.float()).backward()
                self.optim.step()
                self.optim.zero_grad()

    @torch.no_grad()
    def predict(self, x):
        x = torch.tensor(np.array(x, dtype=np.float32))
        x = x / x.norm(dim=0)
        return self.clf(x.to(self.device)).sigmoid().flatten().cpu().numpy()

    def metrics(self, y, pred):
        return {
            'auc': roc_auc_score(y, pred),
            'acc': accuracy_score(y, pred > 0.5),
            'precision': precision_score(y, pred > 0.5),
            'recall': recall_score(y, pred > 0.5),
        }


class Classifier:
    def start_train(self, x, y):
        if self.feature_num is None:
            self.feature_num = x.shape[1]
        self.clf_rf.fit(x, y)

    def predict(self, x):
        if self.feature_num is None:
            self.feature_num = x.shape[1]
        return self.clf_rf.predict_proba(x)

    def weight(self):
        return self.clf_rf.coef_

    def bias(self):
        return self.clf_rf.intercept_

    def metrics(self, y, pred):
        return {
            'auc': roc_auc_score(y, pred),
            'acc': accuracy_score(y, pred > 0.5),
            'precision': precision_score(y, pred > 0.5),
            'recall': recall_score(y, pred > 0.5, ),
        }

    def important_features(self, features, k):
        f_importances = self.clf_rf.feature_importances_
        f_names = np.arange(self.feature_num)
        f_std = np.std([tree.feature_importances_ for tree in self.clf_rf.estimators_], axis=0)
        zzs = sorted(zip(f_importances, f_names, f_std), key=lambda x: x[0], reverse=True)
        idx = [zz[1] for zz in zzs][:k]
        features = features[features.columns[idx]]
        return features


class RandomForest(Classifier):
    def __init__(self, n_estimators=100, feature_num=None):
        self.clf_rf = RandomForestClassifier(n_estimators=n_estimators)
        self.feature_num = feature_num


class LogisticRegression(Classifier):
    def __init__(self, feature_num=None, C=1.0, penalty='l2', solver='lbfgs', max_iter=10000):
        self.clf_rf = _LogisticRegression(random_state=0, C=C, penalty=penalty, solver=solver,
                                          max_iter=max_iter)
        self.feature_num = feature_num

    def metrics(self, y, pred, prefix=''):
        return {
            f'{prefix}_acc': accuracy_score(y, pred[:, -1] > 0.5),
            f'{prefix}_auc_macro': roc_auc_score(y, pred[:, -1], average='macro'),
            f'{prefix}_auc_weighted': roc_auc_score(y, pred[:, -1], average='weighted'),
            # f'{prefix}_precision_binary': precision_score(y, pred[:, -1] > 0.5, average='binary'),
            f'{prefix}_precision_macro': precision_score(y, pred[:, -1] > 0.5, average='macro'),
            f'{prefix}_precision_weighted': precision_score(y, pred[:, -1] > 0.5, average='weighted'),
            # f'{prefix}_recall_binary': recall_score(y, pred[:, -1] > 0.5, average='binary'),
            f'{prefix}_recall_macro': recall_score(y, pred[:, -1] > 0.5, average='macro'),
            f'{prefix}_recall_weighted': recall_score(y, pred[:, -1] > 0.5, average='weighted'),
        }


class SVM(RandomForest):
    def __init__(self):
        super().__init__()
        from sklearn import svm
        self.clf_rf = svm.SVC(gamma='auto')
