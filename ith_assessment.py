import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import RepeatedKFold

from dataset.subregion import SubRegion
from radiomics_analysis import select_lasso, select_mrmr, select_ttest, select_spearman, select_pearson, \
    RadiomicsHelper, \
    RandomForest, LogisticRegression, MLP
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from monai.utils import set_determinism
from shap import LinearExplainer, summary_plot


def compute_best_threshold_youden(y_true, y_score):
    """
    计算最大约登指数（Youden's J）及对应最佳诊断阈值。

    参数：
    - y_true: 真实标签（0/1）
    - y_score: 预测概率（连续值）

    返回：
    - dict 包括最佳阈值、最大 J、对应的敏感度与特异度
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    J = tpr - fpr
    ix = J.argmax()
    print({
        "best_threshold": thresholds[ix],
        "youden_J": J[ix],
        "sensitivity": tpr[ix],
        "specificity": 1 - fpr[ix]
    })


class ITHAssess:
    def __init__(self, datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped',
                 radioname='data.csv',
                 edfilename='SVmask.csv',
                 select_fn=(select_ttest(),
                            select_spearman(threshold=0.8),
                            # select_pearson(threshold=0.8),
                            # select_mrmr(),
                            select_lasso(
                                # alphas=np.logspace(-3, 1, 50),
                                alphas=None,
                                max_iter=10000)),
                 keep_feat_list=None,
                 clinical_list=None,
                 normalize_method='z-score'
                 ):
        self.datadir = datadir
        self.radiopath = os.path.join(datadir, 'CSV', radioname)
        self.edfilepath = os.path.join(datadir, 'CSV', edfilename)
        self.select_fn = select_fn
        self.normalize = normalize_method
        if clinical_list is None:
            self.clinical_list = ['diameter', 'LVI', 'Ki67_status',
                                  'molecular_type(LuminalA_1, LuminalB_2, HER2过表达_3, TN_4)']
            # self.clinical_list = ['diameter', 'molecular_type(LuminalA_1, LuminalB_2, HER2过表达_3, TN_4)']
        else:
            self.clinical_list = clinical_list
        if keep_feat_list is None:
            self.keep_feat_list = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
        else:
            self.keep_feat_list = keep_feat_list

    def feature_norm(self, features, skip=[]):
        columns = features.columns
        columns = [c for c in columns if c not in skip]
        skip = features[skip]
        features = features[columns]
        if self.normalize == 'z-score':
            data = StandardScaler().fit_transform(features)
        elif self.normalize == 'minmax':
            data = MinMaxScaler().fit_transform(features)
        else:
            data = features
        features = pd.DataFrame(data=data, columns=columns)
        if not skip.empty:
            features = pd.concat([features, skip], axis=1)
        return features

    def feature_clean(self, features, feat_list=None, ignore_columns=None, norm=False):
        if feat_list is None:
            feat_list = self.keep_feat_list
        include_list = ignore_columns if ignore_columns is not None else []

        for c_names in features.columns.to_list():
            if any([feature_type in c_names for feature_type in feat_list]):
                include_list.append(c_names)
        features = features[include_list]
        if norm:
            return self.feature_norm(features)
        return features

    def get_ed_features(self, labels, norm=False, select_features=None, clean=False):
        sr = SubRegion(self.datadir)
        sr.sr_split()
        radio = RadiomicsHelper(filepath=self.edfilepath)
        features = radio.read_features(keys=('path_image', 'svmask')).reset_index(drop=True)
        features = sr.sr_ed(features, k_num=5)
        if 'label' not in features.columns:
            features = features.merge(labels[['path_image', 'label']], on='path_image')
            # features = features.set_index('path_image').loc[labels['path_image']].reset_index()

        if select_features is None:
            select_features = self.select_features(features, norm=norm)
        if clean:
            return (self.feature_clean(features[select_features], norm=norm),
                    features['label'], select_features)
        return features[select_features], select_features

    def get_ed_features_fusion(self, labels, norm=False, select_features=None, clean=False,
                               reduction="weighted", method="feat"):
        sr = SubRegion(self.datadir)
        sr.sr_split()
        radio = RadiomicsHelper(filepath=self.edfilepath)
        features = radio.read_features(keys=('path_image', 'svmask')).reset_index(drop=True)
        features = sr.sr_ed_fusion(features, k_num=5, reduction=reduction, method=method)
        if 'label' not in features.columns:
            features = features.merge(labels[['path_image', 'label']], on='path_image')
            # features = features.set_index('path_image').loc[labels['path_image']].reset_index()

        if select_features is None:
            select_features = self.select_features(features, norm=norm)
        if clean:
            return (self.feature_clean(features[select_features], norm=norm),
                    features['label'], select_features)
        return features[select_features], select_features

    def get_radio_features(self, norm=False, select_features=None, clean=False):
        radio = RadiomicsHelper(filepath=self.radiopath)
        features = radio.read_features(keys=('path_image', 'path_mask')).reset_index(drop=True)
        if select_features is None:
            select_features = self.select_features(features, norm)
        if clean:
            return (self.feature_clean(features[select_features], norm=norm),
                    features['label'], select_features)

        return features[select_features], select_features

    def get_clinical_features(self, norm=False, clean=False):
        radio = RadiomicsHelper(filepath=self.radiopath)
        features = radio.read_features(keys=('path_image', 'path_mask')).reset_index(drop=True)[
            self.clinical_list + ['path_image', 'label']]

        if clean:
            return self.feature_clean(features, feat_list=self.clinical_list, norm=norm), features[
                'label'], self.clinical_list
        return features, self.clinical_list

    def select_features(self, radio_features, norm=False):
        select_radio_features = RadiomicsHelper.select_features(self.select_fn,
                                                                self.feature_clean(radio_features, norm=norm),
                                                                radio_features['label'])
        print(f'selected features: {select_radio_features}')
        return select_radio_features + ['path_image', 'label']

    @staticmethod
    def add_prefix_except(df, prefix, exclude_cols=['path_image', 'label']):
        return df.rename(columns={
            col: f"{prefix}{col}" if col not in exclude_cols else col
            for col in df.columns
        })

    @staticmethod
    def remove_prefix_except(df, prefix, exclude_cols=['path_image', 'label']):
        return df.rename(columns={
            col: col.replace(prefix, '') if col not in exclude_cols else col
            for col in df.columns
        })

    @staticmethod
    def select_by_prefix(df, prefix):
        cols = [col for col in df.columns if col.startswith(prefix)]
        return df[cols]

    def get_all_features(self, norm=False, select_radio_features=None, select_ed_features=None, select_radio_fn=None,
                         select_ed_fn=None, label_path=False):
        clinical_features = self.get_clinical_features(norm)[0]
        if select_radio_fn:
            self.select_fn = select_radio_fn
        radio_features, select_radio_features = self.get_radio_features(norm=norm,
                                                                        select_features=select_radio_features)
        if select_ed_fn:
            self.select_fn = select_ed_fn
        ed_features, select_ed_features = self.get_ed_features(radio_features, norm=norm,
                                                               select_features=select_ed_features)
        ed_features = self.add_prefix_except(ed_features, 'ed_')
        radio_features = self.add_prefix_except(radio_features, 'radio_')
        clinical_features = self.add_prefix_except(clinical_features, 'clinical_')
        concat_features = ed_features.merge(
            radio_features.drop_duplicates().merge(clinical_features, on=['path_image', 'label']),
            on=['path_image', 'label'])

        label = concat_features[['path_image', 'label']] if label_path else concat_features['label']
        concat_features = self.feature_norm(concat_features.drop(columns=['path_image', 'label']))
        ed_only = self.select_by_prefix(concat_features, 'ed_')
        radio_only = self.select_by_prefix(concat_features, 'radio_')
        clinical_only = self.select_by_prefix(concat_features, 'clinical_')
        return ed_only, radio_only, clinical_only, label, select_radio_features, select_ed_features

    def get_concat_features(self, norm=False, select_radio_features=None, select_ed_features=None, select_radio_fn=None,
                            select_ed_fn=None):
        ed_only, radio_only, clinical_only, label, select_radio_features, select_ed_features = self.get_all_features(
            norm, select_radio_features, select_ed_features, select_radio_fn, select_ed_fn)
        concat_features = pd.concat([ed_only, clinical_only], axis=1)
        return concat_features, label, select_radio_features, select_ed_features

    def save_splitted_data(self):
        ed_only, _, clinical_only, label, *_ = self.get_all_features(True, label_path=True)
        train_index, test_index = (
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/train_index.csv')[
                'train_index'],
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/test_index.csv')[
                'test_index'])
        features = pd.concat([ed_only, label], axis=1)
        train, test = features.loc[train_index][['path_image', 'label']], features.loc[test_index][
            ['path_image', 'label']]
        info = pd.read_csv(os.path.join(self.datadir, 'CSV', 'data.csv')).drop_duplicates()
        train = train.drop_duplicates().merge(info, on='path_image')
        test = test.drop_duplicates().merge(info, on='path_image')
        train.to_csv(os.path.join(self.datadir, 'CSV', 'train_rad_split.csv'), index=False)
        test.to_csv(os.path.join(self.datadir, 'CSV', 'test_rad_split.csv'), index=False)

    def ITHscore(self):
        features, _, _, label, *_ = self.get_all_features(True, label_path=True)
        train_index, test_index = (
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/train_index.csv')[
                'train_index'],
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/test_index.csv')[
                'test_index'])
        X_train, X_val, y_train, y_val = (features.loc[train_index], features.loc[test_index],
                                          label['label'].loc[train_index], label['label'].loc[test_index])
        cls = LogisticRegression()
        RadiomicsHelper.train_fold(X_train, X_val, y_train, y_val, cls=cls)
        # ith_score = cls.predict(X_train)[:, 1]
        # compute_best_threshold_youden(y_train, ith_score)
        rna_data = pd.read_excel(os.path.join(self.datadir, 'CSV', 'SLN_with RNA_seq.xlsx'), dtype=str)
        all_data = pd.read_csv(os.path.join(self.datadir, 'CSV', 'data.csv')).rename(
            columns={'SLN状态（0_无转移，1_转移）': 'label'})[['path_image', 'label', 'ID']]
        rna_data = rna_data.merge(all_data, on='ID')
        radio_features = self.add_prefix_except(self.get_radio_features(norm=True)[0], 'Rad_').drop_duplicates()
        ed_features = self.add_prefix_except(self.get_ed_features(radio_features, norm=True)[0],
                                             'ITH_').drop_duplicates()
        ed_features = ed_features.merge(radio_features, on=['path_image', 'label'])
        ed_features = self.feature_norm(ed_features, skip=['path_image', 'label'])
        rna_data_features = rna_data.merge(ed_features, on=['path_image', 'label']).drop_duplicates()
        # rna_data_features.to_csv(os.path.join(self.datadir, 'CSV', 'SLN_with_RNA_features.csv'), index=False)
        rna_info = rna_data_features[rna_data.columns]
        rna_data_features = rna_data_features.drop(columns=rna_data.columns)
        pred = cls.predict(rna_data_features)
        rna_info['ITHscore'] = pred[:, 1]

        rna_info.to_csv(os.path.join(self.datadir, 'CSV', 'SLN_with_RNA_ITH.csv'), index=False)
        pass

    @staticmethod
    def prediction(features, label, X_test, y_test):
        train_index, test_index = (
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/train_index.csv')[
                'train_index'],
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/test_index.csv')[
                'test_index']
        )

        X_train, X_val, y_train, y_val = (features.loc[train_index], features.loc[test_index],
                                          label.loc[train_index], label.loc[test_index])
        metric = RadiomicsHelper.bootstrap_train(X_train, X_val, y_train, y_val, cls=LogisticRegression)
        print(metric)
        metric = RadiomicsHelper.bootstrap_train(X_train, X_test, y_train, y_test, cls=LogisticRegression)
        print(metric)

    @staticmethod
    def plot(features, label, bootstrap=False):
        train_index, test_index = (
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/train_index.csv')[
                'train_index'],
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/test_index.csv')[
                'test_index'])
        X_train, X_val, y_train, y_val = (features.loc[train_index], features.loc[test_index],
                                          label.loc[train_index], label.loc[test_index])
        cls = LogisticRegression()
        if bootstrap:
            probs = RadiomicsHelper.bootstrap_train(X_train, X_val, y_train, y_val, cls=cls, return_probs=True)
        else:
            RadiomicsHelper.train_fold(X_train, X_val, y_train, y_val, cls=cls)
            probs = cls.predict(X_val)[:, 1]
        return probs, y_val

    def explain_linear(self):
        features, _, cli, label, *_ = self.get_all_features(True, label_path=True)
        # features = pd.concat([ed, cli], axis=1)
        train_index, test_index = (
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/train_index.csv')[
                'train_index'],
            pd.read_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/CSV/test_index.csv')[
                'test_index'])
        X_train, X_val, y_train, y_val = (features.loc[train_index], features.loc[test_index],
                                          label['label'].loc[train_index], label['label'].loc[test_index])
        X_train = self.remove_prefix_except(X_train, 'ed_')
        X_val = self.remove_prefix_except(X_val, 'ed_')
        cls = LogisticRegression()
        metrics = RadiomicsHelper.train_fold(X_train, X_val, y_train, y_val, cls=cls)
        print(pd.DataFrame([metrics]).T)
        explainer = LinearExplainer(cls.clf_rf, X_train, feature_perturbation="correlation_dependent")
        shap_values = explainer.shap_values(X_val)
        # plt.figure(figsize=(10, 3), dpi=200)
        summary_plot(shap_values, X_val, feature_names=X_val.columns, plot_type='violin',
                     show=False, max_display=20)
        # plt.xlim(-10, 10)  # 可按实际数据动态调大
        fig = plt.gcf()
        fig.set_size_inches(16, 10)
        fig.set_dpi(300)
        plt.tight_layout()
        plt.savefig(os.path.join(self.datadir, 'figures', "shap_ITH_plot.pdf"), dpi=300)
        plt.show()


def clinical_predictor():
    keep_feat_list = None
    set_determinism(seed=0)
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped',
                         keep_feat_list=keep_feat_list)
    external = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_external/cropped',
                         keep_feat_list=keep_feat_list)
    _, _, clinical_only, label, select_radio_features, select_ed_features = internal.get_all_features(True)
    _, _, X_test, y_test, *_ = external.get_all_features(True, select_radio_features, select_ed_features)
    ITHAssess.prediction(clinical_only, label, X_test, y_test)
    pass


def ed_predictor():
    set_determinism(seed=0)
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped',
                         select_fn=(select_ttest(),
                                    select_spearman(threshold=0.8),
                                    select_mrmr(),
                                    select_lasso(
                                        alphas=None,
                                        max_iter=10000)))
    external = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_external/cropped')
    features, _, _, label, select_radio_features, select_ed_features = internal.get_all_features(True)
    X_test, _, _, y_test, *_ = external.get_all_features(True, select_radio_features, select_ed_features)
    ITHAssess.prediction(features, label, X_test, y_test)
    pass


def radio_predictor():
    set_determinism(seed=0)
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped',
                         select_fn=(select_ttest(),
                                    select_spearman(threshold=0.8),
                                    select_mrmr(),
                                    select_lasso(alphas=None,
                                                 max_iter=10000)))
    external = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_external/cropped')
    _, features, _, label, select_radio_features, select_ed_features = internal.get_all_features(True)
    _, X_test, _, y_test, *_ = external.get_all_features(True, select_radio_features, select_ed_features)
    ITHAssess.prediction(features, label, X_test, y_test)
    pass


def concat_predictor():
    set_determinism(seed=0)
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped')
    external = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_external/cropped')
    features, label, select_radio_features, select_ed_features = internal.get_concat_features(True)
    X_test, y_test, *_ = external.get_concat_features(True, select_radio_features, select_ed_features)
    ITHAssess.prediction(features, label, X_test, y_test)
    pass


def rna():
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped')
    internal.ITHscore()


def plot():
    from radiomics_analysis.visualize import plot_decision_curve, plot_calibration_curve, plot_single_roc, plot_heatmap
    from radiomics_analysis.stats import delong_pvalue_matrix
    set_determinism(seed=0)
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped')
    ed_only, radio_only, clinical_only, label, select_radio_features, select_ed_features = internal.get_all_features(
        True)
    probs_d = {}
    savedir = '/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped/figures'
    os.makedirs(savedir, exist_ok=True)
    for name, feat in zip(['ITH-Model', 'ConvRad-Model', 'ClinicoPathol-Model', 'Com-Model'],
                          [ed_only, radio_only, clinical_only, pd.concat([ed_only, clinical_only], axis=1)]):
        set_determinism(seed=0)
        probs, y_val = internal.plot(feat, label, bootstrap=True)
        probs_d.update({name: probs})
    plot_single_roc(y_val, probs_d, save_path=os.path.join(savedir, f'roc_show_fill_all.pdf'))
    plot_single_roc(y_val, probs_d, save_path=os.path.join(savedir, f'roc_all.pdf'), show_fill=False)

    probs_d = {}
    for name, feat in zip(['ITH-Model', 'ConvRad-Model', 'ClinicoPathol-Model', 'Com-Model'],
                          [ed_only, radio_only, clinical_only, pd.concat([ed_only, clinical_only], axis=1)]):
        set_determinism(seed=0)
        probs, y_val = internal.plot(feat, label, bootstrap=False)
        probs_d.update({name: probs})
        plot_calibration_curve(y_val, {name: probs}, save_path=os.path.join(savedir, f'calibration_{name}.pdf'))
        plot_decision_curve(y_val, {name: probs}, save_path=os.path.join(savedir, f'decision_{name}.pdf'))
    names, matrix = delong_pvalue_matrix(y_val, probs_d)
    plot_heatmap(names, matrix, save_path=os.path.join(savedir, f'heatmap_all.pdf'))
    pass


def plot_noLVI():
    from radiomics_analysis.visualize import plot_decision_curve, plot_calibration_curve, plot_single_roc, plot_heatmap
    from radiomics_analysis.stats import delong_pvalue_matrix
    set_determinism(seed=0)
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped')
    ed_only, _, clinical_only, label, select_radio_features, select_ed_features = internal.get_all_features(True)
    com = pd.concat([ed_only, clinical_only], axis=1)
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped',
                         clinical_list=['diameter', 'Ki67_status',
                                        'molecular_type(LuminalA_1, LuminalB_2, HER2过表达_3, TN_4)'])
    ed_only, _, clinical_only, label, select_radio_features, select_ed_features = internal.get_all_features(True)
    com_nolvi = pd.concat([ed_only, clinical_only], axis=1)
    probs_d = {}
    for name, feat in zip(['Com-Model', 'ComNoLVI-Model'], [com, com_nolvi]):
        set_determinism(seed=0)
        probs, y_val = internal.plot(feat, label, bootstrap=False)
        probs_d.update({name: probs})
    plot_single_roc(y_val, probs_d, show_ci=False)
    names, matrix = delong_pvalue_matrix(y_val, probs_d)
    plot_heatmap(names, matrix)
    pass


def icc_compute():
    from radiomics_analysis.stats import compute_icc_features
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped_icc', select_fn=None)
    ed_only, radio_only, clinical_only, label, select_radio_features, select_ed_features = internal.get_all_features(
        True, label_path=True)
    features_icc = pd.concat([radio_only, label], axis=1)
    features_icc['path_image'] = features_icc['path_image'].map(
        lambda x: x.replace('_icc', '').replace('/data_smr/liuy/Project/BreastCancer/dataset/', ''))
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped', select_fn=None)
    ed_only, radio_only, clinical_only, label, select_radio_features, select_ed_features = internal.get_all_features(
        True, label_path=True)
    features = pd.concat([radio_only, label], axis=1).drop_duplicates(subset='path_image')
    features_sorted = features.set_index('path_image').reindex(features_icc['path_image']).reset_index()
    icc = compute_icc_features(features_sorted.drop(columns=['path_image', 'label']),
                               features_icc.drop(columns=['path_image', 'label']), icc_type='icc3')
    icc = pd.DataFrame(icc).reset_index().rename(columns={'index': 'feature', 0: 'icc'})
    icc.to_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped_icc/CSV/icc3.csv', index=False)

    icc = compute_icc_features(features_sorted.drop(columns=['path_image', 'label']),
                               features_icc.drop(columns=['path_image', 'label']), icc_type='icc2')
    icc = pd.DataFrame(icc).reset_index().rename(columns={'index': 'feature', 0: 'icc'})
    icc.to_csv('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped_icc/CSV/icc2.csv', index=False)
    pass


def explain_linear():
    internal = ITHAssess(datadir='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/cropped',
                         normalize_method='none')
    # internal.explain_linear()
    internal.save_splitted_data()


if __name__ == "__main__":
    # clinical_predictor()
    # radio_predictor()
    # ed_predictor()
    # concat_predictor()
    # icc_compute()
    # search_()
    # rna()
    # plot()
    # plot_noLVI()
    explain_linear()
