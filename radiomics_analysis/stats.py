# -*- coding: utf-8 -*- 
"""
@Time : 2024/9/3 14:35 
@Author :   liuyang 
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
@File :     t_test.py 
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from scipy.stats import levene, ttest_ind, mannwhitneyu, norm, spearmanr
from feature_engine.selection import MRMR
from scipy import stats


def remove_highly_correlated_features(data, threshold=0.85, method='pearson'):
    """
    去除高相关的特征，保证特征之间冗余度小
    :param data: 输入的 DataFrame 数据
    :param threshold: 相关性阈值，去除相关性高于该值的特征
    :param method: 计算相关性的方法，'pearson' 或 'spearman'
    :return: 剩余的特征 DataFrame
    """
    # 创建相关系数矩阵
    if method == 'pearson':
        corr_matrix = data.corr(method='pearson').abs()  # Pearson 相关系数矩阵
    elif method == 'spearman':
        corr_matrix = data.corr(method='spearman').abs()  # Spearman 相关系数矩阵
    else:
        raise ValueError("Method should be either 'pearson' or 'spearman'")
    corr_matrix = corr_matrix.fillna(1)
    # 去掉对角线上的元素（自己与自己的相关性）
    np.fill_diagonal(corr_matrix.values, 0)

    # 创建一个布尔值的mask，标记相关性大于阈值的部分
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[i]
                to_drop.add(colname)

    # 返回去除高相关性特征后的数据
    return data.drop(columns=to_drop).columns


def select_spearman(threshold=0.85):
    def fn(data, y):
        return remove_highly_correlated_features(data, threshold, 'spearman')

    return fn


def select_pearson(threshold=0.85):
    def fn(data, y):
        return remove_highly_correlated_features(data, threshold, 'pearson')

    return fn


def select_var(x, threshold=1e10):
    selector = VarianceThreshold(threshold=threshold)  # 注意修改参数达到筛选目的
    selector.fit_transform(x)
    return x.columns[selector.get_support()]


def select_ttest(t_threshold=0.05):
    def fn(x, y):
        selected_features = []
        data_1 = x[y == 0]
        data_2 = x[y == 1]
        for colName in data_1.columns:
            equal_var = levene(data_1[colName], data_2[colName])[1] > t_threshold
            p_value = ttest_ind(data_1[colName], data_2[colName], equal_var=equal_var)[1]
            if p_value < t_threshold:
                selected_features.append(colName)
            else:
                # 如果 t 检验未显著，进行 Mann-Whitney U 检验
                if mannwhitneyu(data_1[colName], data_2[colName])[1] > t_threshold:
                    selected_features.append(colName)
        return selected_features

    return fn


def select_mrmr(method="MIQ", regression=False, max_features=None):
    sel = MRMR(method=method, regression=regression, max_features=max_features)

    def fn(x, y):
        sel.fit(x, y)
        selected_features = [n for n in x.columns if n not in sel.features_to_drop_]
        return selected_features

    return fn


def compute_icc_features(df1: pd.DataFrame,
                         df2: pd.DataFrame,
                         icc_type: str = 'icc3', ):
    """
    Efficient ICC computation between two omics feature matrices.

    Args:
        df1, df2: DataFrames of shape [N, D], matching index/columns
        icc_type: 'icc3' or 'icc2'
        threshold: minimum ICC to keep feature
        return_passed_only: if True, return only passing features' names

    Returns:
        - pd.Series of ICC scores per feature
        - OR List[str] of passed feature names if return_passed_only=True
    """
    assert df1.shape == df2.shape
    assert (df1.columns == df2.columns).all()

    X1, X2 = df1.values, df2.values
    N, D = X1.shape

    if icc_type == 'icc3':
        # Fast vectorized ICC(3,1)
        mgrand = (X1 + X2).mean(axis=0) / 2

        msb = 2 * (((((X1 + X2) / 2) - mgrand) ** 2).sum(axis=0)) / (N - 1)
        mse = ((((X1 - ((X1 + X2) / 2)) ** 2 +
                 (X2 - ((X1 + X2) / 2)) ** 2)).sum(axis=0)) / (2 * (N - 1))

        icc = (msb - mse) / (msb + (1) * mse + 1e-8)  # ICC(3,1)

    elif icc_type == 'icc2':
        # Slower fallback: loop over features
        def _icc2(x, y):
            data = pd.DataFrame({'r1': x, 'r2': y})
            n, k = data.shape
            mean_subj = data.mean(axis=1)
            mean_rater = data.mean(axis=0)
            grand = data.values.flatten().mean()
            MSB = k * ((mean_subj - grand) ** 2).sum() / (n - 1)
            MSR = n * ((mean_rater - grand) ** 2).sum() / (k - 1)
            MSE = ((data.sub(mean_subj, axis=0)) ** 2).values.sum() / (n * (k - 1))
            return (MSB - MSE) / (MSB + (k - 1) * MSE + k * (MSR - MSE) / n + 1e-8)

        icc = [_icc2(X1[:, i], X2[:, i]) for i in range(D)]
        icc = np.array(icc)

    else:
        raise ValueError("icc_type must be 'icc3' or 'icc2'")

    icc_series = pd.Series(icc, index=df1.columns)
    return icc_series


def select_icc(df1: pd.DataFrame,
               df2: pd.DataFrame,
               icc_type: str = 'icc3',
               threshold: float = 0.75):
    if not df1.columns.equals(df2.columns):
        raise ValueError("两个 DataFrame 的列名必须一致")
    if df1.shape != df2.shape:
        raise ValueError("两个 DataFrame 的形状必须一致")

    icc_feats = compute_icc_features(df1, df2, icc_type)
    return icc_feats[icc_feats >= threshold].index.tolist()


def select_lasso(alphas=np.logspace(-3, 1, 50), cv=10, max_iter=100000):
    def fn(x, y):
        # 检查 x 是否为 DataFrame
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Input x must be a pandas DataFrame with feature names as columns.")

        model_lassoCV = LassoCV(alphas=alphas, cv=cv, max_iter=max_iter).fit(x, y)
        coef = pd.Series(model_lassoCV.coef_, index=x.columns)
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)))
        return coef[coef != 0].index.to_list()

    return fn


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    p_value = stats.norm.sf(z) * 2
    # return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    return p_value[0]


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def delong_pvalue_matrix(y_true, y_scores_dict):
    """
    计算多模型两两 DeLong 检验的 p 值矩阵。

    Parameters
    ----------
    y_true : array-like of shape (n,)
        二分类真值 0/1
    y_scores_dict : dict
        键 = 模型名，值 = 预测概率数组

    Returns
    -------
    names : list of str
        模型名称顺序
    p_matrix : ndarray of shape (k, k)
        p_matrix[i,j] = 两模型 i vs j 的 p 值，i==j 时设为 np.nan
    """
    model_names = list(y_scores_dict.keys())
    n_models = len(model_names)
    p_matrix = np.full((n_models, n_models), np.nan)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            p_val = delong_roc_test(
                np.array(y_true),
                np.array(y_scores_dict[model_names[i]]),
                np.array(y_scores_dict[model_names[j]])
            )
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

    return model_names, p_matrix
