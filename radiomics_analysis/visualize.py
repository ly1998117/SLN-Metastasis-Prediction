import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix


def bootstrap_roc_ci(y_true, y_scores, fpr_grid):
    tprs_interp = []
    for y_score in y_scores:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)

    tprs_interp = np.array(tprs_interp)
    lower = np.percentile(tprs_interp, 2.5, axis=0)
    upper = np.percentile(tprs_interp, 97.5, axis=0)
    return lower, upper


def bootstrap_auc_ci(y_true, y_scores):
    bootstrapped_scores = []

    for y_score in y_scores:
        score = roc_auc_score(y_true, y_score)
        bootstrapped_scores.append(score)

    auc = np.mean(bootstrapped_scores)
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    return auc, lower, upper


def plot_single_roc(y_true, y_score_dict, title='ROC Curve', show_ci=True, show_fill=True, save_path=None):
    """
    Plot ROC curve for multiple models.

    Parameters:
    - y_true: 1D array-like of true binary labels
    - y_score_dict: dict, keys are model names, values are predicted probabilities
    - title: str, plot title
    - save_path: str or None, if given will save plot to this path (.pdf)
    """

    plt.figure(figsize=(6, 5), dpi=300)
    sb.set_theme(style='white')

    for model_name, y_score in y_score_dict.items():
        if show_ci:
            fpr, tpr, _ = roc_curve(y_true, y_score.mean(axis=0))
            auc, ci_low, ci_high = bootstrap_auc_ci(np.array(y_true), y_score)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name}  AUC={auc:.2f}({ci_low:.2f}, {ci_high:.2f})')
            if show_fill:
                fpr_grid = np.linspace(0, 1, 100)
                lower, upper = bootstrap_roc_ci(np.array(y_true), y_score, fpr_grid)
                plt.fill_between(fpr_grid, lower, upper, alpha=0.2)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name}  AUC={auc:.2f}')

    plt.plot([0, 1], [0, 1], '--', color='grey', lw=1.5)
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title(title, fontsize=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(loc='lower right', fontsize=11, frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_calibration_curve(y_true, y_prob_dict, save_path=None):
    plt.figure(figsize=(6, 6), dpi=300)
    sb.set_theme(style='white')

    for name, y_prob in y_prob_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, 'o-', label=f'{name}', linewidth=2.5, color='red')

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration', linewidth=2)

    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Observed Proportion', fontsize=14)
    plt.title('Calibration Curve', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------
# 1. 计算模型 Net Benefit  ------------------------------
# ------------------------------------------------------------------
def calculate_net_benefit_model(thresholds, y_prob, y_true):
    """
    thresholds : (m,) ndarray
    y_prob     : array-like (n,)  — 可以是 list / ndarray / pd.Series
    y_true     : array-like (n,)
    """
    y_prob = np.asarray(y_prob)  # <<< 强制转成 ndarray
    y_true = np.asarray(y_true)  # <<< idem
    thresholds = np.asarray(thresholds)

    n = len(y_true)
    pred_mat = y_prob[:, None] >= thresholds[None, :]  # (n, m) 广播
    tp = np.logical_and(pred_mat, y_true[:, None] == 1).sum(axis=0)
    fp = np.logical_and(pred_mat, y_true[:, None] == 0).sum(axis=0)

    odds = thresholds / (1 - thresholds)
    return tp / n - fp / n * odds


# ------------------------------------------------------------------
# 2. Treat-All Net Benefit （一次即可）
# ------------------------------------------------------------------
def calculate_net_benefit_all(thresholds, y_true):
    y_true = np.asarray(y_true)
    n = len(y_true)
    tp_all = (y_true == 1).sum()  # 全预测 1 时的 TP
    fp_all = (y_true == 0).sum()  # 全预测 1 时的 FP
    odds = thresholds / (1 - thresholds)
    return tp_all / n - fp_all / n * odds


def plot_decision_curve(y_true, y_prob_dict, save_path=None):
    thresholds = np.arange(0, 1, 0.001)  # 避开 0/1
    sb.set_theme(style='white')
    # palette = sb.color_palette("tab10", n_colors=len(y_prob_dict))
    plt.figure(figsize=(7, 5), dpi=300)

    # 计算并绘制 Treat-All、Treat-None
    nb_all = calculate_net_benefit_all(thresholds, y_true)
    plt.plot(thresholds, nb_all, '--', color='gray', lw=2.5, label='Treat All')
    plt.plot(thresholds, np.zeros_like(thresholds), '--', color='black', lw=2.5, label='Treat None')

    # 多模型曲线
    y_min, y_max = 0, 0
    for name, y_prob in y_prob_dict.items():
        nb = calculate_net_benefit_model(thresholds, y_prob, y_true)
        plt.plot(thresholds, nb, lw=2.5, label=name, color='red')

        # 填充模型优于 Treat-All 的区域
        plt.fill_between(thresholds,
                         np.maximum(nb, nb_all), nb_all,
                         where=nb > nb_all,
                         color='crimson', alpha=0.15)

        y_min = min(y_min, nb.min())
        y_max = max(y_max, nb.max())

    # 轴范围与样式
    margin = 0.05 * (y_max - y_min if y_max != y_min else 1)
    plt.ylim(y_min - margin, y_max + margin)
    plt.xlim(0, 1)
    plt.xlabel('Threshold Probability', fontdict={'family': 'Times New Roman', 'fontsize': 15})
    plt.ylabel('Net Benefit', fontdict={'family': 'Times New Roman', 'fontsize': 15})
    plt.title('Decision Curve Analysis', fontsize=16, family='Times New Roman')
    plt.legend(loc='best', fontsize=11, frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_heatmap(names, mat, title='Pairwise DeLong P-values',
                 cmap='Blues_r', save_path=None):
    """
    用 matplotlib 画热图；mat 为下三角/全矩阵均可。
    """
    k = len(names)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    # ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
    # 方格文本
    for i in range(k):
        for j in range(k):
            txt = ''
            if i != j:
                v = mat[i, j]
                if v < 0.001:
                    txt = format(v, '.1e')
                else:
                    txt = format(v, '.3f')
                if v < 0.05:
                    txt += ' *'  # 显著性标记
            ax.text(j, i, txt, ha='center', va='center', fontsize=10, color='white')

    # 坐标轴
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


#################################################################################################
