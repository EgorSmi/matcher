import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def pr_auc_macro(
    df: pd.DataFrame,
    target_column: str = "target",
    scores_column: str = "scores",
    prec_level: float = 0.75,
    cat_column: str = "cat3_grouped",
) -> float:

    y_true = df[target_column]
    y_pred = df[scores_column]
    categories = df[cat_column]

    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    # calculate metric for each big category
    for i, category in enumerate(unique_cats):
        # take just a certain category
        cat_idx = np.where(categories == category)[0]
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]

        # if there is no matches in the category then PRAUC=0
        if sum(y_true_cat) == 0:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue

        # get coordinates (x, y) for (recall, precision) of PR-curve
        y, x, _ = precision_recall_curve(y_true_cat, y_pred_cat)

        # reverse the lists so that x's are in ascending order (left to right)
        y = y[::-1]
        x = x[::-1]

        # get indices for x-coordinate (recall) where y-coordinate (precision)
        # is higher than precision level (75% for our task)
        good_idx = np.where(y >= prec_level)[0]

        # if there are more than one such x's (at least one is always there,
        # it's x=0 (recall=0)) we get a grid from x=0, to the rightest x
        # with acceptable precision
        if len(good_idx) > 1:
            gt_prec_level_idx = np.arange(0, good_idx[-1] + 1)
        # if there is only one such x, then we have zeros in the top scores
        # and the curve simply goes down sharply at x=0 and does not rise
        # above the required precision: PRAUC=0
        else:
            pr_aucs.append(0)
            weights.append(counts[i] / len(categories))
            continue

        # calculate category weight anyway
        weights.append(counts[i] / len(categories))
        # calculate PRAUC for all points where the rightest x
        # still has required precision
        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
        except ValueError:
            pr_aucs.append(0)

    return np.average(pr_aucs, weights=weights)


def old_pr_auc_macro(
    df: pd.DataFrame,
    target_column: str = "target",
    scores_column: str = "scores",
    prec_level: float = 0.75,
    cat_column: str = "cat3_grouped"
) -> float:
    categories = df[cat_column]
    y_true = df[target_column]
    y_pred = df[scores_column]

    weights = []
    pr_aucs = []

    unique_cats, counts = np.unique(categories, return_counts=True)

    for i, category in enumerate(unique_cats):
        cat_idx = categories == category
        y_pred_cat = y_pred[cat_idx]
        y_true_cat = y_true[cat_idx]
        y, x, thr = precision_recall_curve(y_true_cat, y_pred_cat)
        gt_prec_level_idx = np.where(y >= prec_level)[0]

        try:
            pr_auc_prec_level = auc(x[gt_prec_level_idx], y[gt_prec_level_idx])
            if not np.isnan(pr_auc_prec_level):
                pr_aucs.append(pr_auc_prec_level)
                weights.append(counts[i] / len(categories))
        except ValueError as err:
            pr_aucs.append(0)
            weights.append(0)
    return np.average(pr_aucs, weights=weights)