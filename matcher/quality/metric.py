import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def pr_auc_macro(
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
