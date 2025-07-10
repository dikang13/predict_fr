import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc

def print_confusion_matrix(y_true, y_pred):
    """Print a nicely formatted confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix:")
    print("                 Predicted")
    print("               0     1")
    print(f"Actual    0  {tn:3d}   {fp:3d}")
    print(f"          1  {fn:3d}   {tp:3d}")
    print()
    print("Legend: TN=True Neg, FP=False Pos, FN=False Neg, TP=True Pos")
    print(f"        TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    
def print_metric_summary(name, values):
    median = np.median(values)
    std = np.std(values)
    print(f"{name}: {median:.4f} ± {std:.4f}")
    
    
def eval_metrics(y_true, y_pred, y_pred_proba_positive, verbose=False):
    """Evaluate precision, recall, F1 score, and AUPRC (area under precision-recall curve)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Binary classification metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUROC
    auroc = roc_auc_score(y_true, y_pred_proba_positive)

    # AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba_positive)
    auprc = auc(recall_curve, precision_curve)

    if verbose:
        print_confusion_matrix(y_true, y_pred)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1_score:.4f}")
        print(f"AUROC:     {auroc:.4f}")
        print(f"AUPRC:     {auprc:.4f}")

    return f1_score, auroc, auprc