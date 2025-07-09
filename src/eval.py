import numpy as np

def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])


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

    
def calculate_auc(y_true, y_pred_proba):
    """
    Calculate AUC (Area Under the ROC Curve) from scratch.
    
    Parameters:
    y_true: true binary labels
    y_pred_proba: predicted probabilities for the positive class
    
    Returns:
    auc: Area Under the ROC Curve
    """
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate TPR and FPR at different thresholds
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Random classifier
    
    tp = 0
    fp = 0
    auc = 0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # Add current TP count
    
    # Normalize by total possible area
    auc = auc / (n_pos * n_neg)
    return auc


def classification_report(y_true, y_pred, y_pred_proba=None, verbose=False):
    """Generate classification report with macro F1 score and AUC"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    # Calculate class-specific accuracies
    accuracy_0 = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    accuracy_1 = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate (recall)
    
    # Calculate macro averages
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2
    macro_accuracy = (accuracy_0 + accuracy_1) / 2
    
    # Calculate AUC if probabilities provided
    auc = None
    if y_pred_proba is not None:
        # Use probability of positive class (class 1)
        if len(y_pred_proba.shape) > 1:
            pos_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
        else:
            pos_proba = y_pred_proba
        auc = calculate_auc(y_true, pos_proba)
    
    if verbose:
        print(f"              precision    recall  f1-score  accuracy   support")
        print(f"           0       {precision_0:.2f}      {recall_0:.2f}      {f1_0:.2f}      {accuracy_0:.2f}      {tn+fp}")
        print(f"           1       {precision_1:.2f}      {recall_1:.2f}      {f1_1:.2f}      {accuracy_1:.2f}      {tp+fn}")
        print(f"     overall       {macro_precision:.2f}      {macro_recall:.2f}      {macro_f1:.2f}      {accuracy:.2f}      {len(y_true)}")
        if auc is not None:
            print(f"\nAUC (Area Under ROC Curve): {auc:.4f}")
    
    return macro_f1, auc


def calculate_performance_gains(perf_trace_start, all_trace_starts, group_name='sparse_food'):
    """
    Calculate performance gains for each neuron class.
    
    Parameters:
    perf_trace_start: dict containing performance results
    all_trace_starts: dict with neuron class keys
    group_name: str, name of the group to analyze
    
    Returns:
    dict: neuron_class -> performance_gains array
    """
    perf_gains_by_class = {}
    
    for neuron_class in all_trace_starts.keys():
        # Get fold results for neural and behavioral data
        fold_result_neu = perf_trace_start[group_name][neuron_class]['neu']
        fold_result_beh = perf_trace_start[group_name][neuron_class]['beh']
        
        # Extract AUC values
        beh_auc = []
        for i in range(len(fold_result_beh)):
            beh_auc.append(fold_result_beh[i]['auc'])
            
        neu_auc = []
        for i in range(len(fold_result_neu)):
            neu_auc.append(fold_result_neu[i]['auc'])
        
        # Convert to numpy arrays and calculate performance gain
        beh_auc = np.array(beh_auc)
        neu_auc = np.array(neu_auc)
        perf_gain = neu_auc - beh_auc
        
        perf_gains_by_class[neuron_class] = perf_gain
    
    return perf_gains_by_class