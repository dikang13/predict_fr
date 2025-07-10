import numpy as np
import pandas as pd
from scipy import stats
from split import *
from evaluate import *
from preprocess import *
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression


def fit_logistic_regression_single_split(X, y, test_size=0.2, random_state=42, verbose=False):
    """
    Fit a logistic regression model to binary classification data.
    
    Parameters:
    X: feature matrix of shape (n_samples, n_features)
    y: binary labels of shape (n_samples,)
    test_size: proportion of data to use for testing
    random_state: random seed for reproducibility
    
    Returns:
    model: fitted LogisticRegression model
    X_train, X_test, y_train, y_test: split datasets
    acc_test: testing accuracy
    f1_test: macro F1 score
    auc_test: AUC score
    """
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create and fit the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Print model performance
    f1_score, auroc, auprc = eval_metrics(y_test, y_pred, y_pred_proba_positive, verbose=verbose)
    
    if verbose: 
        print("Model Performance:")
        print_confusion_matrix(y_test, y_pred)
        print(f"\nF1 Score (Macro): {f1_score:.4f}")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
    
    return (f1_score, auroc, auprc)
    
    
def fit_logistic_regression_grouped_cv(X, y, ani_id, verbose=False):
    """
    Fit logistic regression using Grouped Leave-One-Out Cross-Validation by ani_id with robust feature standardization.
    
    Parameters:
    X: feature matrix of shape (n_samples, n_features)
    y: binary labels of shape (n_samples,)
    ani_id: list of animal IDs, same length as y
    verbose: whether to print detailed output
    
    Returns:
    acc_cv: Cross-validation accuracy
    f1_cv: Cross-validation macro F1 score
    auc_cv: Cross-validation AUC score
    all_predictions: predictions for each sample
    all_probabilities: predicted probabilities for each sample
    fold_results: detailed results for each fold
    """
    
    # Apply robust standardization once to all features
    X_scaled = MaxAbsScaler().fit_transform(X)
    
    if verbose:
        print("Applied MaxAbsScaler standardization:")
        print(f"  Original feature ranges: min={np.min(X, axis=0)}, max={np.max(X, axis=0)}")
        print(f"  Scaled feature ranges: min={np.min(X_scaled, axis=0)}, max={np.max(X_scaled, axis=0)}")
        print(f"  Scaled feature medians: {np.median(X_scaled, axis=0)}")
    
    ani_id = np.array(ani_id)
    unique_ids = np.unique(ani_id)
    n_folds = len(unique_ids)
    n_samples = X_scaled.shape[0]
    
    # Store predictions and probabilities for each sample
    all_predictions   = np.full(n_samples, -1)  # Initialize with -1 to track coverage
    all_probabilities = np.full(n_samples, -1.0)
    fold_results = []
    
    if verbose:
        print(f"\nPerforming Grouped Cross-Validation with {n_folds} folds (unique ani_ids)...")
        print(f"Animal IDs: {unique_ids}")
    
    # Perform grouped LOOCV on scaled features
    splits = grouped_cv_split(X_scaled, y, ani_id)
    
    for fold, (X_train, X_test, y_train, y_test, test_id) in enumerate(splits): # each fold
        if verbose:
            print(f"\nFold {fold + 1}/{n_folds}: Testing on ani_id '{test_id}'")
            print(f"  Train set: {X_train.shape[0]} samples")
            print(f"  Test set:  {X_test.shape[0]} samples")
        
        # Check if we have both classes in training set
        # TODO
        unique_train_labels = np.unique(y_train)
        if len(unique_train_labels) < 2:   # class imbalance might happen per split, but at least both labels should exist
            if verbose:
                print(f"  WARNING: Only one class in training set: {unique_train_labels}")
            # Skip this fold or handle appropriately
            test_mask = ani_id == test_id
            all_predictions[test_mask] = unique_train_labels[0]  # Predict the only class
            all_probabilities[test_mask] = 0.5  # Neutral probability
            fold_results.append({
                'fold': fold + 1,
                'test_id': test_id,
                'n_train': X_train.shape[0],
                'n_test': X_test.shape[0],
                'accuracy': 0.0,
                'f1_score': 0.0,
                'auc': 0.5,
                'warning': 'Single class in training'
            })
            continue
        
        # Train model
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.8,      # Closer to 1.0 means more L1 regularization, closer to 0.0 means more L2 regularization
            C=0.8,             # Inverse regularization strength
            max_iter=1000
        )
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)              # hard binary class prediction at default 0.5 cutoff
        y_pred_proba = model.predict_proba(X_test)  # soft probabilities (same as sigmoid(logits)) for class assignment
        
        # Extract probabilities for the positive class (class 1)
        assert y_pred_proba.shape[1] == 2, "Predicted class probabilities are not in shape (n_samples, n_classes)!!!"
        y_pred_proba_positive = y_pred_proba[:, 1]
        
        # Calculate fold metrics
        # fold_accuracy = np.mean(y_pred == y_test)
        fold_f1, fold_auroc, fold_auprc = eval_metrics(y_test, y_pred, y_pred_proba_positive, verbose=verbose)
        
        # Store results for this fold       
        fold_results.append({
            'fold': fold + 1,
            'test_id': test_id,
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_proba_positive,
            'f1_score': fold_f1,
            'auroc': fold_auroc,
            'auprc': fold_auprc,
        })
        
    # Sanity check after all CV folds are complete: Verify all samples were predicted
    # unpredicted = np.sum(all_predictions == -1)
    # if unpredicted > 0 and verbose:
    #     print(f"WARNING: {unpredicted} samples were not predicted!")
    
    # Extract metrics across folds
    if verbose:
        f1_scores  = [fold['f1_score'] for fold in fold_results]
        aurocs     = [fold['auroc']    for fold in fold_results]
        auprcs     = [fold['auprc']    for fold in fold_results]

        print_metric_summary("F1 Score", f1_scores)
        print_metric_summary("AUROC", aurocs)
        print_metric_summary("AUPRC", auprcs)

    return fold_results


def train_eval(feat, labels, animal_id, verbose=False):
    # Preprocess data by throwing out rows that correspond to the "medium length" fwd runs (10-15% of total)
    feat_filt, labels_filt, animal_id_filt = preprocess_labels(feat, labels, animal_id=animal_id, verbose=verbose)

    # Fit model and gather metrics
    fold_results = fit_logistic_regression_grouped_cv(feat_filt, labels_filt, animal_id_filt, verbose=verbose)

    return fold_results
            

def calc_perf_gain(perf, mdl0='beh', mdl1='comb', metric='auroc', group='sparse_food'):
    """
    Calculate performance gains for each neuron class.
    
    Parameters:
    perf_trace_start: dict containing performance results
    all_trace_starts: dict with neuron class keys
    group_name: str, name of the group to analyze
    
    Returns:
    dict: neuron_class -> performance_gains array
    """
    perf_mdl0 = {}
    perf_mdl1 = {}
    perf_gain = {}
    
    for nc in perf[group].keys():
        # Get fold results for neural and behavioral data
        fold_result_0 = perf[group][nc][mdl0]
        fold_result_1 = perf[group][nc][mdl1]
        
        # Extract AUC values
        metric_0 = []
        for i in range(len(fold_result_0)):
            metric_0.append(fold_result_0[i][metric])
            
        metric_1 = []
        for i in range(len(fold_result_1)):
            metric_1.append(fold_result_1[i][metric])
        
        # Convert to numpy arrays and calculate performance gain
        metric_0 = np.array(metric_0)
        metric_1 = np.array(metric_1)
        
        perf_mdl0[nc] = metric_0
        perf_mdl1[nc] = metric_1       
        perf_gain[nc] = metric_1 - metric_0
    
    return perf_mdl0, perf_mdl1, perf_gain