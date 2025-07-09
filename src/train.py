import numpy as np
import pandas as pd
from scipy import stats

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.costs = []
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.
        
        Parameters:
        X: feature matrix of shape (n_samples, n_features)
        y: binary labels of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)
            
            # Compute cost (log-likelihood)
            cost = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.costs.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.costs[-2] - self.costs[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)
    
    def predict(self, X, thresh=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= thresh).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def sigmoid(z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    
def fit_logistic_regression_grouped_cv(X, y, ani_id, verbose=False):
    """
    Fit logistic regression using Grouped Cross-Validation by ani_id with robust feature standardization.
    
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
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    
    # Apply robust standardization once to all features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    if verbose:
        print("Applied RobustScaler standardization:")
        print(f"  Original feature ranges: min={np.min(X, axis=0)}, max={np.max(X, axis=0)}")
        print(f"  Scaled feature ranges: min={np.min(X_scaled, axis=0)}, max={np.max(X_scaled, axis=0)}")
        print(f"  Scaled feature medians: {np.median(X_scaled, axis=0)}")
    
    ani_id = np.array(ani_id)
    unique_ids = np.unique(ani_id)
    n_folds = len(unique_ids)
    n_samples = X_scaled.shape[0]
    
    # Store predictions and probabilities for each sample
    all_predictions = np.full(n_samples, -1)  # Initialize with -1 to track coverage
    all_probabilities = np.full(n_samples, -1.0)
    fold_results = []
    
    if verbose:
        print(f"\nPerforming Grouped Cross-Validation with {n_folds} folds (unique ani_ids)...")
        print(f"Animal IDs: {unique_ids}")
    
    # Perform grouped CV on scaled features
    for fold, (X_train, X_test, y_train, y_test, test_id) in enumerate(grouped_cv_split(X_scaled, y, ani_id)):
        if verbose:
            print(f"\nFold {fold + 1}/{n_folds}: Testing on ani_id '{test_id}'")
            print(f"  Train set: {X_train.shape[0]} samples")
            print(f"  Test set:  {X_test.shape[0]} samples")
        
        # Check if we have both classes in training set
        unique_train_labels = np.unique(y_train)
        if len(unique_train_labels) < 2:
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
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Extract probabilities for the positive class (class 1)
        y_pred_proba_positive = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]
        
        # Calculate fold metrics
        fold_accuracy = np.mean(y_pred == y_test)
        fold_f1, fold_auc = classification_report(y_test, y_pred, y_pred_proba, verbose=False)
        
        # Store results for this fold
        test_mask = ani_id == test_id
        all_predictions[test_mask] = y_pred
        all_probabilities[test_mask] = y_pred_proba_positive
        
        fold_results.append({
            'fold': fold + 1,
            'test_id': test_id,
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'accuracy': fold_accuracy,
            'f1_score': fold_f1,
            'auc': fold_auc,
            'true_labels': y_test,
            'predictions': y_pred
        })
        
        if verbose:
            print(f"  Fold accuracy: {fold_accuracy:.4f}, F1: {fold_f1:.4f}, AUC: {fold_auc:.4f}")
    
    # Verify all samples were predicted
    unpredicted = np.sum(all_predictions == -1)
    if unpredicted > 0 and verbose:
        print(f"WARNING: {unpredicted} samples were not predicted!")
    
    # Calculate overall CV metrics
    valid_mask = all_predictions != -1
    if np.sum(valid_mask) == 0:
        if verbose:
            print("ERROR: No valid predictions made!")
        return 0.0, 0.0, 0.0, all_predictions, all_probabilities, fold_results
    
    y_valid = y[valid_mask]
    pred_valid = all_predictions[valid_mask].astype(int)
    prob_valid = all_probabilities[valid_mask]
    
    f1_cv, auc_cv = classification_report(y_valid, pred_valid, prob_valid, verbose=verbose) 
    acc_cv = np.mean(pred_valid == y_valid)
    
    if verbose:
        print(f"\nGrouped Cross-Validation Results (with Robust Standardization):")
        print_confusion_matrix(y_valid, pred_valid)
        print(f"\nOverall CV Metrics:")
        print(f"Accuracy: {acc_cv:.4f}")
        print(f"F1 Score (Macro): {f1_cv:.4f}")
        print(f"AUC: {auc_cv:.4f}")

        # Print fold summary
        print(f"\nFold Summary:")
        for result in fold_results:
            if 'warning' in result:
                print(f"  Fold {result['fold']} ({result['test_id']}): {result['warning']}")
            else:
                print(f"  Fold {result['fold']} ({result['test_id']}): acc={result['accuracy']:.3f}, f1={result['f1_score']:.3f}, auc={result['auc']:.3f}")
    
    return acc_cv, f1_cv, auc_cv, all_predictions, all_probabilities, fold_results


def fit_logistic_regression(X, y, test_size=0.2, random_state=42, verbose=False):
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
    print("Model Performance:")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Testing Accuracy: {model.score(X_test, y_test):.4f}")
    print(f"Model Weights: {model.weights}")
    print(f"Model Bias: {model.bias:.4f}")
    
    print("\nClassification Report:")
    macro_f1, auc = classification_report(y_test, y_pred, y_pred_proba, verbose=verbose)
    
    if verbose: 
        print_confusion_matrix(y_test, y_pred)
        print(f"\nOverall F1 Score (Macro): {macro_f1:.4f}")
        if auc is not None:
            print(f"AUC: {auc:.4f}")
    
    # Calculate testing accuracy
    acc_test = model.score(X_test, y_test)
    
    return model, X_train, X_test, y_train, y_test, acc_test, macro_f1, auc