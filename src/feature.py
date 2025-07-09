import numpy as np
from scipy.interpolate import CubicSpline

def get_baseline(arr, t_start=6, t_end=10):
    """
    Calculate median across the time window for all observations at once
    """
    assert len(arr.shape) == 2, "get_baseline() takes in 2D numpy arrays only"
    baseline = np.median(arr[:, t_start:t_end], axis=1, keepdims=True)
    
    return baseline


def compute_derivative(arr, t_start=6, t_end=18):
    
    assert len(arr.shape) == 2, "compute_derivative() takes in 2D numpy arrays only"
    
    # Initialize derivative array for only the fitted portion
    arr_deriv = np.zeros((arr.shape))
    
    # Create x values for the spline fitting
    x = np.arange(t_end-t_start)
    
    # Compute derivative for each row
    for i in range(arr.shape[0]):
        # Fit cubic spline to first n_values of the row
        cs = CubicSpline(x, arr[i, t_start:t_end])
        
        # Compute derivative at the fitted points
        arr_deriv[i, t_start:t_end] = cs(x, nu=1)  # nu=1 for first derivative
    
    return arr_deriv


def print_class_statistics(X, y, feature_names=None):
    """
    Print detailed statistics for each class and feature.
    
    Parameters:
    X: feature matrix of shape (n_samples, n_features)
    y: labels
    feature_names: optional list of feature names
    """
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
    
    unique_classes = np.unique(y)
    
    print("="*60)
    print("CLASS STATISTICS")
    print("="*60)
    
    for class_label in unique_classes:
        class_mask = y == class_label
        class_data = X[class_mask]
        
        print(f"\nClass {class_label} (n={np.sum(class_mask)} samples):")
        print("-" * 40)
        
        for i, feature_name in enumerate(feature_names):
            feature_data = class_data[:, i]
            print(f"{feature_name:>12}: mean={np.mean(feature_data):8.3f}, "
                  f"std={np.std(feature_data):7.3f}, "
                  f"min={np.min(feature_data):8.3f}, "
                  f"max={np.max(feature_data):8.3f}")
    
    print("\n" + "="*60)
    print("FEATURE SEPARABILITY ANALYSIS")
    print("="*60)
    
    # Calculate effect sizes (Cohen's d) between classes
    if len(unique_classes) == 2:
        class0_data = X[y == unique_classes[0]]
        class1_data = X[y == unique_classes[1]]
        
        for i, feature_name in enumerate(feature_names):
            mean0 = np.mean(class0_data[:, i])
            mean1 = np.mean(class1_data[:, i])
            std0 = np.std(class0_data[:, i])
            std1 = np.std(class1_data[:, i])
            
            # Cohen's d (effect size)
            pooled_std = np.sqrt((std0**2 + std1**2) / 2)
            cohens_d = (mean1 - mean0) / pooled_std if pooled_std > 0 else 0
            
            separability = "Poor" if abs(cohens_d) < 0.2 else \
                          "Small" if abs(cohens_d) < 0.5 else \
                          "Medium" if abs(cohens_d) < 0.8 else "Large"
            
            print(f"{feature_name:>12}: Cohen's d = {cohens_d:7.3f} ({separability} effect)")