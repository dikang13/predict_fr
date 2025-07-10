import numpy as np
# from scipy.interpolate import CubicSpline
from sklearn.linear_model import TheilSenRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning


def get_baseline(arr, t_start=6, t_end=10):
    """
    Calculate median across the time window for all observations at once
    """
    assert len(arr.shape) == 2, "Expects 2D numpy arrays in shape (n_samples, t_timepoints) only"
    baseline = np.median(arr[:, t_start:t_end], axis=1, keepdims=True)
    
    return baseline   # shape (n_samples, 1)


# def get_slope_cubicSpline(arr, t_start=8, t_end=12):
    
#     assert len(arr.shape) == 2, "Expects 2D numpy arrays in shape (n_samples, t_timepoints) only"
    
#     # Initialize derivative array for only the fitted portion
#     arr_deriv = np.zeros_like(arr)
    
#     # Create x values for the spline fitting
#     x = np.arange(t_start, t_end)
    
#     # Compute derivative for each row
#     for i in range(arr.shape[0]):
#         # Fit cubic spline to first n_values of the row
#         cs = CubicSpline(x, arr[i, t_start:t_end])
        
#         # Compute derivative at the fitted points
#         arr_deriv[i, t_start:t_end] = cs(x, nu=1)  # nu=1 for first derivative
    
#     return arr_deriv


def get_slope_TheilSen(arr, t_start=8, t_end=12):
    """
    Compute robust slope (derivative) from t_start to t_end using Theil-Sen regression.
    Input:
        arr: 2D numpy array of shape (n_rows, n_timepoints)
        t_start, t_end: time range to fit the slope over
    Returns:
        arr_deriv: 2D array of same shape, with slopes filled in t_start:t_end and 0 elsewhere
    """
    assert len(arr.shape) == 2, "Expects a 2D numpy array in shape (n_samples, t_timepoints) only"

    slopes = np.zeros((arr.shape[0], 1))
    x = np.arange(t_start, t_end)

    for i in range(arr.shape[0]):
        y_segment = arr[i, t_start:t_end]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # ignore convergence warnings since similar values is not a problem
            model = TheilSenRegressor().fit(x.reshape(-1, 1), y_segment)
            slopes[i] = model.coef_[0]

    return slopes    # shape (n_samples, 1)


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