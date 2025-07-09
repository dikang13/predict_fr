import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Simple train-test split function"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def grouped_cv_split(X, y, ani_id):
    """
    Generator function that yields train-test splits for Grouped Cross-Validation.
    All samples from the same ani_id stay together in either train or test.
    
    Parameters:
    X: feature matrix of shape (n_samples, n_features)
    y: labels of shape (n_samples,)
    ani_id: list of animal IDs, same length as y
    
    Yields:
    X_train, X_test, y_train, y_test: for each grouped CV fold
    """
    # Convert to numpy array for easier indexing
    ani_id = np.array(ani_id)
    unique_ids = np.unique(ani_id)
    
    for test_id in unique_ids:
        # Test set: all samples from current ani_id
        test_mask = ani_id == test_id
        train_mask = ~test_mask
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        yield X_train, X_test, y_train, y_test, test_id
        