import numpy as np

def flag_nan_in_array(arr, t_start=6, t_end=18):
    """
    Flag instances where the t_start:t_end values contain at least one NaN.
    
    Parameters:
    arr: numpy array of shape (n_observations, n_timepoints) or (1, n_observations, n_timepoints)
    
    Returns:
    flags: boolean array of shape (n_observations, ) where True indicates NaN found
    indices: list of indices where NaN was found
    """
    # Squeeze the array to remove the first dimension if necessary
    if len(arr.shape) == 2:
        data = arr
    elif len(arr.shape) == 3 and arr.shape[0] == 1:
        data = arr.squeeze()
    else:
        raise ValueError(f"Array must have shape (n_observations, n_timepoints) or (1, n_observations, n_timepoints), got {arr.shape}")
    
    # Check for NaN in the first n_values for each instance
    flags = np.isnan(data[:, t_start:t_end]).any(axis=1)
    
    # Get indices where NaN was found
    indices = np.where(~flags)[0].tolist()
    
    return flags, indices


def preprocess_labels(X, y, animal_id=None):
    """
    Remove samples where label == 1 and convert label == 2 to label == 1.
    
    Parameters:
    X: feature matrix of shape (n_samples, n_features)
    y: labels with values 0, 1, 2
    animal_id: optional list of animal IDs, same length as y
    
    Returns:
    X_filtered: filtered feature matrix
    y_filtered: binary labels (0 or 1)
    animal_id_filtered: filtered animal IDs (if provided)
    """
    # Find indices where label != 1
    keep_indices = y != 1
    
    # Filter features and labels
    X_filtered = X[keep_indices]
    y_filtered = y[keep_indices]
    
    # Convert label 2 to label 1
    y_filtered = np.where(y_filtered == 2, 1, y_filtered)
    
    # Filter animal_id if provided
    if animal_id is not None:
        # Convert to numpy array for boolean indexing, then back to list
        animal_id_filtered = animal_id[keep_indices]
    
    print(f"Original data shape: {X.shape}")
    print(f"Filtered data shape: {X_filtered.shape}")
    print(f"Original label distribution: {np.bincount(y)}")
    print(f"Filtered label distribution: {np.bincount(y_filtered)}")
    
    if animal_id is not None:
        print(f"Original animal_id length: {len(animal_id)}")
        print(f"Filtered animal_id length: {len(animal_id_filtered)}")
        
        # Return filtered animal_id as well
        return X_filtered, y_filtered, animal_id_filtered
    else:
        return X_filtered, y_filtered