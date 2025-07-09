import numpy as np

def bootstrap_ecdf_by_animal(animal_data_list, n_bootstrap=100, confidence_level=0.95):
    """
    Bootstrap confidence intervals for ECDF by resampling animals.
    
    Parameters:
    -----------
    animal_data_list : list of arrays
        List where each array contains run durations for one animal
    n_bootstrap : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
    --------
    x_vals : array
        X values for plotting
    ecdf_vals : array
        Original ECDF values
    lower_ci : array
        Lower confidence interval
    upper_ci : array
        Upper confidence interval
    """
    n_animals = len(animal_data_list)
    all_data = np.concatenate(animal_data_list)
    
    # Create x values for evaluation (more points for smoother curves)
    x_vals = np.linspace(0, np.max(all_data), 200)
    
    # Calculate original ECDF
    def ecdf(x_vals, data):
        return np.array([np.mean(data <= x) for x in x_vals])
    
    original_ecdf = ecdf(x_vals, all_data)
    
    # Bootstrap resampling by animal
    bootstrap_ecdfs = []
    
    for _ in range(n_bootstrap):
        # Resample animals with replacement
        bootstrap_animal_indices = np.random.choice(n_animals, size=n_animals, replace=True)
        bootstrap_data = np.concatenate([animal_data_list[i] for i in bootstrap_animal_indices])
        bootstrap_ecdf = ecdf(x_vals, bootstrap_data)
        bootstrap_ecdfs.append(bootstrap_ecdf)
    
    bootstrap_ecdfs = np.array(bootstrap_ecdfs)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_ci = np.percentile(bootstrap_ecdfs, 100 * alpha/2, axis=0)
    upper_ci = np.percentile(bootstrap_ecdfs, 100 * (1 - alpha/2), axis=0)
    
    return x_vals, original_ecdf, bootstrap_ecdfs, lower_ci, upper_ci