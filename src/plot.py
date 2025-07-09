import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

def create_box_plot(perf_gains_by_class, figsize=(15, 5), save_path=None):
    """
    Create a box plot for performance gains by neuron class using the defined color scheme.
    
    Parameters:
    perf_gains_by_class: dict with neuron_class -> performance_gains
    figsize: tuple, figure size
    save_path: str, path to save the figure (optional)
    """
    
    # Prepare data for plotting
    data_values = list(perf_gains_by_class.values())
    class_names = list(perf_gains_by_class.keys())
    
    # Get colors for each neuron class
    plot_colors = []
    for class_name in class_names:
        if class_name in colors_neuron_classes:
            plot_colors.append(colors_neuron_classes[class_name])
        else:
            # Default color for unknown classes
            plot_colors.append('#808080')  # Gray
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create box plot
    box_parts = ax.boxplot(data_values, labels=class_names, patch_artist=True, 
                          notch=True, showmeans=True, meanline=True,
                          flierprops=dict(marker='o', markersize=5, alpha=0.7))
    
    # Apply neuron class colors to boxes
    for i, patch in enumerate(box_parts['boxes']):
        patch.set_facecolor(plot_colors[i])
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Customize other box plot elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(box_parts[element], color='grey', alpha=0.8)
    
    # Median lines
    plt.setp(box_parts['medians'], color='black', linewidth=2)
    
    # # Mean lines
    # plt.setp(box_parts['means'], color='blue', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Neuron Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Δ AUC', fontsize=14, fontweight='bold')
    ax.set_title('Performance Gain from Neural Features', fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    # legend_elements = [Line2D([0], [0], color='red', lw=2, label='Median'),
    #                   Line2D([0], [0], color='blue', lw=2, label='Mean')]
    # ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_strip_plot(perf_gains_by_class, figsize=(5, 12), save_path=None, title=None, baseline=0):
    """
    Create an enhanced strip plot with neuron class colors and additional statistical information.
    
    Parameters:
    perf_gains_by_class: dict with neuron_class -> performance_gains
    figsize: tuple, figure size
    save_path: str, path to save the figure (optional)
    """
    
    # Prepare data
    data_for_plot = []
    for neuron_class, gains in perf_gains_by_class.items():
        for gain in gains:
            data_for_plot.append({
                'Neuron Class': neuron_class,
                'Δ AUC': gain
            })
    
    df = pd.DataFrame(data_for_plot)
    
    # Create subplot
    fig, ax2 = plt.subplots(1, 1, figsize=figsize)
    
    class_names = list(perf_gains_by_class.keys())
    medians = [np.median(gains) for gains in perf_gains_by_class.values()]
    stds = [np.std(gains) for gains in perf_gains_by_class.values()]
    
    # Get colors for each neuron class
    plot_colors = []
    for class_name in class_names:
        if class_name in colors_neuron_classes:
            plot_colors.append(colors_neuron_classes[class_name])
        else:
            # Default color for unknown classes
            plot_colors.append('#808080')  # Gray
    
    y_pos = np.arange(len(class_names))
    
    # Create horizontal bar plot with neuron class colors
    bars = ax2.barh(y_pos, medians, xerr=stds, color=plot_colors, alpha=0.8, 
                    capsize=5, edgecolor='black', linewidth=1)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names, fontsize=10)
    ax2.set_ylim(-1, len(class_names))
    ax2.set_xlabel('Median ± SD', fontsize=12, fontweight='bold')
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.axvline(x=baseline, color='black', linestyle='--', alpha=0.5, linewidth=1)
    # ax2.grid(True, alpha=0.3)
    
    # Flip the y-axis to reverse the order of bars
    ax2.invert_yaxis()
    
    # Add value labels on bars
    for i, (median, std) in enumerate(zip(medians, stds)):
        ax2.text(median + std + 0.01, i, f'{median:.2f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    
def visualize_feature_distributions(X, y, feature_names=None):
    """
    Visualize the distribution of features by class.
    
    Parameters:
    X: feature matrix of shape (n_samples, n_features)
    y: labels 
    feature_names: optional list of feature names
    """
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
    
    n_features = X.shape[1]
    unique_classes = np.unique(y)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(8,8))
    fig.suptitle('Feature Distributions by Class', fontsize=16)
    
    # Individual feature histograms
    for i in range(n_features):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        for class_label in unique_classes:
            class_data = X[y == class_label, i]
            ax.hist(class_data, alpha=0.7, label=f'Class {class_label}', 
                   bins=20, density=True)
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feature_names[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Feature correlation/scatter plot for first two features
    if n_features >= 2:
        ax = axes[1, 1]
        for class_label in unique_classes:
            class_mask = y == class_label
            ax.scatter(X[class_mask, 0], X[class_mask, 1], 
                      alpha=0.6, label=f'Class {class_label}', s=50)
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(f'{feature_names[0]} vs {feature_names[1]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
