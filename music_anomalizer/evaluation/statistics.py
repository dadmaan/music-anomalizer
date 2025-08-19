import numpy as np
from scipy import stats
from scipy.special import boxcox, inv_boxcox
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# MARK: TTest
def divide_predictions_and_ttest(predictions):
    """
    Divide SVDD model predictions into 5 groups, perform pairwise t-tests, and plot results.
    
    Args:
    - predictions (list or numpy array): SVDD model predictions.
    
    Returns:
    - Dictionary with t-test results for each pair of groups.
    """
    # Divide predictions into 5 equal groups
    n = len(predictions)
    group_size = n // 5
    groups = [predictions[i*group_size:(i+1)*group_size] for i in range(5)]
    
    # Initialize results dictionary
    results = {}
    
    # Perform pairwise t-tests between each pair of groups
    for i in range(5):
        for j in range(i+1, 5):
            t_stat, p_value = stats.ttest_ind(groups[i], groups[j])
            results[f'Group {i+1} vs Group {j+1}'] = {'t-statistic': t_stat, 'p-value': p_value}
    
    # Plotting
    bins = np.linspace(min(predictions), max(predictions), 30)
    plt.figure(figsize=(12, 8))
    for idx, group in enumerate(groups):
        plt.hist(group, bins, alpha=0.5, label=f'Group {idx+1}')
    
    plt.legend(loc='upper right')
    plt.title('Distribution of SVDD Scores Across Groups')
    plt.xlabel('SVDD Score')
    plt.ylabel('Frequency')
    plt.show()
    
    return results

def perform_pairwise_ttests(group1, group2, group3, title='Distribution of SVDD Scores Across Models'):
    """
    Perform pairwise t-tests between three groups of SVDD results.
    
    Args:
    - group1 (list or numpy array): Results from the base model.
    - group2 (list or numpy array): Results from the proposed model.
    - group3 (list or numpy array): Results from the proposed model with residuals connection.
    
    Returns:
    - A dictionary containing the t-test results for each pair.
    """
    results = {}
    
    # Compare group 1 and group 2
    t_stat, p_value = stats.ttest_ind(group1, group2)
    results['base_vs_proposed'] = {'t-statistic': t_stat, 'p-value': p_value}
    
    # Compare group 1 and group 3
    t_stat, p_value = stats.ttest_ind(group1, group3)
    results['base_vs_residual'] = {'t-statistic': t_stat, 'p-value': p_value}
    
    # Compare group 2 and group 3
    t_stat, p_value = stats.ttest_ind(group2, group3)
    results['proposed_vs_residual'] = {'t-statistic': t_stat, 'p-value': p_value}

    # Plotting
    bins = np.linspace(min(min(group1), min(group2), min(group3)), max(max(group1), max(group2), max(group3)), 30)
    plt.figure(figsize=(10, 6))
    plt.hist(group1, bins, alpha=0.5, label='Base')
    plt.hist(group2, bins, alpha=0.5, label='Proposed')
    plt.hist(group3, bins, alpha=0.5, label='Proposed with Residuals')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('SVDD Score')
    plt.ylabel('Frequency')
    plt.show()
    
    return results


def perform_pairwise_ttests_dynamic(train_results, dataset_name, title='Pairwise T-test of Anomaly Scores Across Models'):
    """
    Perform pairwise t-tests dynamically between all groups in train_results for a given dataset.

    Args:
    - train_results (dict): Dictionary containing training results for all datasets and models.
    - dataset_name (str): Name of the dataset to analyze.
    - title (str): Title for the distribution plot.

    Returns:
    - A dictionary containing the t-test results for each pair of groups.
    """
    from itertools import combinations

    # Extract the groups for the given dataset
    groups = train_results[dataset_name]
    group_names = list(groups.keys())
    group_scores = [np.array(groups[group]) for group in group_names]

    # Perform pairwise t-tests
    results = {}
    for (name1, scores1), (name2, scores2) in combinations(zip(group_names, group_scores), 2):
        t_stat, p_value = stats.ttest_ind(scores1, scores2)
        results[f"{name1}_vs_{name2}"] = {'t-statistic': t_stat, 'p-value': p_value}

    # Plotting
    bins = np.linspace(
        min([score.min() for score in group_scores]),
        max([score.max() for score in group_scores]),
        30
    )
    plt.figure(figsize=(10, 6))
    for name, scores in zip(group_names, group_scores):
        plt.hist(scores, bins, alpha=0.5, label=name)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Set y-axis to log scale
    plt.show()

    return results

# MARK: Threshold
def determine_threshold_by_boxcox(data, lmbda=0):
    """
    Determines a threshold value for a given dataset using the Box-Cox transformation.

    The function applies the Box-Cox transformation to stabilize variance and make the data more normally distributed. 
    It then calculates the threshold as one standard deviation above the mean of the transformed data. 
    Finally, it applies the inverse Box-Cox transformation to convert the threshold back to the original data scale.

    Parameters:
    - data (array-like): The input dataset. Must be a one-dimensional, continuous dataset where all values are positive, as the Box-Cox transformation requires positive values.
    - lmbda (float, optional): The lambda parameter for the Box-Cox transformation. Default is 0, which corresponds to a log transformation.

    Returns:
    - float: The threshold value on the original scale of the data. 
    This threshold can be used for outlier detection or other statistical tests where a cutoff is required.
    """
    boxcox_train_dist = boxcox(data, lmbda)
    box_cox_thres = np.mean(boxcox_train_dist) + np.std(boxcox_train_dist)
    threshold = inv_boxcox(box_cox_thres, lmbda)
    return threshold


def determine_threshold_by_quantile(data, quantile=0.95):
    """
    Determine the threshold for anomaly detection based on a quantile.

    Parameters:
    - data (np.array): The SVDD scores from the validation set.
    - quantile (float): The quantile to set the threshold (e.g., 0.95 for 95th percentile).

    Returns:
    - float: The calculated threshold.
    This threshold can be used for outlier detection or other statistical tests where a cutoff is required.
    """
    threshold = np.quantile(data, quantile)
    return threshold


def determine_threshold_by_std_dev(scores, num_std_dev=3):
    """
    Detects anomalies based on statistical deviation.

    Parameters:
    - scores: ndarray
        An array of anomaly scores or distances from the model.
    - num_std_dev: int or float, optional
        The number of standard deviations to use for defining the threshold. Default is 3.

    Returns:
    - threshold: float
        The calculated threshold for anomaly detection.
    """
    mean = np.mean(scores)
    std_dev = np.std(scores)
    threshold = mean + (num_std_dev * std_dev)
    
    return threshold

# MARK: PCA
def plot_pca_train_test(X_train, X_test, train_dist, val_dist, threshold, title_train='Train Set', title_test='Test Set'):
    """
    Plots PCA projections with anomalies for both training and testing datasets side by side.

    Parameters:
    - X_train (np.array): Training dataset.
    - X_test (np.array): Testing dataset.
    - threshold (float): Threshold value for classifying anomalies.
    """
    # Set up the matplotlib figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot for training data
    plot_pca_with_anomalies(X_train, train_dist, threshold, ax1, title_train)
    
    # Plot for testing data
    plot_pca_with_anomalies(X_test, val_dist, threshold, ax2, title_test)
    
    plt.tight_layout()
    plt.show()
    
def plot_pca_with_anomalies(X, y, threshold, ax, title='PCA Projection with Anomalies'):
    """
    Applies PCA to the data, classifies anomalies, and plots the results with color coding.

    Parameters:
    - X (np.array): The dataset (2D array).
    - y (np.array): The anomaly scores (1D array).
    - threshold (float): Threshold value for classifying anomalies.
    - ax (matplotlib.axes.Axes): The axes on which to plot the graph.
    """
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"PCA explained variance rate: {pca.explained_variance_ratio_}")

    anomaly_labels = create_binary_labels_from_threshold(y, threshold)  # Sum as a dummy score

    colors = ['blue' if label == 0 else 'red' for label in anomaly_labels]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5)

    # for i, label in enumerate(anomaly_labels):
    #     color = 'blue' if label == 0 else 'red'
    #     marker = 'o' if label == 0 else 'x'
    #     ax.scatter(X_pca[i, 0], X_pca[i, 1], color=color, marker=marker, label='Normal' if label == 0 else 'Anomaly')
    
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    ax.grid(True)

def create_binary_labels_from_threshold(anomaly_scores, threshold):
    """
    Creates binary labels for anomaly detection based on a given threshold.

    Parameters:
    - anomaly_scores (list or np.array): An array of anomaly scores.
    - threshold (float): The threshold value above which a data point is considered an anomaly.

    Returns:
    - np.array: An array of binary labels (0 for normal, 1 for anomaly).
    """
    # Ensure the input is a numpy array
    anomaly_scores = np.array(anomaly_scores)
    
    # Generate binary labels: 1 if the score exceeds the threshold, 0 otherwise
    binary_labels = (anomaly_scores > threshold).astype(int)
    
    return binary_labels
