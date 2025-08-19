import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Baseline 1: Isolation Forest ---

def isolation_forest(X_train_embeddings, X_eval_embeddings, 
                           n_estimators=100, contamination='auto', 
                           random_state=42, **kwargs):
    """
    Applies Isolation Forest to detect anomalies based on embeddings.

    Args:
        X_train_embeddings (np.ndarray): Embeddings of the training data 
                                         (shape: [n_train_samples, embedding_dim]).
                                         Used to fit the model.
        X_eval_embeddings (np.ndarray): Embeddings of the evaluation data 
                                        (shape: [n_eval_samples, embedding_dim]).
                                        Used to calculate anomaly scores.
        n_estimators (int): The number of base estimators (trees) in the ensemble.
        contamination (float or 'auto'): The expected proportion of outliers 
                                         in the data set. Used when fitting 
                                         to define the threshold offset. 
                                         'auto' typically works well.
        random_state (int): Controls the pseudo-randomness of the selection process.
        **kwargs: Additional keyword arguments passed to IsolationForest constructor.


    Returns:
        np.ndarray: Anomaly scores for X_eval_embeddings. 
                    Higher scores indicate a higher likelihood of being an anomaly.
                    Note: We negate score_samples as it returns higher values 
                    for inliers by default.
    """
    print(f"Applying Isolation Forest...")
    iforest = IsolationForest(n_estimators=n_estimators, 
                              contamination=contamination,
                              random_state=random_state,
                              **kwargs)
    
    # Fit the model on the training embeddings (assumed mostly normal)
    iforest.fit(X_train_embeddings)
    
    # Calculate anomaly scores for the evaluation set
    # score_samples returns the negative mean path length. Higher value = more normal.
    # We negate it so that higher score = more anomalous.
    anomaly_scores_train = -iforest.score_samples(X_train_embeddings)
    anomaly_scores_valid = -iforest.score_samples(X_eval_embeddings)
    
    print(f"Isolation Forest scoring complete. Scores shape (Train; Valid): {anomaly_scores_train.shape}; {anomaly_scores_valid.shape}")
    return anomaly_scores_train, anomaly_scores_valid

# --- Baseline 2: PCA Reconstruction Error ---

def pca_reconstruction_error(X_train_embeddings, X_eval_embeddings, 
                                   n_components=None, variance_threshold=0.95,
                                   standardize=True):
    """
    Applies PCA and uses reconstruction error as the anomaly score.

    Args:
        X_train_embeddings (np.ndarray): Embeddings of the training data 
                                         (shape: [n_train_samples, embedding_dim]).
                                         Used to fit PCA.
        X_eval_embeddings (np.ndarray): Embeddings of the evaluation data 
                                        (shape: [n_eval_samples, embedding_dim]).
                                        Used to calculate anomaly scores.
        n_components (int, float, or None): Number of components to keep.
            - If int: Use that exact number.
            - If float (0 < n < 1): Select components explaining this variance.
            - If None: Use variance_threshold to determine components.
        variance_threshold (float): If n_components is None, select the number
                                     of components needed to explain at least 
                                     this much variance (e.g., 0.95 for 95%).
        standardize (bool): Whether to standardize data before applying PCA.
                            Generally recommended.

    Returns:
        np.ndarray: Reconstruction errors (anomaly scores) for X_eval_embeddings.
                    Higher scores indicate a higher likelihood of being an anomaly.
    """
    print(f"Applying PCA Reconstruction Error...")
    
    _X_train = X_train_embeddings.copy()
    _X_eval = X_eval_embeddings.copy()
    
    if standardize:
        print("Standardizing data...")
        scaler = StandardScaler()
        _X_train = scaler.fit_transform(_X_train)
        _X_eval = scaler.transform(_X_eval) # Use same scaler fitted on train

    # Determine the number of components if not specified directly
    if n_components is None:
        print(f"Determining PCA components for {variance_threshold*100}% variance...")
        pca_full = PCA()
        pca_full.fit(_X_train)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_derived = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Selected {n_components_derived} components based on variance threshold.")
        _n_components = n_components_derived
    elif isinstance(n_components, float) and 0 < n_components < 1:
         print(f"Selecting PCA components to explain {n_components*100}% variance...")
         _n_components = n_components
    else:
        print(f"Using specified n_components = {n_components}")
        _n_components = n_components

    # Fit PCA on training data with the chosen number of components
    pca = PCA(n_components=_n_components)
    print("Fitting PCA model...")
    pca.fit(_X_train)
    
    # Transform train and evaluation data to PCA space
    
    X_train_pca = pca.transform(_X_train)
    X_eval_pca = pca.transform(_X_eval)
    
    # Reconstruct evaluation data back to original space
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    X_eval_reconstructed = pca.inverse_transform(X_eval_pca)
    
    # Calculate reconstruction error (sum of squared errors per sample)
    # Note: if standardized, this error is in the standardized space.
    reconstruction_errors_train = np.sum((_X_train - X_train_reconstructed)**2, axis=1)
    reconstruction_errors_valid = np.sum((_X_eval - X_eval_reconstructed)**2, axis=1)
    
    print(f"PCA Reconstruction Error scoring complete. Scores shape (Train; Valid): {reconstruction_errors_train.shape} and {reconstruction_errors_valid.shape}")
    return reconstruction_errors_train, reconstruction_errors_valid


# # --- Example Usage ---

# if __name__ == "__main__":
#     print("--- Running Baseline Example ---")
    
#     # 1. Generate Dummy Data (Replace with your actual embeddings)
#     embedding_dim = 64  # Example embedding dimension
#     n_train = 1000
#     n_val = 200
    
#     # Simulate normal training data (e.g., tight cluster)
#     X_train_embeddings = np.random.rand(n_train, embedding_dim) * 0.5 + 0.25 
    
#     # Simulate validation data (mostly normal, some potential outliers)
#     X_val_normal = np.random.rand(int(n_val * 0.9), embedding_dim) * 0.5 + 0.25
#     X_val_anomalous = np.random.rand(int(n_val * 0.1), embedding_dim) * 1.5 # Wider spread
#     X_val_embeddings = np.vstack([X_val_normal, X_val_anomalous])
#     np.random.shuffle(X_val_embeddings) # Shuffle validation set
    
#     print(f"Generated dummy data:")
#     print(f"Train embeddings shape: {X_train_embeddings.shape}")
#     print(f"Validation embeddings shape: {X_val_embeddings.shape}")
    
#     # 2. Apply Baselines
#     print("\n--- Isolation Forest ---")
#     if_scores = apply_isolation_forest(X_train_embeddings, X_val_embeddings, random_state=42)
#     print("First 5 IF scores:", if_scores[:5])
    
#     print("\n--- PCA Reconstruction Error ---")
#     pca_scores = apply_pca_reconstruction_error(X_train_embeddings, X_val_embeddings, 
#                                                 variance_threshold=0.95, standardize=True)
#     print("First 5 PCA scores:", pca_scores[:5])
    
#     # 3. (Optional) Visualize Score Distributions
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     sns.histplot(if_scores, kde=True)
#     plt.title("Isolation Forest Anomaly Scores (Validation)")
#     plt.xlabel("Score (Higher = More Anomalous)")
    
#     plt.subplot(1, 2, 2)
#     sns.histplot(pca_scores, kde=True)
#     plt.title("PCA Rec. Error Anomaly Scores (Validation)")
#     plt.xlabel("Score (Higher = More Anomalous)")
    
#     plt.tight_layout()
#     plt.show()

#     print("\n--- Example Finished ---")

