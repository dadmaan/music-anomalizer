import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import torch
from music_anomalizer.utils import create_folder

class LatentSpaceVisualizer:
    """
    A class for visualizing the latent space of an Autoencoder model using various methods.

    Attributes:
        latent_representations (np.ndarray): The latent representations extracted from DeepSVDD.
        latent_dim (int): The number of dimensions in the latent space.
        model_name (str): The name of the model, used for titling plots.
        anomaly_scores (np.ndarray): The anomaly scores for (train, validation) sets.

    Methods:
        plot_latent_distributions(ax, kde): Plots the distributions of each latent dimension.
        plot_pca_embeddings(ax, labels=None): Performs PCA on the latent representations and plots the first two principal components.
        plot_latent_heatmap(ax): Plots a heatmap of the latent representations.
        plot_scores(ax, kde): Plots the distribution of anomaly scores for train and valid sets.
        visualize_all(): Creates a figure with subplots to visualize all aspects of the latent space side by side.
    """
    def __init__(self, latent_representations, latent_dim, anomaly_scores=None):
        """
        Initializes the LatentSpaceVisualizer with latent representations, dimensionality, and model name.

        Parameters:
            latent_representations (torch.Tensor | np.ndarray): The latent space representations.
            latent_dim (int): The number of dimensions in the latent space.
        """
        self.latent_representations = latent_representations
        self.anomaly_scores = anomaly_scores
        self.latent_dim = latent_dim
        if isinstance(latent_representations, torch.Tensor):
            self.latent_representations = latent_representations.detach().cpu().numpy()
    
    def get_colors_for_anomalies(self, threshold):
        """
        Determines colors for plotting based on anomaly scores exceeding a given threshold.

        This method converts anomaly scores into binary labels and assigns a color to each:
        'blue' for normal data points (below threshold) and 'red' for anomalies (above threshold).

        Parameters:
        - threshold (float): The cutoff value above which a score is considered anomalous.

        Returns:
        - tuple: A tuple containing two elements:
            - colors (list of str): A list of colors corresponding to each data point.
            - labels (numpy.ndarray): A binary array where 1 indicates an anomaly and 0 indicates normal.
        """
        a_scr = np.array(self.anomaly_scores[0] if len(self.anomaly_scores) > 1 else self.anomaly_scores).flatten()
        # if a_scr.ndim == 0:
        #     a_scr = a_scr.reshape(1)
        # Generate binary labels: 1 if the score exceeds the threshold, 0 otherwise
        binary_labels = (a_scr > threshold).astype(int)
        colors = ['blue' if label == 0 else 'red' for label in binary_labels]
        
        return colors, binary_labels
    
    def plot_latent_distributions(self, ax, kde, fontsize=12):
        "Plots histograms of the latent dimensions."
        
        palette = sns.color_palette("hsv", self.latent_dim)
        for i in range(self.latent_dim):
            sns.histplot(self.latent_representations[:, i], kde=kde, color=palette[i], label=f'Dim {i+1}', element='step', stat='density', alpha=0.3, ax=ax)
        ax.set_title(f'Distributions of Latent Space Representations', fontsize=fontsize)
        ax.set_xlabel('Latent Values', fontsize=fontsize)
        ax.set_ylabel('Density', fontsize=fontsize)

    def plot_pca_embeddings(self, ax, threshold=None, fontsize=12):
        "Performs PCA on the latent representations and plots the first two principal components."
        
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(self.latent_representations)
        var_ratio = pca.explained_variance_ratio_
        print(f"Explained variance by component: PC1={var_ratio[0]:.3f}, PC2={var_ratio[1]:.3f}")
        print("Total variance explained by 2 components:", np.sum(var_ratio))
        
        if threshold is not None:
            colors,_ = self.get_colors_for_anomalies(threshold)
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.5)
            legend_elements = [Patch(facecolor='blue', edgecolor='blue', label='Normal'),
                       Patch(facecolor='red', edgecolor='red', label='Anomaly')]
        else:
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
        
        ax.set_xlabel(f'Principal Component 1 (EVR={var_ratio[0]:.3f})', fontsize=fontsize)
        ax.set_ylabel(f'Principal Component 2 (EVR={var_ratio[1]:.3f})', fontsize=fontsize)
        ax.set_title(f'PCA of Latent Space Representations', fontsize=fontsize)
        ax.grid(True)
        if threshold is not None:
            ax.legend(handles=legend_elements, loc='best')

    def plot_latent_heatmap(self, ax, fontsize=12):
        "Plots a heatmap of the latent space representations."
        
        sns.heatmap(self.latent_representations, cmap='viridis', cbar=True, ax=ax)
        ax.set_title(f'Heatmap of Latent Space Representations', fontsize=fontsize)
        ax.set_xlabel('Latent Dimensions', fontsize=fontsize)
        ax.set_ylabel('Samples', fontsize=fontsize)
        
    def plot_scores(self, ax, kde, threshold, fontsize=12):
        "Plots histograms of the anomaly scores distribution."
        
        if len(self.anomaly_scores) > 1:
            train_scores, valid_scores = self.anomaly_scores
            sns.histplot(train_scores, kde=kde, label='Train', element='step', stat='density', alpha=0.3, ax=ax)
            sns.histplot(valid_scores, kde=kde, label='Valid', element='step', stat='density', alpha=0.3, ax=ax)
        else:
            sns.histplot(self.anomaly_scores[0], kde=kde, label=None, element='step', stat='density', alpha=0.3, ax=ax)

        ax.set_yscale('log')
        ax.set_title(f'Distributions of Anomaly Scores', fontsize=fontsize)
        ax.set_xlabel('Anomaly Score', fontsize=fontsize)
        ax.set_ylabel('Frequency', fontsize=fontsize)
        
        if threshold:
            plt.axvline(x=threshold, color='tab:red', linestyle='--', linewidth=3.5, label="Threshold")
            
        ax.legend()
        
    def get_figure(self):
        return self.fig.figure
    
    def visualize_all(self, grid_layout=(1, 4), save_dir=None, threshold=None, kde=False, title=None, fontsize=12, 
                      width_per_col=6, height_per_row=3, plot_anomaly_distribution=False):
        """
        Visualizes various aspects of the latent space in a grid layout.

        Parameters:
            grid_layout (tuple): Tuple specifying the grid layout (rows, cols).
            save_dir (str): Directory to save the visualization. If None, the plot is not saved.
            threshold (float): Threshold for anomaly detection. Used in PCA and anomaly score plots.
            kde (bool): Whether to use Kernel Density Estimation in histograms.
            title (str): Title for the entire visualization.
            fontsize (int): Font size for plot titles and labels.
            width_per_col (int): Width of each column in the grid.
            height_per_row (int): Height of each row in the grid.
            plot_anomaly_distribution (bool): Whether to include the anomaly score distribution plot.
        """
        if not grid_layout or len(grid_layout) != 2:
            raise ValueError("Grid layout must be a tuple of two integers (rows, cols).")

        num_plots = 4 if plot_anomaly_distribution else 3
        if grid_layout[0] * grid_layout[1] < num_plots:
            raise ValueError(f"Grid layout {grid_layout} is too small for {num_plots} plots.")

        # Define the figure size
        fig_size = (width_per_col * grid_layout[1], height_per_row * grid_layout[0])
        self.fig, axs = plt.subplots(grid_layout[0], grid_layout[1], figsize=fig_size)
        axs = axs.flatten()  # Flatten in case of multi-row layout

        # Plot each visualization
        self.plot_latent_distributions(axs[0], kde, fontsize=fontsize - 2)
        self.plot_pca_embeddings(axs[1], threshold, fontsize=fontsize - 2)
        self.plot_latent_heatmap(axs[2], fontsize=fontsize - 2)

        if plot_anomaly_distribution:
            if not self.anomaly_scores or threshold is None:
                raise ValueError("Anomaly scores and threshold must be provided to plot anomaly distribution.")
            self.plot_scores(axs[3], kde, threshold, fontsize=fontsize - 2)

        # Hide unused axes
        for ax in axs[num_plots:]:
            ax.set_visible(False)

        # Set the overall title
        if title:
            self.fig.suptitle(title, fontsize=fontsize)

        plt.tight_layout()

        # Save the figure if a directory is provided
        if save_dir:
            create_folder(save_dir)
            save_path = os.path.join(save_dir, f'{self.model_name}_analysis.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight', pad_inches=0.1)

        plt.show()

        