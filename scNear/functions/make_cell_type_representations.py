import os
import numpy as np
import time as time
import pandas as pd
from scipy.spatial.distance import cdist
from .predict import predict as predict

def generate_representation(data_, 
                            model, 
                            model_path: str, 
                            target_key: str,
                            save_path: str="cell_type_vector_representation", 
                            batch_size: int=32, 
                            device: str=None, 
                            use_multiple_gpus: bool=False,
                            method: str="centroid"):
    """
    Generate representations of cell types based on the provided data using the specified method.

    Parameters
    ----------
    data_ : AnnData
        An AnnData object containing the data.

    model : object
        The trained model for generating predictions.
    
    model_path : str
        Path to the trained model.
    
    target_key : str
        Key for accessing the target labels in `data_.obs`.
    
    save_path : str, optional
        Path to save the representation files (default is "cell_type_vector_representation").
    
    batch_size : int, optional
        Batch size for making predictions (default is 32).
    
    device : str, optional
        Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu" (default is None).
    
    use_multiple_gpus : bool, optional
        If using GPU, whether to use one GPU or all avilable GPUs (default is False).
    
    method : str, optional
        Method for generating representations: "centroid", "median", or "medoid" (default is "centroid").

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the cell type representations.
    """

    # Make predictions
    predictions = predict(data_=data_, 
                          model=model, 
                          model_path=model_path,
                          device=device, 
                          use_multiple_gpus=use_multiple_gpus,
                          batch_size=batch_size)
    
    data_.obsm["predictions"] = predictions

    # Make path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    def CentroidRepresentation(adata_):
        """
        Calculates and returns centroids for each unique label in the dataset.

        Parameters
        ----------
        adata_ : AnnData
            An AnnData object containing the data.

        Returns
        -------
        dict
            A dictionary where each key is a unique label and the corresponding value is the centroid representation.
        """

        unique_labels = np.unique(adata_.obs[target_key])
        centroids = {}  # A dictionary to store centroids for each label.

        for label in unique_labels:
            # Find the indices of data points with the current label.
            label_indices = np.where(adata_.obs[target_key] == label)[0]

            # Extract the latent space representations for data points with the current label.
            latent_space_for_label = adata_.obsm["predictions"][label_indices]

            # Calculate the centroid for the current label cluster.
            centroid = np.mean(latent_space_for_label, axis=0)

            centroids[label] = centroid

        return centroids

    def MedianRepresentation(adata_):
        """
        Calculates and returns median centroids for each unique label in the dataset.

        Parameters
        ----------
        adata_ : AnnData
            An AnnData object containing the data.

        Returns
        -------
        dict
            A dictionary where each key is a unique label and the corresponding value is the median centroid representation.
        """

        unique_labels = np.unique(adata_.obs[target_key])
        median_centroids = {}  # A dictionary to store median centroids for each label.

        for label in unique_labels:
            # Find the indices of data points with the current label.
            label_indices = np.where(adata_.obs[target_key] == label)[0]

            # Extract the latent space representations for data points with the current label.
            latent_space_for_label = adata_.obsm["predictions"][label_indices]

            # Calculate the median for each feature across the data points with the current label.
            median = np.median(latent_space_for_label, axis=0)

            median_centroids[label] = median

        return median_centroids

    def MedoidRepresentation(adata_):
        """
        Calculates and returns medoid centroids for each unique label in the dataset.

        Parameters
        ----------
        adata_ : AnnData
            An AnnData object containing the data.

        Returns
        -------
        dict
            A dictionary where each key is a unique label and the corresponding value is the medoid centroid representation.
        """
        unique_labels = np.unique(adata_.obs[target_key])
        medoid_centroids = {}  # A dictionary to store medoid centroids for each label.

        for label in unique_labels:
            # Find the indices of data points with the current label.
            label_indices = np.where(adata_.obs[target_key] == label)[0]

            # Extract the latent space representations for data points with the current label.
            latent_space_for_label = adata_.obsm["predictions"][label_indices]

            # Calculate pairwise distances between data points in the group.
            distances = cdist(latent_space_for_label, latent_space_for_label, 'euclidean')

            # Find the index of the medoid (index with the smallest sum of distances).
            medoid_index = np.argmin(distances.sum(axis=0))

            # Get the medoid (data point) itself.
            medoid = latent_space_for_label[medoid_index]

            medoid_centroids[label] = medoid

        return medoid_centroids
    
    if method == "centroid":
        representation = CentroidRepresentation(adata_=data_)
    elif method == "median":
        representation = MedianRepresentation(adata_=data_)
    elif method == "medoid":
        representation = MedoidRepresentation(adata_=data_)

    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame(representation)

    # Save the DataFrame as a CSV file
    df.to_csv(f"{save_path}/CellTypeRepresentations.csv", index=False)

    return df
