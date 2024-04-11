import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import random
import json
from tqdm import tqdm
import time as time
import pandas as pd
import copy
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA


class prep_data(data.Dataset):
    """
    A class for preparing and handling data.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing single-cell RNA-seq data.

    target_key : str
        The key in the adata.obs dictionary specifying the target labels.

    save_model_path: str
        Path to model. Creates a subfolder in this path with information needed for predictions.

    HVG : bool, optional
        Whether to use highly variable genes for feature selection (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 2000).
    
    batch_keys : list, optional
        A list of keys for batch labels (default is None).

    model_output_dim : int, optional
        Output dimension from the model to be used.

    for_classification : bool, optional
        Whether to process data if it's for classifier training or latent space training (default is False)

    Methods
    -------
    cell_type_centroid_distances()
        Calculates the centorid distance matrix of PCA transformed scRNA-Seq data space.

    __len__()
        Returns the number of data points in the dataset.

    __getitem(idx)
        Returns a data point, its label, and batch information for a given index.
    """

    def __init__(self, 
                 adata, 
                 unique_targets,
                 target_key: str,
                 save_model_path: str,
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 batch_keys: list=None,
                 model_output_dim: int=100,
                 for_classification: bool=False):
        
        self.adata = adata
        self.unique_targets = unique_targets
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.HVG = HVG
        self.HVGs = HVGs
        self.expression_levels_min = None
        self.expression_levels_max = None
        self.pathway_names = None
        self.feature_means = None
        self.feature_stdevs = None

        # Filter highly variable genes if specified
        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.hvg_genes = self.adata.var_names[self.adata.var["highly_variable"]] # Store the HVG names for making predictions later
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
        else:
            self.hvg_genes = self.adata.var_names
        
        # self.X contains the HVGs expression levels
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        # self.labels contains that target values
        self.labels = self.adata.obs[self.target_key]

        if for_classification:
            # Encode the target information
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.unique_targets)
            self.target_label_encoded = self.label_encoder.transform(self.labels)
            self.onehot_label_encoder = OneHotEncoder()
            self.onehot_label_encoder.fit(self.label_encoder.transform(self.unique_targets).reshape(-1, 1))
            self.target = self.onehot_label_encoder.transform(self.target_label_encoded.reshape(-1, 1))
            self.target = self.target.toarray()
        else:
            # Encode the target information
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.unique_targets)
            self.target = self.label_encoder.transform(self.labels)

        # Calculate the avergae centroid distance between cell type clusters of PCA transformed data
        self.cell_type_centroids_distances_matrix = self.cell_type_centroid_distances(n_components=model_output_dim)

        # Encode the batch effect information for each batch key
        if self.batch_keys is not None:
            self.batch_encoders = {}
            self.encoded_batches = []
            for batch_key in self.batch_keys:
                encoder = LabelEncoder()
                encoded_batch = encoder.fit_transform(self.adata.obs[batch_key])
                self.batch_encoders[batch_key] = encoder
                self.encoded_batches.append(encoded_batch)

            self.encoded_batches = [torch.tensor(batch, dtype=torch.long) for batch in self.encoded_batches]

        # Save information needed to make prediction later on new data
        file_path = f"{save_model_path}/ModelMetadata/"

        # Create folder if it doesn't exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # Save HVG gene names
        torch.save(self.hvg_genes, f"{file_path}hvg_genes.pt")

        # Save target encoders
        if for_classification:
            torch.save(self.label_encoder, f"{file_path}label_encoder.pt")
            torch.save(self.onehot_label_encoder, f"{file_path}onehot_label_encoder.pt")
    
    def cell_type_centroid_distances(self, n_components: int=100):
        """
        Calculate the average centroid distances between different cell types across batch effects using PCA.

        Parameters
        -------
        n_components : int, optional 
            Number of principal components to retain after PCA (default is 100).

        Returns
        -------
            average_distance_df: DataFrame containing the normalized average centroid distances between different cell types.
        """

        # Step 1: Perform PCA on AnnData.X
        pca = PCA(n_components=n_components)
        adata = self.adata.copy()  # Make a copy of the original AnnData object
        adata_pca = pca.fit_transform(adata.X)

        # Step 2: Calculate centroids for each cell type cluster of each batch effect
        centroids = {}
        for batch_effect in adata.obs[self.batch_keys[0]].unique():
            for cell_type in adata.obs['cell_type'].unique():
                mask = (adata.obs[self.batch_keys[0]] == batch_effect) & (adata.obs['cell_type'] == cell_type)
                centroid = np.mean(adata_pca[mask], axis=0)
                centroids[(batch_effect, cell_type)] = centroid

        # Step 3: Calculate the average centroid distance between all batch effects
        average_distance_matrix = np.zeros((len(adata.obs['cell_type'].unique()), len(adata.obs['cell_type'].unique())))
        for i, cell_type_i in enumerate(adata.obs['cell_type'].unique()):
            for j, cell_type_j in enumerate(adata.obs['cell_type'].unique()):
                distances = []
                for batch_effect in adata.obs[self.batch_keys[0]].unique():
                    centroid_i = torch.tensor(centroids[(batch_effect, cell_type_i)], dtype=torch.float32, requires_grad=False)
                    centroid_j = torch.tensor(centroids[(batch_effect, cell_type_j)], dtype=torch.float32, requires_grad=False)
                    try:
                        #distance = euclidean(centroids[(batch_effect, cell_type_i)], centroids[(batch_effect, cell_type_j)])
                        distance = torch.norm(centroid_j - centroid_i, p=2)
                        if not torch.isnan(distance).any():
                            distances.append(distance)
                    except: # Continue if centroids[(batch_effect, cell_type_i)] doesn't exist
                        continue
                average_distance = np.mean(distances)
                average_distance_matrix[i, j] = average_distance

        # Convert average_distance_matrix into a DataFrame
        average_distance_df = pd.DataFrame(average_distance_matrix, index=self.label_encoder.transform(adata.obs['cell_type'].unique()), columns=self.label_encoder.transform(adata.obs['cell_type'].unique()))

        # Replace NaN values with 0
        average_distance_df = average_distance_df.fillna(0)
        # Normalize to get relative distance
        average_distance_df = average_distance_df/average_distance_df.max().max()

        return average_distance_df

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns
        -------
        int
            The number of data points.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Returns a data point, its label, batch information, and selected pathways/gene sets for a given index.

        Parameters
        ----------
        idx : int
            The index of the data point to retrieve.

        Returns
        -------
        tuple
            A tuple containing data point, data label, and batch information (if available).
        """

        # Get HVG expression levels
        data = self.X[idx] 

        # Get labels
        data_label = self.target[idx]

        if self.batch_keys is not None:
            # Get batch effect information
            batches = [encoded_batch[idx] for encoded_batch in self.encoded_batches]
        else:
            batches = torch.tensor([])

        return data, data_label, batches
    

class prep_data_validation(data.Dataset):
    """
    A class for preparing and handling data.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing single-cell RNA-seq data.

    train_env: prep_data()
        prep_data() environment used for training.

    target_key : str
        The key in the adata.obs dictionary specifying the target labels.

    for_classification : bool, optional
        Whether to process data so it's suitable for classification (True) or only produce latent space (False) (defualt is False)

    HVG : bool, optional
        Whether to use highly variable genes for feature selection (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 2000).

    batch_keys : list, optional
        A list of keys for batch labels (default is None).
 
    Methods
    -------
    __len__()
        Returns the number of data points in the dataset.

    __getitem(idx)
        Returns a data point, its label, and batch information for a given index.
    """

    def __init__(self, 
                 adata, 
                 train_env,
                 target_key: str,
                 for_classification: bool=False,
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 batch_keys: list=None):
        
        self.adata = adata
        self.target_key = target_key
        self.batch_keys = batch_keys
        self.HVG = HVG
        self.HVGs = HVGs

        # Filter highly variable genes if specified
        if HVG:
            self.adata = self.adata[:, train_env.hvg_genes].copy()
        
        # self.X contains the HVGs expression levels
        self.X = self.adata.X
        self.X = torch.tensor(self.X)
        # self.labels contains that target values
        self.labels = self.adata.obs[self.target_key]

        # Encode the target information
        if for_classification:
            temp = train_env.label_encoder.transform(self.labels)
            self.target = train_env.onehot_label_encoder.transform(temp.reshape(-1, 1))
            self.target = self.target.toarray()
        else:
            self.target = train_env.label_encoder.transform(self.labels)

        # Encode the batch effect information for each batch key
        if self.batch_keys is not None:
            self.encoded_batches = []
            for batch_key in self.batch_keys:
                encoder = train_env.batch_encoders[batch_key]
                encoded_batch = encoder.transform(self.adata.obs[batch_key])
                self.encoded_batches.append(encoded_batch)

            self.encoded_batches = [torch.tensor(batch, dtype=torch.long) for batch in self.encoded_batches]
    
    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns
        -------
        int
            The number of data points.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Returns a data point, its label, batch information, and selected pathways/gene sets for a given index.

        Parameters
        ----------
        idx : int
            The index of the data point to retrieve.

        Returns
        -------
        tuple
            A tuple containing data point, data label, and batch information (if available).
        """

        # Get HVG expression levels
        data_point = self.X[idx] 

        # Get labels
        data_label = self.target[idx]

        if self.batch_keys is not None:
            # Get batch effect information
            batches = [encoded_batch[idx] for encoded_batch in self.encoded_batches]
        else:
            batches = torch.tensor([])

        return data_point, data_label, batches


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    From: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
        

class EarlyStopping():
    """
    Early Stopping Callback for Training

    This class is a callback for early stopping during training based on validation loss. It monitors the validation loss and stops training if the loss does not improve for a certain number of consecutive epochs.

    Parameters
    -------
        tolerance (int, optional): Number of evaluations to wait for an improvement in validation loss before stopping. Default is 10.
    """
    
    def __init__(self, tolerance: int=10):

        self.tolerance = tolerance
        self.min_val = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss >= self.min_val:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.min_val = val_loss
            self.counter = 0
    

class CustomSNNLoss(nn.Module):
    """
    A Custom loss function, utilizing contrastive learning through the Soft Nearest Neighbor Loss and cell type centroid loss.

    Parameters
    ----------
    use_target_weights : bool, optional
        If True, calculate target weights based on label frequency (default is True).
    
    use_batch_weights : bool, optional
        If True, calculate class weights for specified batch effects based on label frequency (default is True).   
    
    targets : Tensor, optional
        A tensor containing the class labels for the input vectors. Required if use_target_weights is True.
    
    batches : Tensor, optional
        A list of tensors containing the batch effect labels. Required if use_batch_weights is True.
    
    batch_keys : list, optional
        A list containing batch keys to account for batch effects (default is None).
    
    temperature : float, optional
        Initial scaling factor applied to the cosine similarity of the target contribution to the loss (default is 0.25).
   
    min_temperature : float, optional
        The minimum temperature value allowed during optimization (default is 0.1).
    
    max_temperature : float, optional
        The maximum temperature value allowed during optimization (default is 1.0).
    
    device : str, optional
        Which device to be used (default is "cuda").
    """

    def __init__(self, 
                 cell_type_centroids_distances_matrix,
                 use_target_weights: bool=True, 
                 use_batch_weights: bool=True, 
                 targets: torch.tensor=None, 
                 batches: list=None,
                 batch_keys: list=None, 
                 temperature: float=0.25, 
                 min_temperature: float=0.1,
                 max_temperature: float=1.0,
                 device: str="cuda"):
        super(CustomSNNLoss, self).__init__()
        
        # Define temperature variables to be optimized durring training
        self.temperature_target = nn.Parameter(torch.tensor(temperature), requires_grad=True) 
        if batch_keys is not None:
            self.temperatures_batches = []
            for _ in range(len(batch_keys)):
                temperature = 0.5 # Set the temperature term for the batch effect contribution to be 0.5
                self.temperatures_batches.append(temperature)

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.device = device
        self.use_target_weights = use_target_weights
        self.use_batch_weights = use_batch_weights
        self.batch_keys = batch_keys
        self.cell_type_centroids_distances_matrix = cell_type_centroids_distances_matrix

        # Calculate weights for the loss based on label frequency
        if self.use_target_weights:
            if targets is not None:
                self.weight_target = self.calculate_class_weights(targets)
            else:
                raise ValueError("'use_target_weights' is True, but 'targets' is not provided.")
        if self.use_batch_weights: 
            if batch_keys is not None:
                self.weight_batch = []
                for i in range(len(batch_keys)):
                    self.weight_batch.append(self.calculate_class_weights(batches[i]))
            else:
                raise ValueError("'use_weights' is True, but 'batch_keys' is not provided.")

    def calculate_class_weights(self, targets):
        """
        Calculate class weights based on label frequency.

        Parameters
        ----------
        targets : Tensor
            A tensor containing the class labels.
        """

        class_counts = torch.bincount(targets)  # Count the occurrences of each class
        class_weights = 1.0 / class_counts.float()  # Calculate inverse class frequencies
        class_weights /= class_weights.sum()  # Normalize to sum to 1

        class_weight_dict = {class_label: weight for class_label, weight in enumerate(class_weights)}

        return class_weight_dict
    
    def cell_type_centroid_distances(self, X, cell_type_vector):
        """
        Calculate the Mean Squared Error (MSE) loss between target centroids and current centroids based on cell type information.

        Parameters:
        X : torch.tensor
            Input data matrix with each row representing a data point and each column representing a feature.

        cell_type_vector : torch.tensor
            A vector containing the cell type annotations for each data point in X.

        Returns:
            loss: The MSE loss between target centroids and current centroids.
        """

        # Step 1: Calculate centroids for each cell type cluster 
        centroids = {}
        for cell_type in cell_type_vector.unique():
            mask = (cell_type_vector == cell_type)
            centroid = torch.mean(X[mask], axis=0)
            centroids[cell_type.item()] = centroid

        # Step 2: Calculate the average centroid distance between all cell types
        average_distance_matrix_input = torch.zeros((len(cell_type_vector.unique()), len(cell_type_vector.unique())))
        for i, cell_type_i in enumerate(cell_type_vector.unique()):
            for j, cell_type_j in enumerate(cell_type_vector.unique()):
                centroid_i = centroids[cell_type_i.item()]
                centroid_j = centroids[cell_type_j.item()]
                average_distance = torch.norm(centroid_j - centroid_i, p=2)
                average_distance_matrix_input[i, j] = average_distance

        # Replace values with 0 if they were 0 in the PCA centorid matrix
        cell_type_centroids_distances_matrix_filter = self.cell_type_centroids_distances_matrix.loc[cell_type_vector.unique().tolist(),cell_type_vector.unique().tolist()]
        mask = (cell_type_centroids_distances_matrix_filter != 0.0)
        mask = torch.tensor(mask.values, dtype=torch.float32)
        average_distance_matrix_input = torch.mul(mask, average_distance_matrix_input)
        average_distance_matrix_input = average_distance_matrix_input / torch.max(average_distance_matrix_input)

        cell_type_centroids_distances_matrix_filter = torch.tensor(cell_type_centroids_distances_matrix_filter.values, dtype=torch.float32)

        # Only use non-zero elemnts for loss calculation
        non_zero_mask = cell_type_centroids_distances_matrix_filter != 0
        average_distance_matrix_input = average_distance_matrix_input[non_zero_mask]
        cell_type_centroids_distances_matrix_filter = cell_type_centroids_distances_matrix_filter[non_zero_mask]
        cell_type_centroids_distances_matrix_filter = cell_type_centroids_distances_matrix_filter / torch.max(cell_type_centroids_distances_matrix_filter)

        # Step 3: Calculate the MSE between target centroids and current centroids
        # Set to zero if loss can't be calculated, like if there's only one cell type per batch effect element for all elements
        loss = 0
        try:
            loss = F.mse_loss(average_distance_matrix_input, cell_type_centroids_distances_matrix_filter)
        except:
            loss = 0
            pass

        return loss

    def forward(self, input, targets, batches=None):
        """
        Compute the SNN loss and cell type centroid loss for the input vectors and targets.

        Parameters
        ----------
        input : Tensor
            Input vectors (predicted latent space).

        targets : Tensor
            Class labels for the input vectors.

        batches : list, optional
            List of batch keys to account for batch effects.

        Returns
        -------
        loss : Tensor
            The calculated SNN loss + cell type centroid loss.
        """

        ### Target loss

        # Restrict the temperature term
        if self.temperature_target.item() <= self.min_temperature:
            self.temperature_target.data = torch.tensor(self.min_temperature)
        elif self.temperature_target.item() >= self.max_temperature:
            self.temperature_target.data = torch.tensor(self.max_temperature)

        # Calculate the cosine similarity matrix, and also apply exp()
        cosine_similarity_matrix = torch.exp(F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2) / self.temperature_target)

        # Define a loss dictionary containing the loss of each label
        loss_dict = {str(target): torch.tensor([]).to(self.device) for target in targets.unique()}
        for idx, (sim_vec, target) in enumerate(zip(cosine_similarity_matrix,targets)):
            positiv_samples = sim_vec[(targets == target)]
            negativ_samples = sim_vec[(targets != target)]
            # Must be more or equal to 2 samples per sample type for the loss to work since we dont count
            # the diagonal values, hence "- sim_vec[idx]". We don't count this since we are not interested
            # in the cosine similarit of a vector to itself
            if len(positiv_samples) >= 2 and len(negativ_samples) >= 1:
                positiv_sum = torch.sum(positiv_samples) - sim_vec[idx]
                negativ_sum = torch.sum(negativ_samples)
                loss = -torch.log(positiv_sum / (positiv_sum + negativ_sum))
                loss_dict[str(target)] = torch.cat((loss_dict[str(target)], loss.unsqueeze(0)))

        del cosine_similarity_matrix

        # Calculate the weighted average loss of each cell type
        weighted_losses = []
        for target in targets.unique():
            losses_for_target = loss_dict[str(target)]
            # Make sure there's values in losses_for_target of given target
            if (len(losses_for_target) > 0) and (torch.any(torch.isnan(losses_for_target))==False):
                if self.use_target_weights:
                    weighted_loss = torch.mean(losses_for_target) * self.weight_target[int(target)]
                else:
                    weighted_loss = torch.mean(losses_for_target)

                weighted_losses.append(weighted_loss)

        # Calculate the sum loss accross cell types
        loss_target = torch.sum(torch.stack(weighted_losses))

        ### Minimize difference between PCA cell type centorid of data and centroids of cell types in latent space
        loss_centorid = self.cell_type_centroid_distances(input, targets)

        ### Batch effect loss

        if batches is not None:

            loss_batches = []
            for outer_idx, batch in enumerate(batches):

                # Calculate the cosine similarity matrix, and also apply exp()
                cosine_similarity_matrix = torch.exp(F.cosine_similarity(input.unsqueeze(1), input.unsqueeze(0), dim=2) / self.temperatures_batches[outer_idx])

                # Define a loss dictionary containing the loss of each label
                loss_dict = {str(target_batch): torch.tensor([]).to(self.device) for target_batch in batch.unique()}
                for idx, (sim_vec, target_batch, target) in enumerate(zip(cosine_similarity_matrix,batch,targets)):
                    positiv_samples = sim_vec[(targets == target) & (batch == target_batch)]
                    negativ_samples = sim_vec[(targets == target) & (batch != target_batch)]
                    # Must be more or equal to 2 samples per sample type for the loss to work since we dont count
                    # the diagonal values, hence "- sim_vec[idx]". We don't count this since we are not interested
                    # in the cosine similarit of a vector to itself
                    if len(positiv_samples) >= 2 and len(negativ_samples) >= 1:
                        positiv_sum = torch.sum(positiv_samples) - sim_vec[idx]
                        negativ_sum = torch.sum(negativ_samples)
                        loss = (-torch.log(positiv_sum / (positiv_sum + negativ_sum)))**-1
                        loss_dict[str(target_batch)] = torch.cat((loss_dict[str(target_batch)], loss.unsqueeze(0)))

                # Calculate the weighted average loss of each batch effect
                losses = []
                for batch_target in batch.unique():
                    losses_for_target = loss_dict[str(batch_target)]
                    # Make sure there's values in losses_for_target of given batch effect
                    if (len(losses_for_target) > 0) and (torch.any(torch.isnan(losses_for_target))==False):
                        if self.use_batch_weights:
                            temp_loss = torch.mean(losses_for_target) * self.weight_batch[outer_idx][int(batch_target)]
                        else:
                            temp_loss = torch.mean(losses_for_target)
                        losses.append(temp_loss)

                # Only use loss if it was possible to caluclate it from previous steps
                if losses != []:
                    loss_ = torch.sum(torch.stack(losses))
                    loss_batches.append(loss_)

                del cosine_similarity_matrix

            if loss_batches != []:
                loss_batch = torch.mean(torch.stack(loss_batches, dim=0))
            else:
                loss_batch = torch.tensor([0.0]).to(self.device)

            # Apply weights to the three loss contributions
            loss = 0.9*loss_target + 0.1*loss_batch + 1.0*loss_centorid 

            return loss
        else:
            return loss_target + loss_centorid
    

class train_module():
    """
    A class for training the machine learning model using single-cell RNA sequencing data as input and/or pathway/gene set information.

    Parameters
    ----------
    data_path : str or AnnData
        Path to the data file or an AnnData object containing single-cell RNA sequencing data. If a path is provided,
        the data will be loaded from the specified file. If an AnnData object is provided, it will be used directly.

    save_model_path : str
        The path to save the trained model.

    HVG : bool, optional
        Whether to identify highly variable genes (HVGs) in the data (default is True).
    
    HVGs : int, optional
        The number of highly variable genes to select (default is 2000).
    
    target_key : str, optional
        The metadata key specifying the target variable (default is "cell_type").
    
    batch_keys : list, optional
        List of batch keys to account for batch effects (default is None).
    
    validation_pct : float, optional
        What percentage of data to use for validation (defualt is 0.2, meaning 20%).
    """

    def __init__(self, 
                 data_path, 
                 save_model_path: str,
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 target_key: str="cell_type", 
                 batch_keys: list=None,
                 validation_pct: float=0.2):
        
        if type(data_path) == str:
            self.adata = sc.read(data_path, cache=True)
        else:
            self.adata = data_path

        self.HVG = HVG
        self.HVGs = HVGs
        self.target_key = target_key
        self.batch_keys = batch_keys

        # Specify the number of folds (splits)
        if (validation_pct > 0.0) and (validation_pct < 1.0):
            n_splits = int(100/(validation_pct*100))  
        elif validation_pct < 0.0:
            raise ValueError('Invalid choice of validation_pct. Needs to be 0.0 <= validation_pct < 1.0')

        unique_targets = np.unique(self.adata.obs[self.target_key])

        if validation_pct == 0.0:
            self.adata_train = self.adata.copy()
            self.adata_validation = self.adata.copy()
        else:
            # Initialize Stratified K-Fold
            stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Iterate through the folds
            self.adata_train = self.adata.copy()
            self.adata_validation = self.adata.copy()
            for train_index, val_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.target_key]):
                # Filter validation indices based on labels present in the training data
                unique_train_labels = np.unique(self.adata.obs[self.target_key].iloc[train_index])
                filtered_val_index = [idx for idx in val_index if self.adata.obs[self.target_key].iloc[idx] in unique_train_labels]

                self.adata_train = self.adata_train[train_index, :].copy()
                self.adata_validation = self.adata_validation[filtered_val_index, :].copy()
                break

        self.data_env = prep_data(adata=self.adata_train, 
                                  unique_targets=unique_targets,
                                  HVG=HVG,  
                                  HVGs=HVGs, 
                                  target_key=target_key, 
                                  batch_keys=batch_keys,
                                  save_model_path=save_model_path)
        
        self.data_env_validation = prep_data_validation(adata=self.adata_validation, 
                                                        train_env = self.data_env,
                                                        HVG=HVG,  
                                                        HVGs=HVGs, 
                                                        target_key=target_key, 
                                                        batch_keys=batch_keys)
        
        self.data_env_for_classification = prep_data(adata=self.adata_train, 
                                                    unique_targets=unique_targets,
                                                    HVG=HVG,  
                                                    HVGs=HVGs, 
                                                    target_key=target_key, 
                                                    batch_keys=batch_keys,
                                                    save_model_path=save_model_path,
                                                    for_classification=True)
        
        self.data_env_validation_for_classification = prep_data_validation(adata=self.adata_validation, 
                                                        train_env = self.data_env_for_classification,
                                                        HVG=HVG,  
                                                        HVGs=HVGs, 
                                                        target_key=target_key, 
                                                        batch_keys=batch_keys,
                                                        for_classification=True)
        

        self.save_model_path = save_model_path

    
    def train_model(self,
                    model, 
                    optimizer, 
                    lr_scheduler, 
                    loss_module, 
                    device, 
                    out_path, 
                    train_loader, 
                    val_loader, 
                    num_epochs, 
                    eval_freq,
                    earlystopping_threshold,
                    use_classifier,
                    only_print_best: bool=False,
                    accum_grad: int=1,
                    model_classifier: nn.Module=None):
        """
        Don't use this function by itself! It's aimed to be used in the train() and train_classifier() functions.
        """

        print()
        print(f"Start Training")
        print()

        # Add model to device
        model.to(device)
        if use_classifier:
            model_classifier.to(device)

        # Initiate EarlyStopping
        early_stopping = EarlyStopping(earlystopping_threshold)

        # Training loop
        best_val_loss = np.inf  
        best_epoch = 0
        train_start = time.time()

        try:
            for epoch in tqdm(range(num_epochs)):

                # Training
                if use_classifier:
                    model.eval()
                    model_classifier.train()
                else:
                    model.train()
                train_loss = []
                all_preds_train = []
                all_labels_train = []
                batch_idx = -1
                for data_inputs, data_labels, data_batches in train_loader:
                    batch_idx += 1

                    data_labels = data_labels.to(device)
                    data_inputs_step = data_inputs.to(device)

                    preds = model(data_inputs_step)

                    if use_classifier:
                        preds_latent = preds.cpu().detach().to(device)
                        preds = model_classifier(preds_latent)
                    
                    # Whether to use classifier loss or latent space creation loss
                    loss = torch.tensor([]).to(device)
                    try:
                        if use_classifier:
                            loss = loss_module(preds, data_labels)/accum_grad
                        else:
                            if self.batch_keys is not None:
                                data_batches = [batch.to(device) for batch in data_batches]
                                loss = loss_module(preds, data_labels, data_batches)/accum_grad
                            else:
                                loss = loss_module(preds, data_labels)/accum_grad
                    except:
                        # If loss can't be calculated for current mini-batch it continues to the next mini-batch
                        # Can happen if a mini-batch only contains one cell type
                        continue

                    loss.backward()

                    train_loss.append(loss.item())

                    # Perform updates to model weights
                    if ((batch_idx + 1) % accum_grad == 0) or (batch_idx + 1 == len(train_loader)):
                        optimizer.step()
                        optimizer.zero_grad()

                    #optimizer.step()
                    #optimizer.zero_grad()

                    all_preds_train.extend(preds.cpu().detach().numpy())
                    all_labels_train.extend(data_labels.cpu().detach().numpy())

                # Validation
                if (epoch % eval_freq == 0) or (epoch == (num_epochs-1)):
                    model.eval()
                    if use_classifier:
                        model_classifier.eval()
                    val_loss = []
                    all_preds = []
                    all_labels = []
                    with torch.no_grad():
                        for data_inputs, data_labels, data_batches in val_loader:

                            data_inputs_step = data_inputs.to(device)
                            data_labels_step = data_labels.to(device)

                            preds = model(data_inputs_step)

                            if use_classifier:
                                preds_latent = preds.cpu().detach().to(device)
                                preds = model_classifier(preds_latent)

                            # Check and fix the number of dimensions
                            if preds.dim() == 1:
                                preds = preds.unsqueeze(0)  # Add a dimension along axis 0

                            # Whether to use classifier loss or latent space creation loss
                            loss = torch.tensor([]).to(device)
                            try:
                                if use_classifier:
                                    loss = loss_module(preds, data_labels_step)/accum_grad
                                else:
                                    if self.batch_keys is not None:
                                        data_batches = [batch.to(device) for batch in data_batches]
                                        loss = loss_module(preds, data_labels_step, data_batches) /accum_grad
                                    else:
                                        loss = loss_module(preds, data_labels_step)/accum_grad
                            except:
                                # If loss can't be calculated for current mini-batch it continues to the next mini-batch
                                # Can happen if a mini-batch only contains one cell type
                                continue

                            val_loss.append(loss.item())
                            all_preds.extend(preds.cpu().detach().numpy())
                            all_labels.extend(data_labels_step.cpu().detach().numpy())

                    # Metrics
                    avg_train_loss = sum(train_loss) / len(train_loss)
                    avg_val_loss = sum(val_loss) / len(val_loss)
                    #avg_val_loss = avg_train_loss

                    # Check early stopping
                    early_stopping(avg_val_loss)

                    # Print epoch information
                    if use_classifier:

                        binary_preds_train = []
                        # Loop through the predictions
                        for pred in all_preds_train:
                            # Apply thresholding
                            binary_pred = np.argmax(pred)

                            binary_preds_train.append(binary_pred)

                        # Convert the list of arrays to a numpy array
                        binary_preds_train = np.array(binary_preds_train)

                        binary_labels_train = []
                        for pred in all_labels_train:
                            binary_pred = np.argmax(pred)

                            binary_labels_train.append(binary_pred)

                        binary_labels_train = np.array(binary_labels_train)

                        binary_preds_valid = []
                        for label in all_preds:
                            binary_pred = np.argmax(label)

                            binary_preds_valid.append(binary_pred)

                        binary_preds_valid = np.array(binary_preds_valid)

                        binary_labels_valid = []
                        for label in all_labels:
                            binary_pred = np.argmax(label)

                            binary_labels_valid.append(binary_pred)

                        binary_labels_valid = np.array(binary_labels_valid)

                        # Calculate accuracy
                        accuracy_train = accuracy_score(binary_labels_train, binary_preds_train)
                        accuracy = accuracy_score(binary_labels_valid, binary_preds_valid)

                        if only_print_best == False:
                            print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f} | Training Accuracy: {accuracy_train} | Validation loss: {avg_val_loss:.4f} | Validation Accuracy: {accuracy}")
                    else:
                        if only_print_best == False:
                            print(f"Epoch {epoch+1} | Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")

                    # Apply early stopping
                    if early_stopping.early_stop:
                        print(f"Stopped training using EarlyStopping at epoch {epoch+1}")
                        break

                    # Save model if performance has improved
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_epoch = epoch + 1

                        if use_classifier:
                            # Move the model to CPU before saving
                            model_classifier.to('cpu')
                            
                            # Save the entire model to a file
                            torch.save(model_classifier.module.state_dict() if hasattr(model_classifier, 'module') else model_classifier.state_dict(), f'{out_path}model_classifier.pt')
                            
                            # Move the model back to the original device
                            model_classifier.to(device)
                        else:
                            # Move the model to CPU before saving
                            model.to('cpu')
                            
                            # Save the entire model to a file
                            #torch.save(model, f'{out_path}model.pt')
                            #torch.save(model.state_dict(), f'{out_path}model.pt')
                            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), f'{out_path}model.pt')
                            
                            # Move the model back to the original device
                            model.to(device)

                # Update learning rate
                lr_scheduler.step()
        except:
            print(f"**Training forced to finish early due to error during training**")

        print()
        print(f"**Finished training**")
        print()
        print(f"Best validation loss (reached after {best_epoch} epochs): {best_val_loss}")
        print()
        train_end = time.time()
        print(f"Training time: {(train_end - train_start)/60:.2f} minutes")

        return best_val_loss
    
    def train(self, 
                 model: nn.Module,
                 device: str=None,
                 seed: int=42,
                 batch_size: int=256,
                 use_multiple_gpus: bool=False,
                 use_target_weights: bool=True,
                 use_batch_weights: bool=True,
                 init_temperature: float=0.25,
                 min_temperature: float=0.1,
                 max_temperature: float=1.0,
                 init_lr: float=0.001,
                 lr_scheduler_warmup: int=4,
                 lr_scheduler_maxiters: int=50,
                 eval_freq: int=1,
                 epochs: int=50,
                 earlystopping_threshold: int=20,
                 accum_grad: int=1):
        """
        Perform training of the machine learning model for making an embedding space.

        Parameters
        ----------
        model : nn.Module
            The model to train.
        
        device : str or None, optional
            The device to run the training on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.
        
        seed : int, optional
            Random seed for ensuring reproducibility (default is 42).
        
        batch_size : int, optional
            Batch size for data loading during training (default is 256).

        use_target_weights : bool, optional
            If True, calculate target weights based on label frequency (default is True).
        
        use_batch_weights : bool, optional
            If True, calculate class weights for specified batch effects based on label frequency (default is True).   
        
        init_temperature : float, optional
            Initial temperature for the loss function (default is 0.25).
        
        min_temperature : float, optional
            The minimum temperature value allowed during optimization (default is 0.1).
        
        max_temperature : float, optional
            The maximum temperature value allowed during optimization (default is 1.0).
        
        init_lr : float, optional
            Initial learning rate for the optimizer (default is 0.001).
        
        lr_scheduler_warmup : int, optional
            Number of warm-up iterations for the cosine learning rate scheduler (default is 4).
        
        lr_scheduler_maxiters : int, optional
            Maximum number of iterations for the cosine learning rate scheduler (default is 25).
        
        eval_freq : int, optional
            Rate at which the model is evaluated on validation data (default is 2).
        
        epochs : int, optional
            Number of training epochs (default is 20).
        
        earlystopping_threshold : int, optional
            Early stopping threshold (default is 10).

        accum_grad : int, optional
            How many mini-batches the gradient should be accumulated for before updating weights (default is 1).

        Returns
        -------
        None
        """

        model_step_1 = copy.deepcopy(model)

        out_path = self.save_model_path

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Ensure reproducibility
        def rep_seed(seed):
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        rep_seed(seed)

        total_train_start = time.time()

        train_loader = data.DataLoader(self.data_env, batch_size=batch_size, shuffle=True, drop_last=True)
        #val_loader = data.DataLoader(self.data_env, batch_size=batch_size, shuffle=False, drop_last=True)
        min_batch_size = int(np.min([self.data_env_validation.X.shape[0], batch_size]))
        val_loader = data.DataLoader(self.data_env_validation, batch_size=min_batch_size, shuffle=False, drop_last=True)

        total_params = sum(p.numel() for p in model_step_1.parameters())
        print(f"Number of parameters: {total_params}")

        # Define custom SNN loss
        loss_module = CustomSNNLoss(cell_type_centroids_distances_matrix=self.data_env.cell_type_centroids_distances_matrix,
                                    use_target_weights=use_target_weights, 
                                    use_batch_weights=use_batch_weights, 
                                    targets=torch.tensor(self.data_env.target), 
                                    batches=self.data_env.encoded_batches, 
                                    batch_keys=self.batch_keys, 
                                    temperature=init_temperature, 
                                    min_temperature=min_temperature, 
                                    max_temperature=max_temperature)
        
        # Define Adam optimer
        optimizer = optim.Adam([{'params': model_step_1.parameters(), 'lr': init_lr}, {'params': loss_module.parameters(), 'lr': init_lr}], weight_decay=5e-5)
        
        # Define scheduler for the learning rate
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=lr_scheduler_warmup, max_iters=lr_scheduler_maxiters)

        # To run on multiple GPUs:
        if (torch.cuda.device_count() > 1) and (use_multiple_gpus):
            model_step_1= nn.DataParallel(model_step_1)

        # Train
        _ = self.train_model(model=model_step_1, 
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler, 
                        loss_module=loss_module, 
                        device=device, 
                        out_path=out_path,
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        num_epochs=epochs, 
                        eval_freq=eval_freq,
                        earlystopping_threshold=earlystopping_threshold,
                        accum_grad=accum_grad,
                        use_classifier=False)

        del model_step_1, loss_module, optimizer, lr_scheduler

        total_train_end = time.time()
        print(f"Total training time: {(total_train_end - total_train_start)/60:.2f} minutes")
        print()

    def train_classifier(self, 
                        model: nn.Module,
                        model_classifier: nn.Module=None,
                        device: str=None,
                        use_multiple_gpus: bool=False,
                        seed: int=42,
                        init_lr: float=0.001,
                        batch_size: int=256,
                        lr_scheduler_warmup: int=4,
                        lr_scheduler_maxiters: int=50,
                        eval_freq: int=5,
                        epochs: int=50,
                        earlystopping_threshold: int=10,
                        accum_grad: int=1,
                        only_print_best: bool=False):
        """
        Perform training of the machine learning model for making an embedding space.

        Parameters
        ----------
        model : nn.Module
            The embedding space model.

        model_classifier: nn.Module
            The classifier model.
        
        device : str or None, optional
            The device to run the training on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.
        
        use_multiple_gpus : bool, optional
            If using GPU, whether to use one GPU or all avilable GPUs (default is False).

        seed : int, optional
            Random seed for ensuring reproducibility (default is 42).

        init_lr : float, optional
            Initial learning rate for the optimizer (default is 0.001).
        
        batch_size : int, optional
            Batch size for data loading during training (default is 256).
        
        lr_scheduler_warmup : int, optional
            Number of warm-up iterations for the cosine learning rate scheduler (default is 4).
        
        lr_scheduler_maxiters : int, optional
            Maximum number of iterations for the cosine learning rate scheduler (default is 25).

        eval_freq : int, optional
            Rate at which the model is evaluated on validation data (default is 2).
        
        epochs : int, optional
            Number of training epochs (default is 20).
        
        earlystopping_threshold : int, optional
            Early stopping threshold (default is 10).

        accum_grad : int, optional
            How many mini-batches the gradient should eb accumulated for before updating weights (default is 1).
        
        only_print_best : bool, optional
            Whether to print the metrics on each epoch (True) or only for the best epoch (False) (default is False).

        Returns
        -------
        val_loss : float
            Validation loss.
        """

        print("Start training classifier")
        print()

        if model_classifier == None:
            raise ValueError('Need to define model_classifier if train_classifier=True.')
        
        out_path = self.save_model_path

        if device is not None:
            device = device
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Ensure reproducibility
        def rep_seed(seed):
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        rep_seed(seed)

        total_train_start = time.time()

        model_step_2 = copy.deepcopy(model)

        # Load model state
        model_step_2.load_state_dict(torch.load(f'{out_path}model.pt'))

        # To run on multiple GPUs:
        if (torch.cuda.device_count() > 1) and (use_multiple_gpus):
            model_step_2= nn.DataParallel(model_step_2)

        # Define data
        train_loader = data.DataLoader(self.data_env_for_classification, batch_size=batch_size, shuffle=True, drop_last=True)
        min_batch_size = int(np.min([self.data_env_validation_for_classification.X.shape[0], batch_size]))
        val_loader = data.DataLoader(self.data_env_validation_for_classification, batch_size=min_batch_size, shuffle=False, drop_last=True)

        # Define loss
        loss_module = nn.CrossEntropyLoss() 
        # Define Adam optimer
        optimizer = optim.Adam([{'params': model_classifier.parameters(), 'lr': init_lr}], weight_decay=5e-5)
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=lr_scheduler_warmup, max_iters=lr_scheduler_maxiters)

        # Train
        val_loss = self.train_model(model=model_step_2, 
                        model_classifier=model_classifier,
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler, 
                        loss_module=loss_module, 
                        device=device, 
                        out_path=out_path,
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        num_epochs=epochs, 
                        eval_freq=eval_freq,
                        earlystopping_threshold=earlystopping_threshold,
                        accum_grad=accum_grad,
                        use_classifier=True,
                        only_print_best=only_print_best)
        
        del model_step_2, loss_module, optimizer, lr_scheduler
        
        total_train_end = time.time()
        print(f"Total training time: {(total_train_end - total_train_start)/60:.2f} minutes")

        return val_loss
    