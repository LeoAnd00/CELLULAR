import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


def predict(data_, 
            model_path: str, 
            model: nn.Module, 
            model_classifier: nn.Module=None,
            batch_size: int=32, 
            device: str=None, 
            use_multiple_gpus: bool=False,
            use_classifier: bool=False):
    """
    Generate latent represntations for data using the trained model.
    Note: data_.X must contain the normalized data, as is also required for training.

    Parameters
    ----------
    data_ : AnnData
        An AnnData object containing data for prediction.
    
    model_path : str
        The path to the directory where the trained model is saved.
    
    model : nn.Module
        If the model is saved as torch.save(model.state_dict(), f'{out_path}model.pt') one have to input a instance of the model. If torch.save(model, f'{out_path}model.pt') was used then leave this as None (default is None).
    
    model_classifier : nn.Module
        The classifier model.

    batch_size : int, optional
        Batch size for data loading during prediction (default is 32).

    device : str or None, optional
        The device to run the prediction on (e.g., "cuda" or "cpu"). If None, it automatically selects "cuda" if available, or "cpu" otherwise.
    
    use_multiple_gpus
        If True, use nn.DataParallel() on model. Default is False.

    use_classifier: bool, optional
        Whether to make cell type annotation predictions or generate latent space (defualt is False). 

    Returns
    -------
    preds : np.array
        Array of predicted latent embeddings.
    """

    data_ = prep_test_data(data_, model_path)

    if device is not None:
        device = device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.load_state_dict(torch.load(f'{model_path}model.pt'))
    # To run on multiple GPUs:
    if (torch.cuda.device_count() > 1) and (use_multiple_gpus):
        model= nn.DataParallel(model)
    model.to(device)

    if use_classifier:
        if model_classifier == None:
            raise ValueError('model_classifier needs to be defined if use_classifier=True')
        
        model_classifier.load_state_dict(torch.load(f'{model_path}model_classifier.pt'))
        
        model_classifier.to(device)

    data_loader = data.DataLoader(data_, batch_size=batch_size, shuffle=False)

    preds = []
    model.eval()
    if use_classifier:
        model_classifier.eval()
    with torch.no_grad():
        for data_inputs in data_loader:

            data_inputs = data_inputs.to(device)

            pred = model(data_inputs)

            if use_classifier:
                preds_latent = pred.cpu().detach().to(device)
                pred = model_classifier(preds_latent)

            pred = pred.cpu().detach().numpy()

            # Ensure all tensors have at least two dimensions
            if pred.ndim == 1:
                pred = np.expand_dims(pred, axis=0)  # Add a dimension along axis 0

            preds.extend(pred)

    if use_classifier:

        if os.path.exists(f"{model_path}/ModelMetadata/onehot_label_encoder.pt"):
            label_encoder = torch.load(f"{model_path}/ModelMetadata/label_encoder.pt")
            onehot_label_encoder = torch.load(f"{model_path}/ModelMetadata/onehot_label_encoder.pt")
        else:
            raise ValueError("There's no files containing target encodings (label_encoder.pt and onehot_label_encoder.pt).")

        preds = np.array(preds)

        binary_preds = []
        pred_prob = []
        # Loop through the predictions
        for pred in preds:
            # Apply thresholding
            binary_pred = np.where(pred == np.max(pred), 1, 0)

            binary_preds.append(binary_pred)
            pred_prob.append(float(pred[binary_pred==1]))

        # Convert the list of arrays to a numpy array
        binary_preds = np.array(binary_preds)

        # Reverse transform the labels
        labels = []
        for row in binary_preds:
            temp = onehot_label_encoder.inverse_transform(row.reshape(1, -1))
            labels.append(label_encoder.inverse_transform(temp.ravel()))
        
        labels = np.array([np.ravel(label)[0] for label in labels])

        pred_prob = np.array(pred_prob)

        return labels, pred_prob

    else:
        
        return np.array(preds)

class prep_test_data(data.Dataset):
    """
    PyTorch Dataset for preparing test data for the machine learning model.

    Parameters:
        adata : AnnData
            An AnnData object containing single-cell RNA sequencing data.
        model_path 
            Path to where model is saved.

    Methods:
        __len__()
            Returns the number of data samples.

        __getitem__(idx) 
            Retrieves a specific data sample by index.
    """

    def __init__(self, adata, model_path):

        # HVG gene names
        hvg_genes = torch.load(f"{model_path}/ModelMetadata/hvg_genes.pt")

        self.adata = adata
        self.adata = self.adata[:, hvg_genes].copy()

        self.X = self.adata.X
        self.X = torch.tensor(self.X)

    def __len__(self):
        """
        Get the number of data samples in the dataset.

        Returns
        ----------
        int: The number of data samples.
        """

        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Get a specific data sample by index.

        Parameters
        ----------
        idx (int): Index of the data sample to retrieve.

        Returns
        ----------
        tuple: A tuple containing the data point and pathways.
        """

        data_point = self.X[idx]

        return data_point

