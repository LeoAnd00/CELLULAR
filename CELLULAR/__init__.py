import os
import torch
import numpy as np
import random
import json
import optuna
from .functions import train as trainer_fun
from .functions import predict as predict_fun
from .functions import make_cell_type_representations as generate_representation_fun
from .functions import data_preprocessing as dp
from .models import Model1 as Model1
from .models import ModelClassifier as ModelClassifier

def train(adata,
          target_key: str,
          batch_key: str,
          latent_dim: int=100,
          HVGs: int=2000,
          model_path: str="trained_models/",
          train_classifier: bool=False,
          optimize_classifier: bool=True,
          only_print_best: bool=True,
          num_trials: int=100,
          use_already_trained_latent_space_generator: bool=False,
          device: str=None,
          use_multiple_gpus: bool=False,
          validation_pct: float=0.2,
          seed: int=42,
          batch_size: int=256,
          init_lr: float=0.001,
          epochs: int=50,
          lr_scheduler_warmup: int=4,
          lr_scheduler_maxiters: int=50,
          eval_freq: int=1,
          earlystopping_threshold: int=20,
          accum_grad: int = 1,
          batch_size_classifier: int = 256,
          init_lr_classifier: float = 0.001,
          lr_scheduler_warmup_classifier: int = 4,
          lr_scheduler_maxiters_classifier: int = 50,
          eval_freq_classifier: int = 1,
          epochs_classifier: int = 50,
          earlystopping_threshold_classifier: int = 10,
          accum_grad_classifier: int = 1):
    """
    Fit CELLULAR to your Anndata.\n
    Worth noting is that adata.X should contain normalized counts.\n
    Saves model and relevant information to be able to make predictions on new data using the predict function.

    Parameters
    ----------
    adata 
        An AnnData object containing single-cell RNA-seq data. adata.X should contain the log1p normalized counts.
    
    target_key (str)
        Specify key in adata.obs that contain target labels. For example "cell_type".
    
    batch_key (str)
        Specify key in adata.obs that contain batch effect key one wants to correct for. For example "patientID".
    
    latent_dim (int, optional)
        Dimension of latent space produced by CELLULAR. Default is 100.
    
    HVGs (int, optional)
        Number of highly variable genes (HVGs) to select as input to CELLULAR. Default is 2000.
    
    model_path (str, optional)
        Path where model will be saved. Default is "trained_models/".
    
    train_classifier (bool, optional)
        Whether to train CELLULAR as a classifier (True) or to produce a latent space (False). Default is False.
    
    optimize_classifier (bool, optional)
        Whether to use Optuna to optimize the classifier part of the model, assuming train_classifier is True. Default is True.
    
    only_print_best (bool, optional)
        Whether to only print the results of the best epoch of each trial (True) or print performance at each epoch (False).
        Default is False.
    
    num_trials (int, optional)
        Number of trials for optimizing classifier, assuming train_classifier and optimize_classifier are True. Default is 100.
    
    use_already_trained_latent_space_generator (bool, optional)
        If you've already trained CELLULAR on making a latent space you can use this model when training the classifier (True), 
        or if you haven't trained it you can train it as a first step of training the classifier (False). Default is False.
    
    device (str, optional)
        Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".
        Default is None.
    
    use_multiple_gpus (bool, optional)
        If True, use nn.DataParallel() on model. Default is False.
    
    validation_pct (float, optional)
        The percentage of data used for validation. Default is 0.2, meaning 20%.
    
    seed (int, optional)
        Which random seed to use. Default is 42.
    
    batch_size (int, optional)
        Mini-batch size used for training latent space producing part of CELLULAR. Default is 256.
    
    init_lr (float, optional)
        Initial learning rate for training latent space producing part of CELLULAR. Default is 0.001.
    
    epochs (int, optional)
        Number of epochs for training latent space producing part of CELLULAR. Default is 50.
    
    lr_scheduler_warmup (int, optional)
        Number of epochs for the warm up part of the CosineWarmupScheduler for training latent space producing part of CELLULAR.
        Default is 4.
    
    lr_scheduler_maxiters (int, optional)
        Number of epochs at which the learning rate would become zero for training latent space producing part of CELLULAR.
        Default is 50.
    
    eval_freq (int, optional)
        Number of epochs between calculating loss of validation data for training latent space producing part of CELLULAR. 
        Default is 1.
    
    earlystopping_threshold (int, optional)
        Number of validated epochs before terminating training if no improvements to the validation loss is made for training 
        latent space producing part of CELLULAR. Default is 20.
    
    accum_grad (int, optional)
        Number of Mini-batches to calculate gradient for before updating weights for training latent space producing part of 
        CELLULAR. Default is 1.
    
    batch_size_classifier (int, optional)
        Mini-batch size used for training classifier part of CELLULAR. Default is 256.
    
    init_lr_classifier (float, optional)
        Initial learning rate for training classifier part of CELLULAR. Default is 0.001.
    
    lr_scheduler_warmup_classifier (int, optional)
        Number of epochs for the warm up part of the CosineWarmupScheduler for training classifier part of CELLULAR.
        Default is 4.
    
    lr_scheduler_maxiters_classifier (int, optional)
        Number of epochs at which the learning rate would become zero for training classifier part of CELLULAR.
        Default is 50.
    
    eval_freq_classifier (int, optional)
        Number of epochs between calculating loss of validation data for training classifier part of CELLULAR.
        Default is 1.
    
    epochs_classifier (int, optional)
        Number of epochs for training classifier part of CELLULAR. Default is 50.
    
    earlystopping_threshold_classifier (int, optional)
        Number of validated epochs before terminating training if no improvements to the validation loss is made for training 
        classifier part of CELLULAR. Default is 10.
    
    accum_grad_classifier (int, optional)
        Number of Mini-batches to calculate gradient for before updating weights for training classifier part of 
        CELLULAR. Default is 1.

    Latent Space Example
    --------
    >>> import CELLULAR
    >>> CELLULAR.train(adata=adata_train, target_key="cell_type", batch_key="batch")
    >>> predictions = CELLULAR.predict(adata=adata_test)

    Classifier Example
    --------
    >>> import CELLULAR
    >>> CELLULAR.train(adata=adata_train, train_classifier=True, target_key="cell_type", batch_key="batch")
    >>> predictions = CELLULAR.predict(adata=adata_test, use_classifier=True)

    Returns
    -------
    None
    """

    # Raise error if the number of HVGs is not possible to achieve
    if adata.n_vars < HVGs:
        raise ValueError('Number of genes in adata is less than number of HVGs specified to be used.')
    
    # Creat folder to save model if it doesn't exist
    if not os.path.exists(f'{model_path}config/'):
        os.makedirs(f'{model_path}config/')
    
    # Initiate training class
    train_env = trainer_fun.train_module(data_path=adata,
                                         save_model_path=model_path,
                                         HVG=True,
                                         HVGs=HVGs,
                                         target_key=target_key,
                                         batch_keys=[batch_key],
                                         validation_pct=validation_pct)
    
    # Define random seed
    rep_seed(seed=seed)

    # Initiate model
    model = Model1.Model1(input_dim=HVGs,
                          output_dim=latent_dim)
    
    # Make configuration dictionary with model variables
    config = {
        'input_dim': HVGs,
        'output_dim': latent_dim
    }

    # Define the file path to save the configuration
    config_file_path = f'{model_path}config/model_config.json'

    # Save the configuration dictionary as a JSON file
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train embedding space generating model
    if use_already_trained_latent_space_generator == False: 
        train_env.train(model=model,
                        device=device,
                        use_multiple_gpus=use_multiple_gpus,
                        seed=seed,
                        batch_size=batch_size,
                        use_target_weights=True,
                        use_batch_weights=True,
                        init_temperature=0.25,
                        min_temperature=0.1,
                        max_temperature=2.0,
                        init_lr=init_lr,
                        lr_scheduler_warmup=lr_scheduler_warmup,
                        lr_scheduler_maxiters=lr_scheduler_maxiters,
                        eval_freq=eval_freq,
                        epochs=epochs,
                        earlystopping_threshold=earlystopping_threshold,
                        accum_grad=accum_grad)
                    
    # Train classifier
    if train_classifier:
        if optimize_classifier:
            def objective(trial):

                # Parameters to optimize
                n_neurons_layer1 = trial.suggest_int('n_neurons_layer1', 64, 2048, step=64)
                n_neurons_layer2 = trial.suggest_int('n_neurons_layer2', 64, 2048, step=64)
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)

                # Set random seed
                rep_seed(seed=seed)

                # Initiate classifier
                model_classifier = ModelClassifier.ModelClassifier(input_dim=latent_dim,
                                                                    first_layer_dim=n_neurons_layer1,
                                                                    second_layer_dim=n_neurons_layer2,
                                                                    classifier_drop_out=dropout,
                                                                    num_cell_types=len(adata.obs[target_key].unique()))

                # Train classifier
                val_loss = train_env.train_classifier(model=model,
                                                        model_classifier=model_classifier,
                                                        device=device,
                                                        use_multiple_gpus=use_multiple_gpus,
                                                        seed=seed,
                                                        init_lr=learning_rate,
                                                        batch_size=batch_size_classifier,
                                                        lr_scheduler_warmup=lr_scheduler_warmup_classifier,
                                                        lr_scheduler_maxiters=lr_scheduler_maxiters_classifier,
                                                        eval_freq=eval_freq_classifier,
                                                        epochs=epochs_classifier,
                                                        earlystopping_threshold=earlystopping_threshold_classifier,
                                                        accum_grad=accum_grad_classifier,
                                                        only_print_best=only_print_best)
                return val_loss
            
            # Define the study and optimize
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=num_trials)

            print('Number of finished trials: ', len(study.trials))
            print('Best trial:')
            trial = study.best_trial

            print('  Value: ', trial.value)
            print('  Params: ')
            opt_dict = {}
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))
                opt_dict[key] = value

            # Make configuration dictionary with optimized model variables
            config = {
                'input_dim': latent_dim,
                'num_cell_types': len(adata.obs[target_key].unique()),
                'first_layer_dim': opt_dict['n_neurons_layer1'],
                'second_layer_dim': opt_dict['n_neurons_layer2'],
                'classifier_drop_out': opt_dict['dropout']
            }

            # Define the file path to save the configuration
            config_file_path = f'{model_path}config/model_classifier_config.json'

            # Save the configuration dictionary as a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Train model with best variable values
                
            # Set random seed
            rep_seed(seed=seed)

            # Initiate classifier
            model_classifier = ModelClassifier.ModelClassifier(input_dim=config["input_dim"],
                                                                num_cell_types=config["num_cell_types"],
                                                                first_layer_dim=config["first_layer_dim"],
                                                                second_layer_dim=config["second_layer_dim"],
                                                                classifier_drop_out=config["classifier_drop_out"])

            # Train classifier
            _ = train_env.train_classifier(model=model,
                                                model_classifier=model_classifier,
                                                device=device,
                                                use_multiple_gpus=use_multiple_gpus,
                                                seed=seed,
                                                init_lr=opt_dict["learning_rate"],
                                                batch_size=batch_size_classifier,
                                                lr_scheduler_warmup=lr_scheduler_warmup_classifier,
                                                lr_scheduler_maxiters=lr_scheduler_maxiters_classifier,
                                                eval_freq=eval_freq_classifier,
                                                epochs=epochs_classifier,
                                                earlystopping_threshold=earlystopping_threshold_classifier,
                                                accum_grad=accum_grad_classifier)

        else:
            # Set random seed
            rep_seed(seed=seed)

            # Initiate classifier
            model_classifier = ModelClassifier.ModelClassifier(input_dim=latent_dim,
                                                                num_cell_types=len(adata.obs[target_key].unique()))
            
            # Train classifier
            _ = train_env.train_classifier(model=model,
                                            model_classifier=model_classifier,
                                            device=device,
                                            use_multiple_gpus=use_multiple_gpus,
                                            seed=seed,
                                            init_lr=init_lr_classifier,
                                            batch_size=batch_size_classifier,
                                            lr_scheduler_warmup=lr_scheduler_warmup_classifier,
                                            lr_scheduler_maxiters=lr_scheduler_maxiters_classifier,
                                            eval_freq=eval_freq_classifier,
                                            epochs=epochs_classifier,
                                            earlystopping_threshold=earlystopping_threshold_classifier,
                                            accum_grad=accum_grad_classifier)
            
            # Make configuration dictionary with model variables
            config = {
                'input_dim': latent_dim,
                'num_cell_types': len(adata.obs[target_key].unique()),
                'first_layer_dim': 512,
                'second_layer_dim': 512,
                'classifier_drop_out': 0.2
            }

            # Define the file path to save the configuration
            config_file_path = f'{model_path}config/model_classifier_config.json'

            # Save the configuration dictionary as a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)

def predict(adata,
            model_path: str="trained_models/",
            batch_size: int=32, 
            device: str=None, 
            use_multiple_gpus: bool=False,
            use_classifier: bool=False,
            return_pred_probs: bool=False):
    """
    Make predictions using CELLULAR.\n
    Worth noting is that adata.X should contain the normalized counts.\n
    Make sure you've got a trained model before calling this function.

    Parameters
    ----------
    adata 
        An AnnData object containing single-cell RNA-seq data. adata.X should contain the normalized counts.
    
    model_path (str, optional)
        Path where model is saved. Default is "trained_models/".
    
    batch_size (int, optional)
        Mini-batch size used for making predictions. Default is 32.
    
    device (str, optional)
        Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".
        Default is None.
    
    use_multiple_gpus (bool, optional)
        If True, use nn.DataParallel() on model. Default is False.
    
    use_classifier (bool, optional)
        Whether to make cell type prediction using classifier part of CELLULAR (True) or predict latent space (False). Default is False.
    
    return_pred_probs (bool, optional)
        Whether to return the probability/likelihood of cell type predictions. Default is False.

    Returns
    -------
    If return_pred_probs == False:
        return pred
        
    If return_pred_probs == True:
        return pred, pred_prob
    """
    
    # Define the file path from which to load the configuration
    config_file_path = f'{model_path}config/model_config.json'

    # Load the configuration from the JSON file
    with open(config_file_path, 'r') as f:
        loaded_config = json.load(f)
    
    # Define model and load variable values
    model = Model1.Model1(input_dim=loaded_config["input_dim"],
                          output_dim=loaded_config["output_dim"])

    # Whether to make prediction using the classifier or not
    if use_classifier:

        # Define the file path from which to load the configuration
        config_file_path = f'{model_path}config/model_classifier_config.json'

        # Load the configuration from the JSON file
        with open(config_file_path, 'r') as f:
            loaded_config_classifier = json.load(f)
        
        # Initiate classifier with loaded variables
        model_classifier = ModelClassifier.ModelClassifier(input_dim=loaded_config_classifier["input_dim"],
                                                           num_cell_types=loaded_config_classifier["num_cell_types"],
                                                           first_layer_dim=loaded_config_classifier["first_layer_dim"],
                                                           second_layer_dim=loaded_config_classifier["second_layer_dim"],
                                                           classifier_drop_out=loaded_config_classifier["classifier_drop_out"])
        
        # Make cell type predictions, and the likelihood of the predictions
        pred, pred_prob = predict_fun.predict(data_=adata,
                                              model_path=model_path,
                                              model=model,
                                              model_classifier=model_classifier,
                                              batch_size=batch_size,
                                              device=device,
                                              use_multiple_gpus=use_multiple_gpus,
                                              use_classifier=use_classifier)
    else: # Predict embedding space
        model_classifier = None

        pred = predict_fun.predict(data_=adata,
                                    model_path=model_path,
                                    model=model,
                                    model_classifier=model_classifier,
                                    batch_size=batch_size,
                                    device=device,
                                    use_multiple_gpus=use_multiple_gpus,
                                    use_classifier=use_classifier)
    
    if return_pred_probs and use_classifier:
        return pred, pred_prob
    else:
        return pred

def generate_representations(adata, 
                             target_key: str,
                             device: str=None, 
                             use_multiple_gpus: bool=False,
                             model_path: str="trained_models/",
                             save_path: str="cell_type_vector_representation",
                             batch_size: int=32,
                             method: str="centroid"):
    """
    Generates cell type representation vectors using the latent space produced by CELLULAR. \n
    Worth noting is that adata.X should contain the normalized counts.

    Parameters
    ----------
    adata 
        An AnnData object containing single-cell RNA-seq data. adata.X should contain the normalized counts.
    
    target_key (str)
        Specify key in adata.obs that contain target labels. For example "cell type".
    
    device (str, optional)
        Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".
        Default is None.
    
    use_multiple_gpus (bool, optional)
        If True, use nn.DataParallel() on model. Default is False.
    
    model_path (str, optional)
        Path where model is saved. Default is "trained_models/".
    
    save_path (str, optional)
        Path where a .csv file containing the vector representation of each cell type will be saved.
        Default is "cell_type_vector_representation/CellTypeRepresentations.csv"
    
    batch_size (int, optional)
        Mini-batch size used for making predictions. Default is 32.
    
    method (str, optional)
        Which method to use for making representations (Options: "centroid", "median", "medoid"). Default is "centroid".

    Returns
    -------
    representations: pd.Dataframe()
    """
    
    # Define the file path from which to load the configuration
    config_file_path = f'{model_path}config/model_config.json'

    # Load the configuration from the JSON file
    with open(config_file_path, 'r') as f:
        loaded_config = json.load(f)
    
    # Initiate model
    model = Model1.Model1(input_dim=loaded_config["input_dim"],
                          output_dim=loaded_config["output_dim"])

    # Create representations
    representations = generate_representation_fun.generate_representation(data_=adata, 
                                                                          model=model,
                                                                          model_path=model_path,
                                                                          device=device, 
                                                                          use_multiple_gpus=use_multiple_gpus,
                                                                          target_key=target_key,
                                                                          save_path=save_path, 
                                                                          batch_size=batch_size, 
                                                                          method=method)

    return representations

def novel_cell_type_detection(adata, model_path: str="trained_models/", threshold: float=0.25):
    """
    Detects the presence of novel cell types in the data based on a given threshold.\n
    This function uses a pre-trained model to predict cell types from the input data (adata).
    It then checks the minimum likelihood of predictions and compares it with the threshold value.
    If the minimum likelihood is lower than the threshold, it suggests the presence of a novel cell type.
    Otherwise, it cannot confidently state the presence of a novel cell type.

    Parameters:
    -----------
    adata
        Anndata object containing the data. adata.X should contain the normalized counts.
    model_path (str)
        Path to the pre-trained model used for prediction.
    threshold (float, optional)
        Threshold value for determining the presence of a novel cell type. Default is 0.25.

    Returns:
    --------
    None
    """

    # Make prediction and retrieve likelihood
    _, pred_prob = predict(adata=adata, model_path=model_path, use_classifier=True, return_pred_probs=True)

    # Define the file path from which to load the configuration
    config_file_path = f'{model_path}config/model_classifier_config.json'

    # Load the configuration from the JSON file
    with open(config_file_path, 'r') as f:
        loaded_config_classifier = json.load(f)

    num_cell_types = loaded_config_classifier["num_cell_types"] # Num cell types
    num_cell_types_inv = 1/num_cell_types
    pred_prob = [(x - num_cell_types_inv) / (1 - num_cell_types_inv) for x in pred_prob] # Normalize

    # Calculate the minimum likelihood detected
    min_prob = np.min(pred_prob)

    if min_prob < threshold:
        print(f"Minimum likelihood was {min_prob:.4f}, which is lower than the threshold value of {threshold}")
        print("This suggests that a novel cell type is present in the data")
    else:
        print(f"Minimum likelihood was {min_prob:.4f}, which is higher than the threshold value of {threshold}")
        print("Can't confidently state whether or not there could be a novel cell type in the data or not")


def log1p_normalize(data):
    """
    Scales all samples by calculating a scale factor and log1p normalization on single-cell RNA sequencing data.\n
    This function will set data.X to be the log1p normalized counts.

    Parameters
    ----------
    data
        An AnnData object containing the count data to be normalized.

    Example usage
    --------
    data = log1p_normalize(data)

    Returns
    -------
    data: log1p normalized Anndata object will be in data.X.
    """

    # Calculate size factor
    L = data.X.sum() / data.shape[0]
    data.obs["size_factors"] = data.X.sum(1) / L

    # Normalize using shifted logarithm (log1p)
    scaled_counts = data.X / data.obs["size_factors"].values[:,None]
    data.X = np.log1p(scaled_counts)

    return data

def pre_process_data(adata_: str=None, 
                     save_path: str = None, 
                     count_file_path: str = None, 
                     gene_data_path: str = None, 
                     barcode_data_path: str = None, 
                     thresholds:list = [5,5,5,5],
                     min_num_cells_per_gene: int = 20):
    """
    Performs quality control, and filtering on the input data.
    The input data can be a matrix.mtx.gz, genes.tsv.gz, and barcodes.tsv.gz file. Or it can be an Anndata object.
    If using a Anndata object as input this should be specified in the adata_ variable. Then you can leave
    count_file_path, gene_data_path, and barcode_data_path empty.

    Parameters
    ----------
    adata_ (str/AnnData)
        If using an adata object as input, input the adata into this variable.
    
    save_path (str)
        Path and name to be used for saving the QC adata file (Ex: "results/qc_adata"). If not specified, the adata will not be downloaded.
    
    count_file_path (str, optional)
        Path to count data file. Default is None. (Ex: "data/matrix.mtx.gz")
    
    gene_data_path (str, optional)
        Path to gene data file. Default is None. (Ex: "data/genes.tsv.gz")
    
    barcode_data_path (str, optional)
        Path to barcode data file. Default is None. (Ex: "data/barcodes.tsv.gz")
    
    thresholds (list, optional)
        Thresholds for MAD during QC filtering. Default is [5, 5, 5, 5]. First is for log_n_counts filtering, second for log_n_genes, third for pct_counts_in_top_20_genes, and forth is for mt_frac filtering.
        - 'log_n_counts': Shifted log of sum of counts per cell.
        - 'log_n_genes': Shifted log of number of unique genes expressed per cell.
        - 'pct_counts_in_top_20_genes': Fraction of total counts among the top 20 genes with the highest counts.
        - 'mt_frac': Fraction of mitochondrial counts.

    min_num_cells_per_gene (int, optional)
        Minimum number of cells a gene must be expressed by to keep the gene in the data. 
    
    Returns
    -------
    adata: QC filtered Anndata.
    """

    # Perform QC filtering on adata
    adata = dp.QC_filter(adata_=adata_,
                         save_path=save_path,
                         count_file_path=count_file_path,
                         gene_data_path=gene_data_path,
                         barcode_data_path=barcode_data_path,
                         thresholds=thresholds,
                         min_num_cells_per_gene=min_num_cells_per_gene)
    
    return adata

def rep_seed(seed: int = 42):
    """
    Sets the random seed for torch, random and numpy.

    Parameters
    ----------
    seed (int, optional)
        Which random seed to use. Default is 42.

    Returns
    -------
    None
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False