import os
import torch
import numpy as np
import random
import json
import optuna
from .functions import train as trainer_fun
from .functions import predict as predict_fun
from .functions import make_cell_type_representations as generate_representation_fun
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
    Fit scNear to your Anndata.\n
    Saves model and relevant information to be able to make predictions on new data.

    Parameters
    ----------
    adata 
        An AnnData object containing single-cell RNA-seq data.
    target_key
        Specify key in adata.obs that contain target labels. For example "cell type".
    batch_key
        Specify key in adata.obs that contain batch effect key one wants to correct for. For example "patientID".
    latent_dim
        Dimension of latent space produced by scNear. Default is 100.
    HVGs
        Number of highly variable genes (HVGs) to select as input to scNear. Default is 2000.
    model_path
        Path where model will be saved. Default is "trained_models/".
    train_classifier
        Whether to train scNear as a classifier (True) or to produce a latent space (False). Default is False.
    optimize_classifier
        Whether to use Optuna to optimize the classifier part of the model, assuming train_classifier is True. Default is True.
    only_print_best
        Whether to only print the results of the best epoch of each trial (True) or print performance at each epoch (False).
        Default is False.
    num_trials
        Number of trials for optimizing classifier, assuming train_classifier and optimize_classifier are True. Default is 100.
    use_already_trained_latent_space_generator
        If you've already trained scNear on making a latent space you can use this model when training the classifier (True),\n 
        or if you haven't trained it you can train it as a first step of training the classifier (False). Default is False.
    device
        Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".\n
        Default is None.
    use_multiple_gpus
        If True, use nn.DataParallel() on model. Default is False.
    validation_pct
        The percentage of data used for validation. Default is 0.2, meaning 20%.
    gene_set_gene_limit
        Minimum number of HVGs a gene set must have to be considered. Default is 10.
    seed
        Which random seed to use. Default is 42.
    batch_size
        Mini-batch size used for training latent space producing part of scNear. Default is 236.
    init_lr
        Initial learning rate for training latent space producing part of scNear. Default is 0.001.
    epochs
        Number of epochs for training latent space producing part of scNear. Default is 100.
    lr_scheduler_warmup
        Number of epochs for the warm up part of the CosineWarmupScheduler for training latent space producing part of scNear.\n
        Default is 4.
    lr_scheduler_maxiters
        Number of epochs at which the learning rate would become zero for training latent space producing part of scNear.\n
        Default is 110.
    eval_freq
        Number of epochs between calculating loss of validation data for training latent space producing part of scNear.\n 
        Default is 1.
    earlystopping_threshold
        Number of validated epochs before terminating training if no improvements to the validation loss is made for training\n 
        latent space producing part of scNear. Default is 20.
    accum_grad
        Number of Mini-batches to calculate gradient for before updating weights for training latent space producing part of\n 
        scNear. Default is 1.
    batch_size_classifier
        Mini-batch size used for training classifier part of scNear. Default is 256.
    init_lr_classifier
        Initial learning rate for training classifier part of scNear. Default is 0.001.
    epochs_classifier
        Number of epochs for training classifier part of scNear. Default is 50.
    lr_scheduler_warmup_classifier
        Number of epochs for the warm up part of the CosineWarmupScheduler for training classifier part of scNear.\n
        Default is 4.
    lr_scheduler_maxiters_classifier
        Number of epochs at which the learning rate would become zero for training classifier part of scNear.\n
        Default is 50.
    eval_freq_classifier
        Number of epochs between calculating loss of validation data for training classifier part of scNear.\n 
        Default is 1.
    earlystopping_threshold_classifier
        Number of validated epochs before terminating training if no improvements to the validation loss is made for training\n 
        classifier part of scNear. Default is 10.
    accum_grad_classifier
        Number of Mini-batches to calculate gradient for before updating weights for training classifier part of\n 
        scNear. Default is 1.

    Latent Space Example
    --------
    >>> import scNear
    >>> scNear.train(adata=adata_train, target_key="cell_type", batch_key="batch")
    >>> predictions = scNear.predict(adata=adata_test)

    Classifier Example
    --------
    >>> import scNear
    >>> scNear.train(adata=adata_train, train_classifier=True, target_key="cell_type", batch_key="batch")
    >>> predictions = scNear.predict(adata=adata_test, use_classifier=True)

    Returns
    -------
    None
    """

    if adata.n_vars < HVGs:
        raise ValueError('Number of genes in adata is less than number of HVGs specified to be used.')
    
    if not os.path.exists(f'{model_path}config/'):
        os.makedirs(f'{model_path}config/')
    
    train_env = trainer_fun.train_module(data_path=adata,
                                         save_model_path=model_path,
                                         HVG=True,
                                         HVGs=HVGs,
                                         target_key=target_key,
                                         batch_keys=[batch_key],
                                         validation_pct=validation_pct)
    
    rep_seed(seed=seed)
    model = Model1.Model1(input_dim=HVGs,
                            output_dim=latent_dim)
    
    # Sample configuration dictionary
    config = {
        'input_dim': HVGs,
        'output_dim': latent_dim
    }

    # Define the file path to save the configuration
    config_file_path = f'{model_path}config/model_config.json'

    # Save the configuration dictionary to a JSON file
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)
        
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
                    
    
    if train_classifier:
        if optimize_classifier:
            def objective(trial):

                # Parameters to optimize
                n_neurons_layer1 = trial.suggest_int('n_neurons_layer1', 64, 2048, step=64)
                n_neurons_layer2 = trial.suggest_int('n_neurons_layer2', 64, 2048, step=64)
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)

                rep_seed(seed=seed)
                model_classifier = ModelClassifier.ModelClassifier(input_dim=latent_dim,
                                                                    first_layer_dim=n_neurons_layer1,
                                                                    second_layer_dim=n_neurons_layer2,
                                                                    classifier_drop_out=dropout,
                                                                    num_cell_types=len(adata.obs[target_key].unique()))
            
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

            # Sample configuration dictionary
            config = {
                'input_dim': latent_dim,
                'num_cell_types': len(adata.obs[target_key].unique()),
                'first_layer_dim': opt_dict['n_neurons_layer1'],
                'second_layer_dim': opt_dict['n_neurons_layer2'],
                'classifier_drop_out': opt_dict['dropout']
            }

            # Define the file path to save the configuration
            config_file_path = f'{model_path}config/model_classifier_config.json'

            # Save the configuration dictionary to a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)

            rep_seed(seed=seed)
            model_classifier = ModelClassifier.ModelClassifier(input_dim=config["input_dim"],
                                                                num_cell_types=config["num_cell_types"],
                                                                first_layer_dim=config["first_layer_dim"],
                                                                second_layer_dim=config["second_layer_dim"],
                                                                classifier_drop_out=config["classifier_drop_out"])
        
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
            rep_seed(seed=seed)
            model_classifier = ModelClassifier.ModelClassifier(input_dim=latent_dim,
                                                                num_cell_types=len(adata.obs[target_key].unique()))
            
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
            
            # Sample configuration dictionary
            config = {
                'input_dim': latent_dim,
                'num_cell_types': len(adata.obs[target_key].unique()),
                'first_layer_dim': 512,
                'second_layer_dim': 512,
                'classifier_drop_out': 0.2
            }

            # Define the file path to save the configuration
            config_file_path = f'{model_path}config/model_classifier_config.json'

            # Save the configuration dictionary to a JSON file
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
    Make predictions using scNear.\n
    Make sure you've got a trained model before calling this function.

    Parameters
    ----------
    adata 
        An AnnData object containing single-cell RNA-seq data.
    model_path
        Path where model will be saved. Default is "trained_models/".
    batch_size
        Mini-batch size used for making predictions. Default is 32.
    device
        Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".\n
        Default is None.
    use_multiple_gpus
        If True, use nn.DataParallel() on model. Default is False.
    use_classifier
        Whether to make cell type prediction using classifier part od scNear (True) or predict latent space (False). Default is False.
    detect_unknowns
        Whether to consider samples with a confidence below unknown_threshold as unknown/novel. Default is False.
    unknown_threshold
        Confidence threshold of which if a sample has a confidence below it, it is considered unknown/novel. Default is 0.5.
    return_pred_probs
        Whether to return the probability/confidence of scNear cell type predictions. Default is False.

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
    
    model = Model1.Model1(input_dim=loaded_config["input_dim"],
                          output_dim=loaded_config["output_dim"])

    if use_classifier:

        # Define the file path from which to load the configuration
        config_file_path = f'{model_path}config/model_classifier_config.json'

        # Load the configuration from the JSON file
        with open(config_file_path, 'r') as f:
            loaded_config_classifier = json.load(f)
        
        model_classifier = ModelClassifier.ModelClassifier(input_dim=loaded_config_classifier["input_dim"],
                                                           num_cell_types=loaded_config_classifier["num_cell_types"],
                                                           first_layer_dim=loaded_config_classifier["first_layer_dim"],
                                                           second_layer_dim=loaded_config_classifier["second_layer_dim"],
                                                           classifier_drop_out=loaded_config_classifier["classifier_drop_out"])
        
        pred, pred_prob = predict_fun.predict(data_=adata,
                                              model_path=model_path,
                                              model=model,
                                              model_classifier=model_classifier,
                                              batch_size=batch_size,
                                              device=device,
                                              use_multiple_gpus=use_multiple_gpus,
                                              use_classifier=use_classifier)
    else:
        model_classifier = None

        pred = predict_fun.predict(data_=adata,
                                    model_path=model_path,
                                    model=model,
                                    model_classifier=model_classifier,
                                    batch_size=batch_size,
                                    device=device,
                                    use_multiple_gpus=use_multiple_gpus,
                                    use_classifier=use_classifier)
    
    if return_pred_probs:
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
    Generates cell type representation vectors using the latent space produced by scNear.

    Parameters
    ----------
    adata 
        An AnnData object containing single-cell RNA-seq data.
    target_key
        Specify key in adata.obs that contain target labels. For example "cell type".
    device
        Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".\n
        Default is None.
    use_multiple_gpus
        If True, use nn.DataParallel() on model. Default is False.
    model_path
        Path where model will be saved. Default is "trained_models/".
    save_path
        Path where a .csv file containing the vector representation of each cell type will be saved.\n
        Default is "cell_type_vector_representation/CellTypeRepresentations.csv"
    batch_size
        Mini-batch size used for making predictions. Default is 32.
    method
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
    

    model = Model1.Model1(input_dim=loaded_config["input_dim"],
                          output_dim=loaded_config["output_dim"])

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

def log1p_normalize(data):
    """
    Perform log1p normalization on single-cell RNA sequencing data.\n
    This was the normalization function used for the data in the scNear article.

    Parameters
    ----------
    - data: An AnnData object containing the count data to be normalized.

    Example usage
    --------
    data = log1p_normalize(data)

    Returns
    -------
    data
    """

    data.layers["pp_counts"] = data.X.copy()

    # Calculate size factor
    L = data.X.sum() / data.shape[0]
    data.obs["size_factors"] = data.X.sum(1) / L

    # Normalize using shifted logarithm (log1p)
    scaled_counts = data.X / data.obs["size_factors"].values[:,None]
    data.layers["log1p_counts"] = np.log1p(scaled_counts)

    data.X = data.layers["log1p_counts"]

    return data

def rep_seed(seed=42):
    """
    Sets the random seed for torch, random and numpy.

    Parameters
    ----------
    seed
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