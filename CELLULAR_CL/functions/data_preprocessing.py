import pandas as pd
import numpy as np
import scanpy as sc
import warnings


def QC_filter(adata_: str=None, 
              save_path: str = None, 
              count_file_path: str = None, 
              gene_data_path: str = None, 
              barcode_data_path: str = None, 
              thresholds: list = [5,5,5,5],
              min_num_cells_per_gene: int = 20):
    """
    Performs quality control, and filtering on the input data.
    The input data can be a matrix.mtx.gz, genes.tsv.gz, and barcodes.tsv.gz file. Or it can be an Anndata object.
    If using a Anndata object as input this should be specified in the adata_ variable. Then you can leave
    count_file_path, gene_data_path, and barcode_data_path empty.

    Args:
    - adata_ (str/AnnData): If using an adata object as input, input the adata into this variable.
    - save_path (str): Path and name to be used for saving the QC adata file. (Ex: "results/qc_adata")
    - count_file_path (str, optional): Path to count data file. Default is None. (Ex: "data/matrix.mtx.gz")
    - gene_data_path (str, optional): Path to gene data file. Default is None. (Ex: "data/genes.tsv.gz")
    - barcode_data_path (str, optional): Path to barcode data file. Default is None. (Ex: "data/barcodes.tsv.gz")
    - thresholds (list, optional): Thresholds for QC filtering. Default is [5, 5, 5, 5]. First is for log_n_counts filtering, second for log_n_genes, third for pct_counts_in_top_20_genes, and forth is for mt_frac filtering.
        - 'log_n_counts': Shifted log of sum of counts per cell.
        - 'log_n_genes': Shifted log of number of unique genes expressed per cell.
        - 'pct_counts_in_top_20_genes': Fraction of total counts among the top 20 genes with the highest counts.
        - 'mt_frac': Fraction of mitochondrial counts.
    - min_num_cells_per_gene (int, optional): Minimum number of cells a gene must be expressed by to be kept. 
    
    Returns:
    adata: QC filtered Anndata.
    """

    # Read data
    if adata_ != None:
        adata = adata_
    else:
        count_data = count_file_path
        gene_data = gene_data_path
        barcode_data = barcode_data_path
        adata = read_sc_data(count_data, gene_data, barcode_data)
        adata.var_names_make_unique()

    # Add QC metrics to adata
    adata = QC().QC_metric_calc(adata)

    # Remove outliers
    qc_adata = QC().QC_filter_outliers(adata, thresholds, expression_limit=min_num_cells_per_gene, print_=True)

    if save_path != None:
        qc_adata.write(f"{save_path}.h5ad")

    return qc_adata



def read_sc_data(count_data: str, gene_data: str, barcode_data: str):
    """
    Read single-cell RNA sequencing data and associated gene and barcode information.

    Parameters:
    - count_data (str): The path to the count data file in a format compatible with Scanpy.
    - gene_data (str): The path to the gene information file in tab-separated format.
    - barcode_data (str): The path to the barcode information file in tab-separated format.

    Returns:
    - data (AnnData): An AnnData object containing the count data, gene information, and barcode information.

    This function loads single-cell RNA sequencing data, gene information, and barcode information, and organizes them into
    an AnnData object. The count data is expected to be in a format supported by Scanpy, and it is transposed to ensure
    genes are represented as rows and cells as columns. Gene information is used to annotate the genes in the data, and
    barcode information is used to annotate the cells in the data.
    """

    #Load data
    data = sc.read(count_data, cache=True).transpose()
    data.X = data.X.toarray()

    # Load genes and barcodes
    genes = pd.read_csv(gene_data, sep='\t', header=None)
    barcodes = pd.read_csv(barcode_data, sep='\t', header=None)

    # set genes
    genes.rename(columns={0:'gene_id', 1:'gene_symbol'}, inplace=True)
    genes.set_index('gene_symbol', inplace=True)
    data.var = genes

    # set barcodes
    barcodes.rename(columns={0:'barcode'}, inplace=True)
    barcodes.set_index('barcode', inplace=True)
    data.obs = barcodes

    return data

class QC():
    """
    Quality Control (QC) class for single-cell RNA sequencing data.

    This class provides methods for performing quality control on single-cell RNA sequencing data, including
    Median Absolute Deviation (MAD) based outlier detection and filtering based on various QC metrics.
    """

    def __init__(self):
        pass

    def median_absolute_deviation(self, data):
        """
        Calculate the Median Absolute Deviation (MAD) of a dataset.

        Parameters:
        - data (list or numpy.ndarray): The dataset for which MAD is calculated.

        Returns:
        - float: The Median Absolute Deviation (MAD) of the dataset.
        """
        median = np.median(data)
        
        absolute_differences = np.abs(data - median)

        mad = np.median(absolute_differences)
        
        return mad

    def QC_metric_calc(self, data):
        """
        Calculate various quality control metrics for single-cell RNA sequencing data.

        Parameters:
        - data (AnnData): An AnnData object containing the single-cell RNA sequencing data.

        Returns:
        AnnData: An AnnData object with additional QC metrics added as observations.

        This method calculates the following QC metrics and adds them as observations to the input AnnData object:
        - 'n_counts': Sum of counts per cell.
        - 'log_n_counts': Shifted log of 'n_counts'.
        - 'n_genes': Number of unique genes expressed per cell.
        - 'log_n_genes': Shifted log of 'n_genes'.
        - 'pct_counts_in_top_20_genes': Fraction of total counts among the top 20 genes with the highest counts.
        - 'mt_frac': Fraction of mitochondrial counts.
        """

        # Sum of counts per cell
        data.obs['n_counts'] = data.X.sum(1)
        # Shifted log of n_counts
        data.obs['log_n_counts'] = np.log(data.obs['n_counts']+1)
        # Number of unique genes per cell
        data.obs['n_genes'] = (data.X > 0).sum(1)
        # Shifted lof og n_genes
        data.obs['log_n_genes'] = np.log(data.obs['n_genes']+1)

        # Fraction of total counts among the top 20 genes with highest counts
        top_20_indices = np.argpartition(data.X, -20, axis=1)[:, -20:]
        top_20_values = np.take_along_axis(data.X, top_20_indices, axis=1)
        data.obs['pct_counts_in_top_20_genes'] = (np.sum(top_20_values, axis=1)/data.obs['n_counts'])

        # Fraction of mitochondial counts
        mt_gene_mask = [gene.startswith('MT-') for gene in data.var_names]
        data.obs['mt_frac'] = data.X[:, mt_gene_mask].sum(1)/data.obs['n_counts']

        return data

    def MAD_based_outlier(self, data, metric: str, threshold: int = 5):
        """
        Detect outliers based on the Median Absolute Deviation (MAD) of a specific metric.

        Parameters:
        - data (AnnData): An AnnData object containing the single-cell RNA sequencing data.
        - metric (str): The name of the observation metric to use for outlier detection.
        - threshold (int): The threshold in MAD units for outlier detection.

        Returns:
        numpy.ndarray: A boolean array indicating outlier cells.

        This method detects outlier cells in the input AnnData object based on the specified metric and threshold.
        Outliers are identified using the MAD-based approach.
        """

        data_metric = data.obs[metric]
        # calculate indexes where outliers are detected
        outlier = (data_metric < np.median(data_metric) - threshold * self.median_absolute_deviation(data_metric)) | (
                    np.median(data_metric) + threshold * self.median_absolute_deviation(data_metric) < data_metric)
        return outlier

    def QC_filter_outliers(self, data, threshold: list = [5,5,5,5], expression_limit: int = 20, print_: bool = True):
        """
        Filter outlier cells from the single-cell RNA sequencing data based on QC metrics.

        Parameters:
        - data (AnnData): An AnnData object containing the single-cell RNA sequencing data.
        - threshold (list): A list of threshold values for each QC metric in the following order:
            - log_n_counts threshold
            - log_n_genes threshold
            - pct_counts_in_top_20_genes threshold
            - mt_frac threshold
        - expression_limit (int): Threshold of how many cell must have counts of a gene in order for it to be preserved.
        - print_ (bool): Whether to print what was filtered away or not.

        Returns:
        AnnData: An AnnData object with outlier cells removed.

        This method performs QC filtering on the input AnnData object by removing cells that are identified as outliers
        based on the specified threshold values for each QC metric. Additionally, it filters out genes with fewer than
        expression_limit unique cells expressing them.
        """

        # Ignore FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        num_cells_before = data.n_obs
        num_genes_before = data.n_vars

        data.obs["outlier"] = (self.MAD_based_outlier(data, "log_n_counts", threshold[0])
            | self.MAD_based_outlier(data, "log_n_genes", threshold[1])
            | self.MAD_based_outlier(data, "pct_counts_in_top_20_genes", threshold[2])
            | self.MAD_based_outlier(data, "mt_frac", threshold[3])
        )

        # Print how many detected outliers by each QC metric 
        outlier1 = (self.MAD_based_outlier(data, "log_n_genes", threshold[1]))
        outlier2 = (self.MAD_based_outlier(data, "log_n_counts", threshold[0]))
        outlier3 = (self.MAD_based_outlier(data, "pct_counts_in_top_20_genes", threshold[2]))
        outlier4 = (self.MAD_based_outlier(data, "mt_frac", threshold[3]))
        
        # Filter away outliers
        data = data[(~data.obs.outlier)].copy()

        # Min "expression_limit" cells - filters out 0 count genes
        sc.pp.filter_genes(data, min_cells=expression_limit)

        if print_:
            print(f"Number of cells before QC filtering: {num_cells_before}")
            print(f"Number of cells removed by log_n_genes filtering: {sum(1 for item in outlier1 if item)}")
            print(f"Number of cells removed by log_n_counts filtering: {sum(1 for item in outlier2 if item)}")
            print(f"Number of cells removed by pct_counts_in_top_20_genes filtering: {sum(1 for item in outlier3 if item)}")
            print(f"Number of cells removed by mt_frac filtering: {sum(1 for item in outlier4 if item)}")
            print(f"Number of cells post QC filtering: {data.n_obs}")
            #Filter genes:
            print('Number of genes before filtering: {:d}'.format(num_genes_before))
            print(f'Number of genes after filtering so there is a minimum of {expression_limit} unique cells per gene: {data.n_vars}')

        return data


