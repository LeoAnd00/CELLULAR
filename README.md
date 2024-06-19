# CELLULAR: CELLUlar contrastive Learning for Annotation and Representation

**Abstract**
Batch effects are a significant concern in single-cell RNA sequencing (scRNA-Seq) data analysis, where variations in the data can be attributed to factors unrelated to cell types. This can make downstream analysis a challenging task. In this study, a neural network model is designed utilizing contrastive learning and a novel loss function for creating an generalizable embedding space from scRNA-Seq data. When benchmarked against multiple established methods on scRNA-Seq integration, the model outperformed existing methods on creating a generalizable embedding space on multiple datasets. A downstream application that was investigated for the embedding space was cell type annotation. When compared against multiple well established cell type classifiers the model in this study displayed a performance competitive with top performing methods across multiple metrics, such as accuracy, balanced accuracy and F1 score. These findings motivates the meaningfulness contained within the generated embedding space by the model, highlighting its potential applications.

## Necessary programming languages
- Python version >= 3.10.5

## Setup
```
pip install CELLULAR
```

## Functionality
The following functions have been included: <br>
* Training function for the embedding space model.
* Training function for the classifier model.
* Predict function for generating an embedding space.
* Predict function for performing cell type annotation.
* Function for novel cell type detection.
* Function for creating cell type representation vectors.
* Function for applying the same normalization strategy as was used in this study, giving the end user the option of using the same strategy or implementing their own.
* Function for automatic preprocessing, although it is still recommended for end users to use their own preprocessing pipeline to make sure it is appropriate for their data.

## Data
Data for the tutorial can be installed from [here](https://doi.org/10.5281/zenodo.10959788).

## Usage

### For making embedding space
```
import scanpy as sc
import scNear

adata_train = sc.read("train_data.h5ad", cache=True)
scNear.train(adata=adata_train, target_key="cell_type", batch_key="batch")

adata_test = sc.read("test_data.h5ad", cache=True)
predictions = scNear.predict(adata=adata_test)
```
### For cell type annotation
```
import scanpy as sc
import scNear

adata_train = sc.read("train_data.h5ad", cache=True)
scNear.train(adata=adata_train, train_classifier=True, target_key="cell_type", batch_key="batch")

adata_test = sc.read("test_data.h5ad", cache=True)
predictions = scNear.predict(adata=adata_test, use_classifier=True)
```
### For novel cell type detection
```
import scanpy as sc
import scNear

adata_train = sc.read("train_data.h5ad", cache=True)
scNear.train(adata=adata_train, target_key="cell_type", batch_key="batch")

adata_test = sc.read("test_data.h5ad", cache=True)
scNear.novel_cell_type_detection(adata=adata_test)
```
### For making cell type representations
```
import scanpy as sc
import scNear

adata_train = sc.read("train_data.h5ad", cache=True)
scNear.train(adata=adata_train, target_key="cell_type", batch_key="batch")

adata_test = sc.read("test_data.h5ad", cache=True)
representations = scNear.generate_representations(adata=adata_test, target_key="cell_type")
```

## Tutorials
See *Tutorial/embedding_space_tutorial.ipynb*, *Tutorial/classification_tutorial.ipynb*, and *Tutorial/pre_process_tutorial.ipynb*.

## Citation
Coming soon
