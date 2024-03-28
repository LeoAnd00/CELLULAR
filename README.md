# scNear - contrastive learning Neural networks to generate scRNA-Seq Embeddings, Annotations and Representations

**Intorduction**
To learn a meaningful representation of cells, this project aims to utilize scRNA-Seq data to train a neural network to
produce an efficient, lower-dimensional embedding space for the data. This space can then be used for downstream tasks, such as cell type annotation, novel cell type detection, generating cell type representations or simply for visualization. 

## Necessary programming languages
- Python version 3.10.5

## Setup
```
pip install scNear
```

## How to run

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

## Authors
Leo Andrekson
