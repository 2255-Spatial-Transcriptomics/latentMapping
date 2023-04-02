"""Version 1 of 3 step training. 
    step 1. sample datasets, train scvi end to end on single cell data, get z
    step 2. train scvi on subset of genes for single cell data, match z
    step 3. train sedr on subset of genes for spatial cell data, match z
    step 4. pass high resolution genetic information embedded with scvi through sedr to get spatial data
"""
import scvi
import torch
import numpy as np

from models.scviModels.VAEs import baseVAE, scVAE
from models.descriminatorModels.classifier_trainer import BinaryClassifierTrainer
from models.sedrModels.src.SEDR_trainer import SEDR_Trainer
from data_loaders.anndata_loader import loadSCDataset, loadSTDataset
from data_loaders.tools import sampleHighestExpressions, findCommonGenes, extractGenes, combineLatentSpaceWithLabels
import anndata
import scanpy as sc

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(2)
np.random.seed(2)

# sc_adata = loadSCDataset()
# st_adata = loadSTDataset() 

# x_adata = sampleHighestExpressions(st_adata)

# # find the common genes between the two datasets
# common_genes = findCommonGenes(st_adata, sc_adata)

# # extract only expression data from sc dataset that share the common genes
# xprime_adata = extractGenes(sc_adata, common_genes)

# # extract only expreesion data from st dataset that share the common genes
# xbar_adata = extractGenes(st_adata, common_genes)

adata_sc_read = sc.read_h5ad("data/st_top5k_genes.h5ad")

adata_sc_read.layers["counts"] = adata_sc_read.X.copy()  # preserve counts
sc.pp.normalize_total(adata_sc_read, target_sum=1e4)
sc.pp.log1p(adata_sc_read)
adata_sc_read.raw = adata_sc_read  # freeze the state in `.raw`

sc.pp.filter_genes(adata_sc_read, min_counts=3)
sc.pp.filter_cells(adata_sc_read, min_counts=3)

sc.pp.highly_variable_genes(
    adata_sc_read,
    n_top_genes=2000,
    subset=True,
    layer="counts",
    flavor="seurat_v3"
)

sc.pp.filter_genes(adata_sc_read, min_counts=3)
sc.pp.filter_cells(adata_sc_read, min_counts=3)


LATENT_DIM = 10
# step 1
# # import a scvi model and train end-to-end
baseVAE.setup_anndata(adata_sc_read, layer='counts')
vae1 = baseVAE(adata_sc_read, n_latent=LATENT_DIM)
vae1.train(max_epochs=5)

