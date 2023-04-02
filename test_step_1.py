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

adata_sc_read = sc.read_h5ad("/content/drive/MyDrive/Capstone scRNA dataset/scRNA_top2k_genes_new.h5ad")

LATENT_DIM = 10
# step 1
# # import a scvi model and train end-to-end
baseVAE.setup_anndata(adata_sc_read, layer='counts')
vae1 = baseVAE(adata_sc_read, n_latent=LATENT_DIM)
vae1.train(max_epochs=5)

