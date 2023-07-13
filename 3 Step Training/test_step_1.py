"""Version 1 of 3 step training. 
    step 1. sample datasets, train scvi end to end on single cell data, get z
    step 2. train scvi on subset of genes for single cell data, match z
    step 3. train sedr on subset of genes for spatial cell data, match z
    step 4. pass high resolution genetic information embedded with scvi through sedr to get spatial data
"""
import scvi
import torch
import numpy as np

from models.scviModels.VAEs import baseVAE, scVAE2, scVAE3
from models.descriminatorModels.classifier_trainer import BinaryClassifierTrainer
from models.sedrModels.src.SEDR_trainer import SEDR_Trainer
from data_loaders.anndata_loader import loadSCDataset, loadSTDataset
from data_loaders.tools import sampleHighestExpressions, findCommonGenes, extractGenes, combineLatentSpaceWithLabels
from models.sedrModels.src.utils_func import mk_dir, adata_preprocess, load_visium_sge
import anndata
import scanpy as sc
from scvi.model.utils import mde
import os 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(2)
np.random.seed(2)


data_root = './data/10x_Genomics_Visium/'
# data_name = 'V1_Breast_Cancer_Block_A_Section_1'
# data_name = 'V1_Breast_Cancer_Block_A_Section_2'
# data_name = 'V1_Human_Brain_Section_1'
data_name = 'V1_Human_Brain_Section_1'
save_fold = os.path.join('./output/10x_Genomics_Visium/', data_name)


# ################## Load data
x_adata = load_visium_sge(sample_id=data_name, save_path=data_root)
x_adata.var_names_make_unique()
x_adata.layers['counts'] = x_adata.X
NUM_EPOCHS = 1
LATENT_DIM = 20

# step 1
# # import a scvi model and train end-to-end
baseVAE.setup_anndata(x_adata, layer='counts')
vae1 = baseVAE(x_adata, n_latent=LATENT_DIM)
vae1.train(max_epochs=NUM_EPOCHS)

x_adata.obsm["X_scVI"] = vae1.get_latent_representation()
sc.pp.neighbors(x_adata, use_rep="X_scVI")


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            print('res', res)
            break
    return res

n_clusters = 20

eval_resolution = res_search_fixed_clus(x_adata, n_clusters)
sc.tl.leiden(x_adata, key_added="leiden", resolution=eval_resolution)


x_adata.obsm["X_mde"] = mde(x_adata.obsm["X_scVI"])

x_adata.write_h5ad(filename='step_1_anndata.h5')
sc.pl.embedding(
    x_adata,
    basis="X_mde",
    color=["leiden"],
    frameon=False,
    ncols=2,
    save=f"brain_epochs={NUM_EPOCHS}_ldim={LATENT_DIM}.png"
)

print('done')
