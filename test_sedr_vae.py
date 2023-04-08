#
import sys
sys.path.append('/Users/davidw/spatial-transcriptomics/latentMapping')
import os
import torch
import argparse
import warnings
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

from models.sedrModels.src.utils_func import mk_dir, adata_preprocess, load_visium_sge
from models.sedrModels.src.SEDRVAE_trainer import SEDRVAE_Trainer
from models.sedrModels.src.get_params import getSEDRVAEParams
from models.sedrModels.progress.bar import Bar
import scvi
from scvi.model.utils import mde

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# ################ Parameter setting
sedrvae_params = getSEDRVAEParams()
sedrvae_params.device = device


# ################## Data download folder
data_root = './data/10x_Genomics_Visium/'
# data_name = 'V1_Breast_Cancer_Block_A_Section_1'
# data_name = 'V1_Breast_Cancer_Block_A_Section_2'
# data_name = 'V1_Human_Brain_Section_1'
data_name = 'V1_Human_Brain_Section_1'
save_fold = os.path.join('./output/10x_Genomics_Visium/', data_name)

# ################## Load data
adata_h5 = load_visium_sge(sample_id=data_name, save_path=data_root)

# sedrvae_params.cell_feat_dim = 10
# adata_h5 = scvi.data.synthetic_iid()

adata_h5.var_names_make_unique()
adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=sedrvae_params.cell_feat_dim)

sedrvae_params.cell_num = adata_h5.shape[0]
sedrvae_params.save_path = mk_dir(save_fold)

print('data loaded')
NUM_EPOCHS = 100

# set sedr version = 1 first implementation of the vae from sedr

sedrvae_version = 1

# currently not yet modified for training with dec
sedrvae_params.using_dec = False

# ################## Model training
sedr_vae = SEDRVAE_Trainer(adata_X, sedrvae_params, sedrvae_version)
if sedrvae_params.using_dec:
    for e in range(NUM_EPOCHS):
        sedr_vae.train_with_dec()
else:
    bar = Bar(f'SEDR VAE v{sedrvae_version} model train without DEC: ', max=NUM_EPOCHS)
    bar.check_tty = False
    for e in range(NUM_EPOCHS):
        time, loss = sedr_vae.train_one_episode_without_dec()
        bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
        bar.suffix = bar_str.format(e + 1, NUM_EPOCHS,
                                    batch_time=time * (NUM_EPOCHS - e) / 60, loss=loss.item())
        bar.next()
    bar.finish()
    
latent_features, mu, logvar, _ = sedr_vae.process()

adata_sedrvae = anndata.AnnData(latent_features)
sc.pp.neighbors(adata_sedrvae, n_neighbors=sedrvae_params.eval_graph_n)

adata_sedrvae.obsm['latent'] = latent_features

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix], arg2(fixed_clus_count)[int]
        return:  resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            print('res', res)
            break
    return res


adata_sedrvae.obsm["X_mde"] = scvi.model.utils.mde(latent_features)

n_clusters = 20
eval_resolution = res_search_fixed_clus(adata_sedrvae, n_clusters)
sc.tl.leiden(adata_sedrvae, key_added="leiden", resolution=eval_resolution)

sc.pl.embedding(
    adata_sedrvae,
    basis="X_mde",
    color=["leiden"],
    frameon=False,
    ncols=2,
    save=f'_sedrvae_v{sedrvae_version}_leiden_'+data_name+".png"
)


