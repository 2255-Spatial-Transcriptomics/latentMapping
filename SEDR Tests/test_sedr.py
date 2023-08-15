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
from models.sedrModels.src.graph_func import graph_construction
from models.sedrModels.src.utils_func import mk_dir, adata_preprocess, load_visium_sge
from models.sedrModels.src.SEDR_trainer import SEDR_Trainer
from models.sedrModels.src.get_params import getSEDRParams
from models.sedrModels.progress.bar import Bar

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

# ################ Parameter setting
sedr_params = getSEDRParams()
sedr_params.device = device


# ################## Data download folder
data_root = './data/10x_Genomics_Visium/'
data_name = 'V1_Breast_Cancer_Block_A_Section_1'
# data_name = 'V1_Breast_Cancer_Block_A_Section_2'
# data_name = 'V1_Human_Brain_Section_1'
# data_name = 'V1_Human_Brain_Section_1'
save_fold = os.path.join('./output/10x_Genomics_Visium/', data_name)


# ################## Load data
adata_h5 = load_visium_sge(sample_id=data_name, save_path=data_root)
adata_h5.var_names_make_unique()
adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=sedr_params.cell_feat_dim)
graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], sedr_params)
sedr_params.cell_num = adata_h5.shape[0]
sedr_params.save_path = mk_dir(save_fold)
print('==== Graph Construction Finished')

# graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], sedr_params)
# sedr_params.cell_num = adata_h5.shape[0]
# sedr_params.save_path = mk_dir(save_fold)
# print('==== Graph Construction Finished')


NUM_EPOCHS = 500

# set sedr version = 1 to use original sedr, no modifications
# version 2: modified sedr where only one latent space is produced for both spatial and sc

sedr_version = 2

# currently not yet modified for training with dec
sedr_params.using_dec = False
# ################## Model training
sedr_net = SEDR_Trainer(adata_X, graph_dict, sedr_params, sedr_version)
if sedr_params.using_dec:
    for e in range(NUM_EPOCHS):
        sedr_net.train_with_dec()
else:
    bar = Bar(f'SEDR v{sedr_version} model train without DEC: ', max=NUM_EPOCHS)
    bar.check_tty = False
    for e in range(NUM_EPOCHS):
        time, loss = sedr_net.train_one_episode_without_dec()
        bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
        bar.suffix = bar_str.format(e + 1, NUM_EPOCHS,
                                    batch_time=time * (NUM_EPOCHS - e) / 60, loss=loss.item())
        bar.next()
    bar.finish()
sedr_feat, _, _, _ = sedr_net.process()


# ################## Result plot
#### using random data
# adata_dummy = np.random.rand(sedr_feat.shape[0], 15)
# adata_sedr = anndata.AnnData(adata_dummy)

adata_sedr = anndata.AnnData(sedr_feat)
adata_sedr.uns['spatial'] = adata_h5.uns['spatial']
adata_sedr.obsm['spatial'] = adata_h5.obsm['spatial']

sc.pp.neighbors(adata_sedr, n_neighbors=sedr_params.eval_graph_n)
sc.tl.umap(adata_sedr)


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

eval_resolution = res_search_fixed_clus(adata_sedr, n_clusters)
sc.tl.leiden(adata_sedr, key_added="SEDR_leiden", resolution=eval_resolution)

image = adata_h5.uns['spatial'][data_name]['images']['hires']
# image = (1-image*2)[:, :, (1,2,0)]
adata_sedr.uns['spatial'][data_name]['images']['hires'] = image
# will be saved under figures/ with sedr version and dataset
sc.pl.spatial(adata_sedr, img_key="hires", color=['SEDR_leiden'], save=f"_sedr_v{sedr_version}_clustering_500ep_"+data_name+".png")




