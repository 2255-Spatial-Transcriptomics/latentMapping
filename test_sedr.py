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
sedr_params.epochs = 1

# ################## Data download folder
data_root = './data/10x_Genomics_Visium/'
data_name = 'V1_Breast_Cancer_Block_A_Section_1'
save_fold = os.path.join('./output/10x_Genomics_Visium/', data_name)


# ################## Load data
adata_h5 = load_visium_sge(sample_id=data_name, save_path=data_root)
adata_h5.var_names_make_unique()
adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=sedr_params.cell_feat_dim)
graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], sedr_params)
sedr_params.cell_num = adata_h5.shape[0]
sedr_params.save_path = mk_dir(save_fold)
print('==== Graph Construction Finished')

# ################## Model training
sedr_net = SEDR_Trainer(adata_X, graph_dict, sedr_params)
if sedr_params.using_dec:
    for e in range(10):
        sedr_net.train_with_dec()
else:
    sedr_net.train_without_dec()
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

# will be saved under figures/<name>.png

sc.pl.spatial(adata_sedr, img_key="hires", color=['SEDR_leiden'], save="_sedr_rand_clustering_"+data_name+".png")




