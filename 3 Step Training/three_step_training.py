# ### Full implementation of training architecture

import os
import torch
import argparse
import warnings
import random
import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData
from VGAE.src.graph_func import graph_construction
from VGAE.src.utils_func import mk_dir, adata_preprocess, load_visium_sge
from VGAE.src.SEDR_train import SEDR_Train

## step 0: load the data

def load_adata(data_name, n_clusters=20, pca_n_comps=200):
    data_root = './data/10x_Genomics_Visium/'
    # data_name = 'Visium_FFPE_Mouse_Brain'
    save_fold = os.path.join('./output/10x_Genomics_Visium/', data_name)
    # ################## Load data
    adata_h5 = load_visium_sge(sample_id=data_name, save_path=data_root)
    adata_h5.var_names_make_unique()
    adata_X = adata_preprocess(adata_h5, min_cells=5, pca_n_comps=pca_n_comps)
    return adata_h5, adata_X

def plot_spatial(spatial_data, data_name):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # get the spatial information
    spatial_x = spatial_data['x_um']
    spatial_y = spatial_data['y_um']

    ax.scatter(spatial_x, spatial_y, s=1)
    ax.set_xlim(min(spatial_x), max(spatial_x))
    ax.set_ylim(min(spatial_y), max(spatial_y))
    plt.title(f"spatial of {data_name}")
    plt.show()
    
def plot_cell2gene(cell_gene_matrix, data_name):
    # cell_to_gene_matrix = sc.pp.normalize_total(cell_gene_matrix, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    plt.figure(figsize=(8,80), dpi=100)
    plt.spy(cell_gene_matrix, markersize=0.01)
    plt.title(f"cell to gene of {data_name}")
    plt.show()
    
    

# load the cell-gene matrix from https://github.com/spacetx-spacejam/datas

datapath = "data/"
data_name = "MERFISH"
cell_gene_matrix_full = pd.read_csv(datapath + 'cell-gene-matrix.csv')
cell_gene_matrix = cell_gene_matrix_full.iloc[:, 1:-6].transpose()
spatial_data = pd.read_csv(datapath + 'Allen_MERFISH_spots_with_anatomy.csv')
genes = pd.read_csv(datapath + 'MERFISH_genes.csv')

# SEDR Dataset
# data_root = './output/data/10x_Genomics_Visium/'
# adata_h5 = load_visium_sge(sample_id=data_name, save_path=data_root)
# adata_h5.var_names_make_unique()

# print(cell_gene_matrix.head())

genes = set(genes['gene'])
# enablePrint()

print('dataset: ', data_name)
print('size:', cell_gene_matrix.shape)
print('number of genes', len(genes))
# plot_cell2gene(cell_gene_matrix, data_name)
# plot_spatial(spatial_data, data_name)


# configure settings for SEDR
sc.settings.figdir = './output/figures/'
sc.settings.writedir = './output/write/'
sc.settings.cachedir = './output/cache/'
sc.settings.datasetdir = './output/data/'

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device + " =====")

argsDict = {
    'k': 20,
    'knn_distanceType': 'euclidean',
    'epochs': 200,
    'cell_feat_dim': 200, 
    'feat_hidden1': 100,
    'feat_hidden2': 20,
    'gcn_hidden1': 32,
    'gcn_hidden2': 8,
    'p_drop': 0.2,
    'using_dec': True,
    'using_mask': False,
    'feat_w': 10,
    'gcn_w': 0.1,
    'dec_kl_w': 10,
    'gcn_lr': 0.01,
    'gcn_decay': 0.01,
    'dec_cluster_n': 10,
    'dec_interval': 20,
    'dec_tol': 0.00,
    'eval_resolution': 1,
    'eval_graph_n': 20,
    'device': device,
    'n_clusters': 20
}
params = argparse.Namespace(**argsDict)

spatial_selected = np.array(cell_gene_matrix_full[['x_um', 'y_um']])
print('selected spatial data', spatial_selected.shape)


# step one 
# train VAE on cell_gene_matrix




# step 1:
# train sphere using VAE_loss

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import sys
import os
sys.path.insert(0, os.getcwd())
from VAE.scphere.util.util import read_mtx
from VAE.scphere.util.trainer import Trainer
from VAE.scphere.model.vae import SCPHERE
from VAE.scphere.util.plot import plot_trace


# import tensorflow and suppress all warnings on M1 chip
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)


# Preparing a sparse matrix and using ~2000 variable genes for efficiency. 
# Data can be downloaded from single cell portal (login with a Google account):
# https://singlecell.broadinstitute.org/single_cell/study/SCP551/scphere#study-download

EPOCH = 5
TT_RATIO = 0.75
SPLIT = False

# NAME = 'lung_human_ASK440'
# data_dir = os.getcwd() + '/VAE/example/data/' + NAME + '/'

# mtx = data_dir + NAME + '.mtx'
# input = read_mtx(mtx)
# input = input.transpose().todense()

input = cell_gene_matrix.to_numpy()
# label_filename = data_dir + NAME + '_celltype.tsv'
# df = pd.read_csv(label_filename)
# label = np.array(df.values.tolist())

nrow, ncol = input.shape

# if SPLIT:
#     train_size = round(nrow*TT_RATIO)
#     test_size = round(nrow*(1-TT_RATIO))

#     train_index = np.random.choice(nrow, train_size, replace=False)
#     test_index = np.array(set(np.arange(0, nrow)) - set(train_index))

#     train_x = input[train_index, :]
#     test_x = input[test_index, :]

#     train_y = label[train_index]
#     test_y = label[test_index]

#     input = train_x


# The batch vector should be 0-based
# For the cases there are no batch vectors, we can set n_batch=0,
# and create an artificial batch vector (just for running scPhere,
# the batch vector will not influence the results), e.g.,
batch = np.zeros(nrow) * -1

# build the model
# n_gene: the number of genes
# n_batch: the number of batches for each component of the batch vector.
#          For this case, we set it to 0 as there is no need to correct for batch effects. 
# z_dim: the number of latent dimensions, setting to 2 for visualizations
# latent_dist: 'vmf' for spherical latent spaces, and 'wn' for hyperbolic latent spaces
# observation_dist: the gene expression distribution, 'nb' for negative binomial
# seed: seed used for reproducibility
# batch_invariant: batch_invariant=True to train batch-invarant scPhere.
#                  To train batch-invariant scPhere, i.e.,
#                  a scPhere model taking gene expression vectors only as inputs and
#                  embedding them to a latent space. The trained model can be used to map new data,
#                  e.g., from new patients that have not been seen during training scPhere
#                  (assuming patient is the major batch vector)
model = SCPHERE(n_gene=ncol, n_batch=0, batch_invariant=False,
                z_dim=2, latent_dist='vmf',
                observation_dist='nb', seed=0)

# training
# model: the built model above
# x: the UMI count matrix
# max_epoch: the number of epochs used to train scPhere
# mb_size: the number of cells used in minibatch training
# learning_rate: the learning rate of the gradient descent optimization algorithm
trainer = Trainer(model=model, x=input, batch_id=batch, max_epoch=EPOCH,
                  mb_size=128, learning_rate=0.001)

trainer.train()

# save the trained model
# save_path = os.getcwd() + '/VAE/example/demo-out/'

# model_name = save_path + NAME + '_model_' + str(EPOCH) + 'epoch'
# model.save_sess(model_name)

# embedding all the data
z_mean = model.encode(input, batch)
# np.savetxt(save_path +
#            NAME + '_latent_' + str(EPOCH) + 'epoch.tsv',
#            z_mean)

# the log-likelihoods
ll = model.get_log_likelihood(input, batch)
# np.savetxt(save_path +
#            NAME + '_ll_' + str(EPOCH) + 'epoch.tsv',
#            z_mean)

# Plotting log-likelihood and kl-divergence at each iteration
# plot_trace([np.arange(len(trainer.status['kl_divergence']))*50] * 2,
#            [trainer.status['log_likelihood'], trainer.status['kl_divergence']],
#            ['log_likelihood', 'kl_divergence'])
# # plt.show()

# plt.savefig(save_path +
#             NAME + str(EPOCH) + '_train.png')




### Predict Accuracy
# model.predict



# step 2 
# losses

# define the scphere vae
vae2 = SCPHERE(n_gene=ncol, n_batch=0, batch_invariant=False,
                z_dim=2, latent_dist='vmf',
                observation_dist='nb', seed=0)

scphere_vae2 = Trainer(model=vae2, x=input, batch_id=batch, max_epoch=EPOCH,
                  mb_size=128, learning_rate=0.001)


# define the SEDR vae
vae3 = 
# training
for iter_i in range(scphere_vae2.max_iter):
    x_mb, y_mb = scphere_vae2.x.next_batch(scphere_vae2.mb_size)
    feed_dict = {scphere_vae2.model.x: x_mb,
                    scphere_vae2.model.batch_id: y_mb,
                    }

    scphere_vae2.session.run(scphere_vae2.optimizer.train_step, feed_dict)

    if (iter_i % 50) == 0:
        var = [scphere_vae2.model.log_likelihood,
                scphere_vae2.model.kl, scphere_vae2.model.ELBO,
                ]

        log_likelihood, kl_divergence, elbo = \
            scphere_vae2.session.run(var, feed_dict)

        scphere_vae2.status['log_likelihood'].append(log_likelihood)
        scphere_vae2.status['kl_divergence'].append(kl_divergence)
        scphere_vae2.status['elbo'].append(elbo)

        info_print = {'Log-likelihood': log_likelihood,
                        'ELBO': elbo, 'KL': kl_divergence}
        print(iter_i, '/', scphere_vae2.max_iter, info_print)
# step 3
# GVAE

# save_fold = 'merfish_test'
# adata = AnnData(np.array(cell_gene_matrix)).transpose()
# adata.var_names_make_unique()
# adata_X = adata_preprocess(adata, min_cells=5, pca_n_comps=params.cell_feat_dim)
# graph_dict = graph_construction(spatial_selected, adata.shape[0], params)
# params.cell_num = adata.shape[0]
# params.save_path = mk_dir(save_fold)
# print('==== Graph Construction Finished')

# sedr_net = SEDR_Train(adata_X, graph_dict, params)
# if params.using_dec:
#     sedr_net.train_with_dec()
# else:
#     sedr_net.train_without_dec()
# sedr_feat, _, _, _ = sedr_net.process()