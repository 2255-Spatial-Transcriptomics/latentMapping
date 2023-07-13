"""Version 2 of 3 step training. 
    using the vae from sedr instead of the one from scvi
"""
import scvi
import torch
import numpy as np

from models.scviModels.VAEs import baseVAE, scVAE2, scVAE3
from models.descriminatorModels.classifier_trainer import BinaryClassifierTrainer
from models.sedrModels.src.SEDR_trainer import SEDR_Trainer
from data_loaders.anndata_loader import loadSCDataset, loadSTDataset
from models.sedrModels.src.get_params import getSEDRVAEParams
from models.sedrModels.src.SEDRVAE_trainer import SEDRVAE_Trainer
from data_loaders.tools import sampleHighestExpressions, findCommonGenes, extractGenes, combineLatentSpaceWithLabels
from models.sedrModels.progress.bar import Bar
import anndata
import scanpy as sc

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(2)
np.random.seed(2)

from models.sedrModels.src.utils_func import mk_dir, adata_preprocess, load_visium_sge
import os

vae1_params = getSEDRVAEParams()
################### Loading data
# sc_adata = loadSCDataset()
# st_adata = loadSTDataset() 


# x_adata = sc.read_h5ad("data/scRNA_top2k_genes_new.h5ad")

# # extract only expression data from sc dataset that share the common genes
# xprime_adata = sc.read_h5ad("data/sc_shared_genes_new.h5ad")

# # extract only expreesion data from st dataset that share the common genes
# xbar_adata = sc.read_h5ad("data/st_shared_genes_new.h5ad")

x_id = 'V1_Human_Brain_Section_1'
x_adata = load_visium_sge(sample_id=x_id, save_path='./data/10x_Genomics_Visium/')
x_adata.var_names_make_unique()
x_adata.layers['counts'] = x_adata.X

######## run this the first time
pca_file = f'data/10x_Genomics_Visium/V1_Human_Brain_Section_1/{x_id}_pca{vae1_params.cell_feat_dim}.npy'

if os.path.exists(pca_file):
    pca_comps = np.load(pca_file)
else:
    pca_comps = adata_preprocess(x_adata, min_cells=5, pca_n_comps=vae1_params.cell_feat_dim)
    with open(pca_file, 'wb') as f:
        np.save(f, pca_comps)


x_adata.pca_comps = pca_comps
vae1_params.cell_num = x_adata.shape[0]
vae1_params.latent_dim = 20
xprime_adata = load_visium_sge(sample_id='V1_Human_Brain_Section_2', save_path='./data/10x_Genomics_Visium/')
xprime_adata.var_names_make_unique()
xprime_adata.layers['counts'] = xprime_adata.X


xbar_adata = load_visium_sge(sample_id='V1_Human_Brain_Section_2', save_path='./data/10x_Genomics_Visium/')
xbar_adata.var_names_make_unique()
xbar_adata.layers['counts'] = xbar_adata.X



################## Step 1 ####################
## utilize a SEDR vae model and train end-to-end
NUM_EPOCHS = 500


vae1_params.device = 'cpu'

print("Running step 1...")
vae1_params.using_dec = False
vae1version = 1

vae1 = SEDRVAE_Trainer(x_adata.pca_comps, vae1_params, vae1version)

if vae1_params.using_dec:
    for e in range(NUM_EPOCHS):
        vae1.train_with_dec()
else:
    bar = Bar(f'SEDR VAE v{vae1version} model train without DEC: ', max=NUM_EPOCHS)
    bar.check_tty = False
    for e in range(NUM_EPOCHS):
        time, loss = vae1.train_one_episode_without_dec()
        bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
        bar.suffix = bar_str.format(e + 1, NUM_EPOCHS,
                                    batch_time=time * (NUM_EPOCHS - e) / 60, loss=loss.item())
        bar.next()

    bar.finish()
    
z, mu, logvar, _ = vae1.process()


print("step 1 finished")
vae1.model.eval()
loss_function = torch.nn.MSELoss()
recon_error = loss_function(
                vae1.model.decoder(
                    torch.tensor(z, requires_grad=False)), 
                    torch.tensor(x_adata.pca_comps.copy(), requires_grad=False)).item()
print(f"latent space of dim {z.shape[1]} has reconstruction error {recon_error}")

# # # step 2

# print("Running step 2 for sc data")
# scVAE2.setup_anndata(xprime_adata)
# vae2 = scVAE2(xprime_adata, n_latent=LATENT_DIM)

# scVAE3.setup_anndata(xbar_adata)
# vae3 = scVAE3(xbar_adata, n_latent=LATENT_DIM)

# discriminator = BinaryClassifierTrainer(n_latent=LATENT_DIM)

# NUM_EPOCHS = 10
# MIN_DISCR_ACC = 70 # percent
# MAX_DISCR_ITER = 10
# for ep_num in range(NUM_EPOCHS):
    
#     vae2.module.models = {'vae2': vae2, 'vae3': vae3, 'discriminator': discriminator}
#     vae3.module.models = {'vae2': vae2, 'vae3': vae3, 'discriminator': discriminator}
#     vae2.module.datasets = {'xbar_adata': xbar_adata, 'xprime_adata': xprime_adata, 'z':z}
#     vae3.module.datasets = {'xbar_adata': xbar_adata, 'xprime_adata': xprime_adata}
        
#     if vae2.is_trained:
#         # first train the discriminator:
#         zprime = vae2.get_latent_representation(xprime_adata)
#         zprime_labels = np.zeros(zprime.shape[0])

#         zbar = vae3.get_latent_representation(xbar_adata)
#         zbar_labels = np.ones(zbar.shape[0])

#         # merge the two latent spaces and shuffle
#         zs = np.concatenate((zbar, zprime))
#         labels = np.concatenate((zbar_labels, zprime_labels))

#         permutation = np.random.permutation(zs.shape[0])
#         zs = zs[permutation, :]
#         labels = labels[permutation]
        
#         # train the discriminator
#         acc = 0 
#         num_iters = 0
#         while acc < MIN_DISCR_ACC:
#             print(f"\ntraining discriminator: iter {num_iters + 1}")
#             zs_tensor = torch.Tensor(zs)
#             labels_tensor = torch.Tensor(labels).unsqueeze(1)
#             loss, acc = discriminator.train(zs_tensor, labels_tensor)
            
#             if num_iters > MAX_DISCR_ITER:
#                 print(f'\nmax iterations reached on discriminator training, breaking with acc = {acc}')
#                 break
#             if acc > MIN_DISCR_ACC:
#                 print(f"\ndiscriminator accuracy = {acc}, breaking ... ")
#             num_iters += 1
        
 
#     vae2.train(max_epochs=1)
#     vae3.train(max_epochs=1)






        