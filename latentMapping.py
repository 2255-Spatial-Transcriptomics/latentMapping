import scvi
import torch
import numpy as np
from models.scviModels.VAEs import baseVAE, scVAE2, scVAE3
from models.descriminatorModels.classifier_trainer import BinaryClassifierTrainer
from models.sedrModels.src.SEDR_trainer import SEDR_Trainer
from data_loaders.anndata_loader import loadSCDataset, loadSTDataset
from data_loaders.tools import sampleHighestExpressions, findCommonGenes, extractGenes, combineLatentSpaceWithLabels
import anndata
import scanpy as sc
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(2)
np.random.seed(2)

## Load our datasets. scRNA_top2k_genes_new.h5ad contains the top 2000 genes for
## scRNA data. st_shared_genes_new.h5ad contains the top genes for the spatial
## transcriptomics data
##
x_adata = sc.read_h5ad("Data/scRNA_top2k_genes_new.h5ad")
xprime_adata = sc.read_h5ad("Data/sc_shared_genes_new.h5ad")
xbar_adata = sc.read_h5ad("Data/st_shared_genes_new.h5ad")

# print(adata_sc_read)

LATENT_DIM = 2
NUM_EPOCHS = 10

# step 1
# # import a scvi model and train end-to-end
baseVAE.setup_anndata(x_adata, layer="counts")
vae1 = baseVAE(x_adata, n_latent=LATENT_DIM, latent_distribution='normal')
vae1.train(max_epochs=NUM_EPOCHS)
z = vae1.get_latent_representation()

print("step 1 finished")
print(f"latent space of dim {LATENT_DIM} has reconstruction error {vae1.get_reconstruction_error()}")

# step 2

print("Preparing data for step 2.1 for sc data")
scVAE2.setup_anndata(xprime_adata)
vae2 = scVAE2(xprime_adata, n_latent=LATENT_DIM)

print("Preparing data for step 2.2 for st data")
scVAE3.setup_anndata(xbar_adata)
vae3 = scVAE3(xbar_adata, n_latent=LATENT_DIM)


vae2.module.models = {'vae1': vae1, 'vae2': vae2}
vae2.module.datasets = {'xbar_adata': xbar_adata, 'xprime_adata': xprime_adata, 'z':z}
vae2.train(max_epochs=NUM_EPOCHS)
z_prime = vae2.get_latent_representation()
print(f"latent space of dim {LATENT_DIM} has reconstruction error {vae2.get_reconstruction_error()}")

plt.figure(figsize=(10,10))
plt.scatter(z[:, 0], z[:, 1], label = "2D latent representation: scRNA 2000 genes")
plt.scatter(z_prime[:, 0], z_prime[:, 1], label = "2D latent representation: shared scRNA 500 genes")
plt.legend(loc = 'upper right')
plt.show()

discriminator = BinaryClassifierTrainer(n_latent=LATENT_DIM)

# MIN_DISCR_ACC = 70 # percent
# MAX_DISCR_ITER = 10
# for ep_num in range(NUM_EPOCHS):
    
#     vae3.module.models = {'vae2': vae2, 'vae3': vae3, 'discriminator': discriminator}
#     vae3.module.datasets = {'xbar_adata': xbar_adata, 'xprime_adata': xprime_adata, 'z_prime': z_prime}
        
#     if vae3.is_trained:
#         # first train the discriminator:
#         z_prime_labels = np.zeros(z_prime.shape[0])

#         zbar = vae3.get_latent_representation(xbar_adata)
#         zbar_labels = np.ones(zbar.shape[0])

#         # merge the two latent spaces and shuffle
#         zs = np.concatenate((zbar, z_prime))
#         labels = np.concatenate((zbar_labels, z_prime_labels))

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
        
 
#     vae3.train(max_epochs=1)
