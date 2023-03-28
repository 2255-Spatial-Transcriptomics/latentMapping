import scvi
import torch
import numpy as np

from models.scviModels.VAEs import baseVAE, scVAE
from models.descriminatorModels.classifier import BinaryClassifier
from models.sedrModels.src.SEDR_trainer import SEDR_Trainer
from data_loaders.anndata_loader import loadSCDataset, loadSTDataset
from data_loaders.tools import sampleHighestExpressions, findCommonGenes, extractGenes, combineLatentSpaceWithLabels
import anndata
import scanpy as sc
from step_3_functions import *


import warnings
warnings.filterwarnings("ignore")

sc_adata = loadSCDataset()
st_adata = loadSTDataset() 

x_adata = sampleHighestExpressions(st_adata)

# find the common genes between the two datasets
common_genes = findCommonGenes(st_adata, sc_adata)

# extract only expression data from sc dataset that share the common genes
xprime_adata = extractGenes(sc_adata, common_genes)

# extract only expreesion data from st dataset that share the common genes
xbar_adata = extractGenes(st_adata, common_genes)

# xsmall_adata = anndata.read_h5ad('data/sc_top2k_genes.h5ad')
# other_adata = anndata.AnnData(xsmall_adata.X)
# other_adata.obs_names = xsmall_adata.obs_names
# other_adata.var_names = xsmall_adata.var_names
# sc.pp.filter_genes(xsmall_adata, min_counts=3)
# sc.pp.filter_cells(xsmall_adata, min_counts=3)
# xsmall_adata.layers["counts"] = xsmall_adata.X.copy()
# sc.pp.normalize_total(xsmall_adata)
# sc.pp.log1p(xsmall_adata)
# xsmall_adata.raw = xsmall_adata
# baseVAE.setup_anndata(xsmall_adata)
# vae1 = baseVAE(xsmall_adata, n_latent=2)
# vae1.train(max_epochs=5)


# step 1
# # import a scvi model and train end-to-end
baseVAE.setup_anndata(x_adata)
vae1 = baseVAE(x_adata, n_latent=2)
vae1.train(max_epochs=5)

# create the latent space for sc data
z = vae1.get_latent_representation(x_adata)


# # # step 2

# print(baseVAE._model_summary_string)


# # # do the actual data processing in another script, the load functions just reads in the post-processed csv files



# # step 2
# vae1_params = {}

scVAE.setup_anndata(xprime_adata)
vae2 = scVAE(xprime_adata, n_latent=2)
discriminator = BinaryClassifier(n_latent=2)

## here we should use SEDR, but using scvi for now

vae3 = SEDR_Trainer()

NUM_EPOCHS = 10
MIN_DISCR_ACC = 1.2
MAX_DISCR_ITER = 10
for ep_num in range(NUM_EPOCHS):
    
    # first train the discriminator:
    if vae2.is_trained:
        zprime = vae2.get_latent_representation(xprime_adata)
        
        # placeholder for now, TODO: get zbar using sedr vae
        zbar = torch.tensor([1,2,3])
        
        # TODO: implement this function
        zs, labels = combineLatentSpaceWithLabels(zprime, zbar)
        
        discriminator_acc = 0
        iter = 0 
        while discriminator_acc < MIN_DISCR_ACC:
            discriminator.train(zs, labels)
            discriminator_acc = discriminator.predict(zs, labels)
            
            if iter > MAX_DISCR_ITER:
                print('\n\nreached max iterations on discriminator')
                break
            iter += 1
            
          
        vae2.module.other_losses = {'similarity_loss': torch.tensor(1-discriminator_acc.clone().detach().requires_grad_(True))}
       
    else:
        
        vae2.module.other_losses = {'similarity_loss': torch.tensor(0)}
       
        
    vae2.train(max_epochs=1)
    



# vae2 = getVAE(vae1_params) # VAE such as scphere or scvi

# vae2Params = {}
# vae2 = getVAE(vae2_params) # VAE in SEDR architecture

# vaeBatchLoader = getVAEBatchLoader(xprime, xbar, z)
# '''
# each batch contains N cells, with cell-gene expression of genes in x', xbar, 
# '''

# descriminatorParams = {}
# descriminator = getDescriminator(descriminatorParams)

# for epoch in range(NUM_EPOCHS):
    
#     # first train the descriminator
    
#     # combine the dataset for descrimination with z's, labels
#     # labels can be 0 if from sc dataset, and 1 if from st dataset
#     zprime = vae1(xprime)
#     zbar = vae2(xbar)
#     # then shuffle the dataset
#     combined_z, zlabels = combineLatentSpaceWithLabels(zprime, zbar)
    
#     it = 0 
#     while it < MAX_DESCR_ITERS:
        
#         descriminatorOptimizer.zero_grad()
        
#         predictions = descriminator(combined_z)
#         accuracy = descriminatorLoss(predictions, zlabels)
#         ''' cross entropy loss for descriminator'''
#         if accuracy > DESCR_ACC:
#             break
#         descriminatorLoss.backward()
#         descriminatorOptimizer.step()
        
#     # make sure this is a good enough number
#     print('descriminator accuracy', accuracy)
    
#     # freeze weights on descriminator and prepare for prediction
#     descriminator.predict()
    
#     # next iterate through the dataset in batches
#     for xprime_batch, xbar_batch, z_batch in vaeBatchLoader:
#         '''
#         we can also pass x through the network each iteration to get z instead of pre-computing
#         if there are not enought batches of the spatial data, just circulate and re-use some batches
#         '''
        
#         VAEOptimizer.zero_grad()
        
#         zprime_batch = vae1.encode(xprime_batch)
#         xprime_batch_reconstructed = vae1.decode(zprime_batch)
        
#         zbar_batch = vae2.encode(xbar_batch)
#         xbar_batch_reconstructed = vae2.decode(zbar_batch)
        
#         # david + sayem
#         loss1 = L2Loss(zprime_batch, z_batch)
#         loss2 = VAELoss(xprime_batch_reconstructed, xprime_batch)
#         loss3 = VAELoss(xbar_batch, xbar_batch_reconstructed)
        
#         combined_zbatches, zbatch_labels = combineLatentSpaceWithLabels(zprime_batch, zbar_batch)
#         classifications = descriminator(combined_zbatches)
        
#         loss4 = GANLoss(classifications, zbatch_labels)
        
#         total_batch_loss = 0
#         for weight, loss in zip(loss_weights, [loss1, loss2, loss3, loss4]):
#             total_batch_loss += weight*loss

#         total_batch_loss.backward()    
#         VAEOptimizer.step()
        
    
# # fix the weights on the vae for SEDR
# vae2.predict()

# # step 3, train VGAE on spatial data, this is the VAGE branch from SEDR 
# # daniel will work on this



# vgae_params = {}
# vgae = getVGAE(vgae_params) # build the vgae
# vgaeBatchLoader = getVGAEBatchLoader(xbar_adata, zbar)
# vgaeOptimizer = vgae.optimizer

# for epoch in (VGAE_TRAIN_EPOCHS):
#     print('starting epcoh', epoch, 'of VGAE training')
    
#     for xbar_batch, x_st_batch in vgaeBatchLoader:
#         vgaeOptimizer.zero_grad()
#         x_st_batch = vgae.construct_graph(zbar)

#         z_st_batch = vgae.encode(x_st_batch)
#         x_st_batch_reconstructed = vgae.decode(z_st_batch)
        
#         loss1 = VGAELoss(x_st_batch, x_st_batch_reconstructed)
        
#         # the gradients on vae2 are fixed, we only perform inference on xbar
#         zbar_batch = vae2.encode(xbar_batch)
#         loss2 = getCrossEntropy(zbar_batch, z_st_batch) # make the spatial embeddings as similar as possible to the cell-gene embeddings

#         loss = WEIGHT_ONE*loss1 + WEIGHT_TWO*loss2
        
#         loss.backward() 
#         vgaeOptimizer.step()        

# # prediction step
# new_cell_gene_matrix = loadSCDataset('prediction_dataset')
# embeddings = vae2.encode(new_cell_gene_matrix)
# spatial_info = vgae.decode(embeddings)


# adata.obsm["X_scVI"] = vae.get_latent_representation()
# sc.pp.neighbors(adata, use_rep="X_scVI")
# sc.tl.leiden(adata)

# adata.obsm["X_mde"] = mde(adata.obsm["X_scVI"])

# sc.pl.embedding(
#     adata,
#     basis="X_mde",
#     color=["batch", "leiden"],
#     frameon=False,
#     ncols=1,
# )

# sc.pl.embedding(adata, basis="X_mde", color=["cell_type"], frameon=False, ncols=1)