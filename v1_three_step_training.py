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

x_adata = sc.read_h5ad("data/scRNA_top2k_genes_new.h5ad")


# find the common genes between the two datasets
# common_genes = findCommonGenes(st_adata, sc_adata)

# extract only expression data from sc dataset that share the common genes
xprime_adata = sc.read_h5ad("data/sc_shared_genes_new.h5ad")

# extract only expreesion data from st dataset that share the common genes
xbar_adata = sc.read_h5ad("data/st_shared_genes_new.h5ad")

LATENT_DIM = 10
# step 1
# # import a scvi model and train end-to-end

print("Running step 1...")
baseVAE.setup_anndata(x_adata)
vae1 = baseVAE(x_adata, n_latent=LATENT_DIM)
vae1.train(max_epochs=10)

# create the latent space for sc data
z = vae1.get_latent_representation(x_adata)

print("step 1 finished")
print(f"latent space of dim {LATENT_DIM} has reconstruction error {vae1.get_reconstruction_error()}")
# # # step 2
breakpoint()
print("Running step 2 for sc data")
scVAE.setup_anndata(xprime_adata)
vae2 = scVAE(xprime_adata, n_latent=LATENT_DIM)
print("Running step 2 for st data")
scVAE.setup_anndata(xbar_adata)
vae3 = scVAE(xbar_adata, n_latent=LATENT_DIM)
print("Running discriminator")
discriminator = BinaryClassifierTrainer(n_latent=LATENT_DIM)

NUM_EPOCHS = 10
MIN_DISCR_ACC = 70 # percent
MAX_DISCR_ITER = 10
for ep_num in range(NUM_EPOCHS):
    
    vae2.module.models = {'vae2': vae2, 'vae3': vae3, 'discriminator': discriminator}
    vae3.module.models = {'vae2': vae2, 'vae3': vae3, 'discriminator': discriminator}
    vae2.module.datasets = {'xbar_adata': xbar_adata, 'xprime_adata': xprime_adata}
    vae3.module.datasets = {'xbar_adata': xbar_adata, 'xprime_adata': xprime_adata}
        
    if vae2.is_trained:
        # first train the discriminator:
        zprime = vae2.get_latent_representation(xprime_adata)
        zprime_labels = np.zeros(zprime.shape[0])

        zbar = vae3.get_latent_representation(xbar_adata)
        zbar_labels = np.ones(zbar.shape[0])

        # merge the two latent spaces and shuffle
        zs = np.concatenate((zbar, zprime))
        labels = np.concatenate((zbar_labels, zprime_labels))

        permutation = np.random.permutation(zs.shape[0])
        zs = zs[permutation, :]
        labels = labels[permutation]
        
        # train the discriminator
        acc = 0 
        num_iters = 0
        while acc < MIN_DISCR_ACC:
            zs_tensor = torch.Tensor(zs)
            labels_tensor = torch.Tensor(labels).unsqueeze(1)
            loss, acc = discriminator.train(zs_tensor, labels_tensor)
            
            if num_iters > MAX_DISCR_ITER:
                print('max iterations reached on discriminator training')
                break
            num_iters += 1
        
        
        # compute the discriminator loss with gradients on both vaes
        zprime_grad = vae2.get_latent_representation_with_grad(xprime_adata)
        zprime_labels = torch.zeros(zprime_grad.shape[0],1)

        zbar_grad = vae3.get_latent_representation_with_grad(xbar_adata)
        zbar_labels = torch.ones(zbar_grad.shape[0],1)

        # merge the two latent spaces and shuffle
        zs_grad = torch.cat((zprime_grad, zbar_grad))
        labels_grad = torch.cat((zprime_labels, zbar_labels)).type(torch.DoubleTensor)
        labels_grad.requires_grad = True
        
        permutation = torch.randperm(zs_grad.size(0))
        zs_grad = zs_grad[permutation, :]
        labels_grad = labels_grad[permutation]
        
        pred_grad = discriminator.forward(zs_grad)
        
        pred_vals = torch.round(torch.sigmoid(pred_grad))

        # this is the inaccuracy of the model, what percentage of the predictions are wrong
        inacc_grad = torch.sum(torch.pow(labels_grad - pred_vals, 2))/labels_grad.shape[0]
        

    # vae3.module.other_losses = {'discriminator_loss': inacc_grad.clone(), 'similarity_loss': torch.tensor([0.])}
        
        
    vae2.train(max_epochs=1)
    vae3.train(max_epochs=1)



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
# vgaeBatchLoader = getVGAEBatchLoader(xbar, spot_xy_locations)

# for epoch in (VGAE_TRAIN_EPOCHS):
#     print('starting epcoh', epoch, 'of VGAE training')
    
#     for xbar_batch, x_st_batch in vgaeBatchLoader:
#         VGAEOptimizer.zero_grad()
#         x_st_graph = vgae.construct_graph(spot_xy_locations)

#         z_bar_batch = vae2.encode(xbar_batch)
        
#         z_combined_batch = vgae.encode(x_st_graph, z_bar_batch)
        
#         x_st_graph_reconstructed = vgae.decode(z_combined_batch)
#         x_st_batch_reconstructed = vgae.recontruct_xy_locations(x_st_graph_reconstructed)
        
#         loss = VGAELoss(x_st_batch, x_st_batch_reconstructed)
        
#         # the gradients on vae2 are fixed, we only perform inference on xbar
#         loss.backward() 
#         VGAEOptimizer.step()
        
        

# # prediction step
# new_cell_gene_matrix = load_sc_dataset('prediction_dataset')
# embeddings = vae2.encode(new_cell_gene_matrix)

# z = vae2.encode(embeddings)

# graph = x_st_graph # use the same graph constructed from step 3
# z_combined = vgae.encode(graph, z)

# graph_reconstructed = vgae.decode(z_combined)

# spatial_info = vgae.reconstruct_xy_locations(graph_reconstructed)


        








# vae.train(max_epochs=10)

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