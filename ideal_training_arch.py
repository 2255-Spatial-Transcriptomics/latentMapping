# pseudocode for model training, these functions will need to be populated

import torch
import numpy as np

# do the actual data processing in a nother script, the load functions just reads in the post-processed csv files
full_cell_gene_matrix_sc = load_sc_dataset('training')

spot_gene_matrix_st, spot_xy_locations  = load_st_dataset()

x = sample_highest_expression(full_cell_gene_matrix_sc, 2000) # get 2000 most expressed genes

# find the common genes between the two datasets
common_genes = findCommonGenes(full_cell_gene_matrix_sc, spot_gene_matrix_st)

# extract only expression data from sc dataset that share the common genes
xprime = extractGenes(full_cell_gene_matrix_sc, common_genes)

# extract only expreesion data from st dataset that share the common genes
xbar = extractGenes(spot_gene_matrix_st, common_genes)

# step one
baseVAEParams = {}
baseVAEModel = getVAE(baseVAEParams)

baseVAEModel.train(x)

z = baseVAEModel.encode(x)

# step 2
vae1_params = {}
vae1 = getVAE(vae1_params) # VAE such as scphere or scvi

vae2Params = {}
vae2 = getVAE(vae2_params) # VAE in SEDR architecture

vaeBatchLoader = getVAEBatchesLoader(xprime, xbar, z)

descriminatorParams = {}
descriminator = getDescriminator(descriminatorParams)

for epoch in range(NUM_EPOCHS):
    
    # first train the descriminator
    
    # combine the dataset for descrimination with z's, labels
    # labels can be 0 if from sc dataset, and 1 if from st dataset
    zprime = vae1(xprime)
    zbar = vae2(xbar)
    # then shuffle the dataset
    combined_z, zlabels = combineLatentSpaceWithLabels(zprime, zbar)
    
    it = 0 
    while it < MAX_DESCR_ITERS:
        
        descriminatorOptimizer.zero_grad()
        
        predictions = descriminator(combined_z)
        accuracy = descriminatorLoss(predictions, zlabels)
        if accuracy > DESCR_ACC:
            break
        descriminatorLoss.backward()
        descriminatorOptimizer.step()
        
    # make sure this is a good enough number
    print('descriminator accuracy', accuracy)
    
    # freeze weights on descriminator and prepare for prediction
    descriminator.predict()
    
    # next iterate through the dataset in batches
    for xprime_batch, xbar_batch, z_batch in vaeBatchLoader:
        
        VAEOptimizer.zero_grad()
        
        zprime_batch = vae1.encode(xprime_batch)
        xprime_batch_reconstructed = vae1.decode(zprime_batch)
        
        zbar_batch = vae2.encode(xbar_batch)
        xbar_batch_reconstructed = vae2.decode(zbar_batch)
        
        
        loss1 = L2Loss(zprime_batch, z_batch)
        loss2 = VAELoss(xprime_batch_reconstructed, xprime_batch)
        loss3 = VAELoss(xbar_batch, xbar_batch_reconstructed)
        
        combined_zbatches, zbatch_labels = combineLatentSpaceWithLabels(zprime_batch, zbar_batch)
        classifications = descriminator(combined_zbatches)
        
        loss4 = GANLoss(classifications, zbatch_labels)
        
        total_batch_loss = 0
        for weight, loss in zip(loss_weights, [loss1, loss2, loss3, loss4]):
            total_batch_loss += weight*loss

        total_batch_loss.backward()    
        VAEOptimizer.step()
        
    
# fix the weights on vae
vae2.predict()

# step 3, train VGAE on spatial data
vgae_params = {}
vgae = getVGAE(vgae_params) # build the vgae
vgaeBatchLoader = getVGAEBatchLoader(xbar, x_st)

for epoch in (VGAE_TRAIN_EPOCHS):
    print('starting epcoh', epoch, 'of VGAE training')
    
    for xbar_batch, x_st_batch in vgaeBatchLoader:
        VGAEOptimizer.zero_grad()
        x_st_batch = vgae.construct_graph(spot_xy_locations)

        z_st_batch = vgae(x_st_batch)
        zbar_batch = vae2.encode(xbar_batch)

        z_combined_batch = torch.concat(zbar_batch, z_st_batch)
        
        xbar_batch_reconstructed, x_st_batch_reconstructed = vgae(z_combined_batch)
        
        loss = VGAELoss(xbar_batch, x_st_batch_reconstructed, x_st_batch, x_st_batch_reconstructed)
        
        loss.backward() 
        VGAEOptimizer.step()
        
        

# prediction step
new_cell_gene_matrix = load_sc_dataset('prediction_dataset')





