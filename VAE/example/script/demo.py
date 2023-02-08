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

# Preparing a sparse matrix and using ~2000 variable genes for efficiency. 
# Data can be downloaded from single cell portal (login with a Google account):
# https://singlecell.broadinstitute.org/single_cell/study/SCP551/scphere#study-download

NAME = 'lung_human_ASK440'
SPLIT = False

data_dir = os.getcwd() + '/VAE/example/data/' + NAME + '/'
EPOCH = 5
TT_RATIO = 0.75

mtx = data_dir + NAME + '.mtx'
input = read_mtx(mtx)
input = input.transpose().todense()

label_filename = data_dir + NAME + '_celltype.tsv'
df = pd.read_csv(label_filename, sep='\n')
label = np.array(df.values.tolist())

nrow, ncol = input.shape

if SPLIT:
    train_size = round(nrow*TT_RATIO)
    test_size = round(nrow*(1-TT_RATIO))

    train_index = np.random.choice(nrow, train_size, replace=False)
    test_index = np.array(set(np.arange(0, nrow)) - set(train_index))

    train_x = input[train_index, :]
    test_x = input[test_index, :]

    train_y = label[train_index]
    test_y = label[test_index]

    input = train_x


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
save_path = os.getcwd() + '/VAE/example/demo-out/'

model_name = save_path + NAME + '_model_' + str(EPOCH) + 'epoch'
model.save_sess(model_name)

# embedding all the data
z_mean = model.encode(input, batch)
np.savetxt(save_path +
           NAME + '_latent_' + str(EPOCH) + 'epoch.tsv',
           z_mean)

# the log-likelihoods
ll = model.get_log_likelihood(input, batch)
np.savetxt(save_path +
           NAME + '_ll_' + str(EPOCH) + 'epoch.tsv',
           z_mean)

# Plotting log-likelihood and kl-divergence at each iteration
plot_trace([np.arange(len(trainer.status['kl_divergence']))*50] * 2,
           [trainer.status['log_likelihood'], trainer.status['kl_divergence']],
           ['log_likelihood', 'kl_divergence'])
# plt.show()

plt.savefig(save_path +
            NAME + str(EPOCH) + '_train.png')




### Predict Accuracy
# model.predict