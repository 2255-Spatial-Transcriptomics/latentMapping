
### General Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

### Scphere Imports
sys.path.insert(0, os.getcwd())
from VAE.scphere.util.util import read_mtx
from VAE.scphere.util.trainer import Trainer
from VAE.scphere.model.vae import SCPHERE
from VAE.scphere.util.plot import plot_trace

### SEDR Imports
import torch
import anndata
import scanpy as sc
from VGAE.src.graph_func import graph_construction
from VGAE.src.utils_func import mk_dir, adata_preprocess, load_visium_sge
from VGAE.src.SEDR_train import SEDR_Train

# Scanpy Configurations
sc.settings.figdir = './output/figures/'
sc.settings.writedir = './output/write/'
sc.settings.cachedir = './output/cache/'
sc.settings.datasetdir = './output/data/'

# Torch Configurations
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)

