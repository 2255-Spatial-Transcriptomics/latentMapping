import scanpy as sc
import scvi
from rich import print
from scib_metrics.benchmark import Benchmarker
from scvi.model.utils import mde
from scvi_colab import install

adata = sc.read(
    "data/lung_atlas.h5ad",
    backup_url="https://figshare.com/ndownloader/files/24539942",
)

adata.raw = adata  # keep full dimension safe
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=100,
    layer="counts",
    batch_key="batch",
    subset=True,
)

scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
vae.train(max_epochs=10)

adata.obsm["X_scVI"] = vae.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata)

adata.obsm["X_mde"] = mde(adata.obsm["X_scVI"])

sc.pl.embedding(
    adata,
    basis="X_mde",
    color=["batch", "leiden"],
    frameon=False,
    ncols=1,
)

sc.pl.embedding(adata, basis="X_mde", color=["cell_type"], frameon=False, ncols=1)