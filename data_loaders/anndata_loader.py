import scvi
import scanpy as sc
def loadSCDataset():
    '''placeholder for now'''
    # adata = scvi.data.synthetic_iid()
    # adata = sc.read(
    #     "data/lung_atlas.h5ad",
    #     backup_url="https://figshare.com/ndownloader/files/24539942",
    # )
    
    adata_sc_read = sc.read_h5ad("data/sc_top2k_genes.h5ad")

    adata_sc_read.layers["counts"] = adata_sc_read.X.copy()  # preserve counts
    sc.pp.normalize_total(adata_sc_read, target_sum=1e4)
    sc.pp.log1p(adata_sc_read)
    adata_sc_read.raw = adata_sc_read  # freeze the state in `.raw`

    sc.pp.filter_genes(adata_sc_read, min_counts=3)
    sc.pp.filter_cells(adata_sc_read, min_counts=3)

    sc.pp.highly_variable_genes(
        adata_sc_read,
        n_top_genes=2000,
        subset=True,
        layer="counts",
        flavor="seurat_v3"
    )

    sc.pp.filter_genes(adata_sc_read, min_counts=3)
    sc.pp.filter_cells(adata_sc_read, min_counts=3)

    return adata_sc_read

def loadSTDataset():
    adata_st_read = sc.read_h5ad("data/st_top5k_genes.h5ad")

    adata_st_read.layers["counts"] = adata_st_read.X.copy()  # preserve counts
    sc.pp.normalize_total(adata_st_read, target_sum=1e4)
    sc.pp.log1p(adata_st_read)
    adata_st_read.raw = adata_st_read  # freeze the state in `.raw`

    sc.pp.filter_genes(adata_st_read, min_counts=3)
    sc.pp.filter_cells(adata_st_read, min_counts=3)

    sc.pp.highly_variable_genes(
        adata_st_read,
        n_top_genes=2000,
        subset=True,
        layer="counts",
        flavor="seurat_v3"
    )

    sc.pp.filter_genes(adata_st_read, min_counts=3)
    sc.pp.filter_cells(adata_st_read, min_counts=3)

    return adata_st_read
