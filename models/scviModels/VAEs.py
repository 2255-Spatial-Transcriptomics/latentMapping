# def getVAE(params):
#     '''
#     returns a Pytorch Variational Autoencoder model based on scvi
#     '''
import sys
sys.path.append('/Users/davidw/spatial-transcriptomics/latentMapping')

from typing import Optional, Sequence
import numpy as np
import scvi
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)

from scvi.model.base import BaseModelClass 
from models.scviModels.scVAEMixin import scVAEMixin
from models.scviModels.trainingPlans import baseVAEUnsupervisedTrainingMixin,scVAEUnsupervisedTrainingMixin
from scvi.module import VAE
from models.scviModels.modules import VAE2

class baseVAE(scVAEMixin, baseVAEUnsupervisedTrainingMixin, BaseModelClass):
    """
    single-cell Variational Inference [Lopez18]_.
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = VAE(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_latent: {}"
        ).format(
            n_latent,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        

class scVAE(scVAEMixin, scVAEUnsupervisedTrainingMixin, BaseModelClass):
    """
    single-cell Variational Inference [Lopez18]_.
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = VAE2(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_latent: {}"
        ).format(
            n_latent,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
    