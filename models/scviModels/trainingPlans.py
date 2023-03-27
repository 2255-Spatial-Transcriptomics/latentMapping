from typing import Optional, Union

import numpy as np

from scvi.dataloaders import DataSplitter
from scvi.train import TrainingPlan, TrainRunner

class baseVAETrainingPlan(TrainingPlan):
    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step for the model."""
        if "kl_weight" in self.loss_kwargs:
            kl_weight = self.kl_weight
            self.loss_kwargs.update({"kl_weight": kl_weight})
            self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")

        return scvi_loss.loss
        

class baseVAEUnsupervisedTrainingMixin:
    """General purpose unsupervised train method."""

    _data_splitter_cls = DataSplitter
    _training_plan_cls = baseVAETrainingPlan
    _train_runner_cls = TrainRunner

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
    

class scVAETrainingPlan(TrainingPlan):
    """Training plan for vae2 (scvi)"""
    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Training step for the model."""
        if "kl_weight" in self.loss_kwargs:
            kl_weight = self.kl_weight
            self.loss_kwargs.update({"kl_weight": kl_weight})
            self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")

        return scvi_loss.loss
    
class scVAEUnsupervisedTrainingMixin:
    """Training method for VAE2 (from scvi) """

    _data_splitter_cls = DataSplitter
    _training_plan_cls = scVAETrainingPlan
    _train_runner_cls = TrainRunner

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
