from typing import Any, List, Dict, Callable, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from src import utils
from itertools import accumulate
from ..evaluations import RecipeRetrievalEvaluator

from ..builder import build_encoder, build_head

py_logger = utils.get_logger(__name__)


class HgCAnModule(LightningModule):
    """
    HgCAn: heterogeneous graph combination with attention network
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    # HEADER_TYPE = {"pairwise_header": PairwiseHeader, "shuffle_header": ShuffleHeader}

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.build_model()
        self.evaluator = RecipeRetrievalEvaluator(self.hparams.evaluate)

    def build_model(self):  # TODO
        # self.recipe_encoder = RecipeEncoder(cfg=self.hparams.model.recipe_encoder,
        #                                     is_shuffle=self.hparams.model.is_shuffle)
        self.recipe_encoder = build_encoder(self.hparams.model.recipe_encoder)
        self.csi_encoder = build_encoder(self.hparams.model.csi_encoder)  # csi:cooking steps image encoder
        for header, args in self.hparams.model.header.items():
            setattr(self, header, build_head(args))

    def step(self, batch: Dict) -> Tuple[Dict, Dict, torch.Tensor]:
        single_recipe_img_nums = batch["single_recipe_img_nums"]
        recipe_img_pos = list(accumulate([0] + single_recipe_img_nums))
        batch_recipe_img_pos = [[start_idx, end_idx] for start_idx, end_idx in
                                zip(recipe_img_pos[:-1], recipe_img_pos[1:])]

        recipe_encoder_rst = self.recipe_encoder(ingredients_feature=batch["ingredients_feature"],
                                                 instruction_feature=batch["instruction_feature"],
                                                 heterogeneous_graph=batch["graph"],
                                                 title_feature=batch["title_feature"],
                                                 recipe_img_pos=batch_recipe_img_pos)
        csi_encoder_rst = self.csi_encoder(img_feature=batch["img_feature"], recipe_img_pos=batch_recipe_img_pos)
        labels = batch["labels"]

        return recipe_encoder_rst, csi_encoder_rst, labels

    def training_step(self, batch: Any, batch_idx: int) -> Dict:
        recipe_encoder_rst, csi_encoder_rst, labels = self.step(batch)

        get_require_field: Callable = lambda dict_data, field: {k: v for k, v in dict_data.items() if
                                                                k.find(field) != -1}
        shuffle_loss = self.shuffle_header(csi_info=get_require_field(csi_encoder_rst, "shuffle"),
                                           instruction_info=get_require_field(recipe_encoder_rst, "shuffle"))
        gph_loss = self.pairwise_header(title_feature=recipe_encoder_rst["title_feature"],
                                        ingredients_feature=torch.stack(recipe_encoder_rst["instruction_feature_mean"]),
                                        csi_feature=torch.stack(csi_encoder_rst["csi_feature_mean"]),
                                        labels=labels)

        # ouput
        loss_output = shuffle_loss
        loss_output.update(gph_loss)
        loss = torch.sum(torch.vstack([v for v in loss_output.values()]))
        loss_output.update({"loss_sum": loss})

        for name, val in loss_output.items():
            if name == "loss_sum":
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=True)
            else:
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:
        recipe_encoder_rst, csi_encoder_rst, labels = self.step(batch)

        # output
        output = {}
        output["csi_features"] = torch.stack(csi_encoder_rst["csi_feature_mean"])
        output["recipe_features"] = torch.stack(recipe_encoder_rst["instruction_feature_mean"]) + recipe_encoder_rst[
            "title_feature"]
        return output

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        rst = {}
        for output in outputs:
            for key, value in output.items():
                if key in rst:
                    rst[key].append(value.cpu().data.numpy())
                else:
                    rst[key] = [value.cpu().data.numpy()]
        csi_feats = np.concatenate(rst["csi_features"])
        recipe_feats = np.concatenate(rst["recipe_features"])

        metrics = self.evaluator(csi_feats=csi_feats, recipe_feats=recipe_feats)
        for metric_name, val in metrics.items():
            py_logger.info(f"{metric_name}: {val}")
            for key, value in val.items():
                self.log(f"validation/{metric_name}/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        from src.utils.thir_party_libs.optimizer import build_optimizer, build_scheduler

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer_cfg = dict(self.hparams.optimizer_params)
        optimizer_cfg["params"] = model_parameters
        optimizer = build_optimizer(optimizer_cfg)

        scheduler_cfg = dict(self.hparams.Scheduler)
        scheduler_cfg["optimizer"] = optimizer
        lr_scheduler = build_scheduler(scheduler_cfg)

        # optimizer = torch.optim.Adam(params=model_parameters,
        #                              lr=self.hparams.optimizer_params.lr)
        #
        # scheduler_kwargs = dict(self.hparams.Scheduler)
        # scheduler_kwargs["optimizer"] = optimizer
        # lr_scheduler = HgCAnScheduler(**scheduler_kwargs)

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
                }
