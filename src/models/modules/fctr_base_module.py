from typing import Any, List, Optional, Union, Dict, Callable

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from typing.io import IO
from src.models.evaluations import FRECEvaluator
from src.utils.evaluations import PostProcess
from src import utils
from torch import nn
from collections import OrderedDict
from src.datamodules.dataset_field import FRECField

py_logger = utils.get_logger(__name__)


class FCTRBaselineModule(LightningModule):
    """
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

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model: nn.Sequential = self.build_model()
        self.init_weight()

        self.evaluator = FRECEvaluator(self.hparams.evaluate)

    def build_model(self) -> nn.Sequential:
        from src.models.builder import build_encoder, build_decoder

        encoder = build_encoder(self.hparams.model.fctr_encoder)
        decoder = build_decoder(self.hparams.model.fctr_decoder)

        # decoder.grd_task.model.encoder = encoder.multimodal_fusion  # share weight # todo noshare
        return nn.Sequential(OrderedDict(encoder=encoder, decoder=decoder))

    def init_weight(self) -> None:
        if getattr(self.hparams, "is_init_parameters", False):
            [nn.init.xavier_uniform_(p) for p in self.model.parameters() if p.dim() > 1]
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def training_step(self, batch: Any, batch_idx: int):
        # images, raw_sents, target_sents, rationales, targets = self.prepare_train_data(batch)
        encoder_rst = self.model.encoder(images=batch["images"],
                                         raw_sentences=batch["raw_sent"],
                                         target_sentences=batch["cor_sent"])
        decoder_rst = self.model.decoder(fctr_encoder_result=encoder_rst,
                                         rationales=batch["rationale"],
                                         target=batch)

        loss = sum([v for k, v in decoder_rst.items() if 'loss' in k])

        print(f"loss:{loss}")  # todo delete
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:
        encoder_rst = self.model.encoder(images=batch["images"],
                                         raw_sentences=batch["raw_sent"])
        decoder_rst = self.model.decoder(fctr_encoder_result=encoder_rst)

        def convert_decoder_rst() -> List:
            field = "predict_sentences"
            length = len(batch["size"])
            predict: List = []
            for idx in range(length):
                single_predict = {}
                for k, v in decoder_rst.items():
                    if field in v.keys():
                        single_predict[k] = {field: decoder_rst[k][field][idx]}
                    else:
                        single_predict[k] = {grd_k: grd_v[idx] for grd_k, grd_v in decoder_rst[k].items()}
                predict.append(single_predict)
            return predict

        target = {k: v for k, v in batch.items() if k in FRECField.validation_field}
        predict = convert_decoder_rst()
        output = {}
        for idx, name in enumerate(target.get("name")):
            output[name] = {key: target.get(key)[idx] for key in target.keys()}
            output[name].update({"predict": predict[idx]})
        return output

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        data = {k: v for output in outputs for k, v in output.items()}
        metrics = self.evaluator(data)
        for metric_name, val in metrics.items():
            py_logger.info(f"{metric_name}: {val}")
            for key, value in val.items():
                self.log(f"validation/{metric_name}/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):  # TODO
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        from src.optimizer import build_optimizer, build_scheduler

        model_parameters = self.set_module_lr()
        optimizer_cfg = dict(self.hparams.optimizer_params)
        optimizer_cfg["params"] = model_parameters
        optimizer = build_optimizer(optimizer_cfg)

        scheduler_cfg = dict(self.hparams.Scheduler)
        scheduler_cfg["optimizer"] = optimizer
        scheduler_cfg["step_size"] = self.trainer.max_epochs * len(self.trainer.datamodule.data_train)
        scheduler = build_scheduler(scheduler_cfg)

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
                }

    def set_module_lr(self) -> Union[None, List[Dict]]:
        param_cfg = self.hparams.optimizer_params.pop("paramwise_cfg", None)
        if param_cfg is None:
            return None

        model_key = []
        model_parameters = []

        for name, val in param_cfg.items():
            module_name: str = val.pop('name_in_model', name)
            for key, lr in val.items():
                m_key = f'{module_name}.{key}'
                model_key.append(m_key)
                params = [p for n, p in self.named_parameters() if m_key in n and p.requires_grad]
                # print(f"{key} {len(params)}") #todo
                model_parameters.append({"params": params, "lr": lr.lr})

        is_include: Callable = lambda _name: any([k in _name for k in model_key])
        other_params = [p for n, p in self.named_parameters() if not is_include(n) and p.requires_grad]
        model_parameters.append({"params": other_params, "lr": self.hparams.optimizer_params.lr})

        return model_parameters

    def load_pretrain_weight(self, file_path: str):
        self.load_state_dict(torch.load(file_path)["state_dict"], strict=False)
