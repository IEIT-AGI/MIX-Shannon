import torch
from typing import Any, List, Union, Dict, Callable, Tuple, Optional
from pytorch_lightning import LightningModule
from src import utils
from ..evaluations import AREEvaluator, AREEvaluator1
from src.datamodules.dataset_field import AIVQA
from src.models.builder import build_encoder, build_decoder
from torch.nn import functional as F
from torch import nn, Tensor
from loguru import logger

py_logger = utils.get_logger(__name__)


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super(CrossAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # self.linear = nn.Linear(embed_dim, dim_feedforward)
        self.norm1c = nn.LayerNorm(embed_dim)
        self.dropout1c = nn.Dropout(dropout)
        # self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value, key_padding_mask):
        # q_c = k_c = self.with_pos_embed(memory_cat, pos_cat)
        memory_cat_p, _ = self.self_attn(query=query, key=key, value=value, key_padding_mask=key_padding_mask)
        memory_cat = value + self.dropout1c(memory_cat_p)
        memory_cat = self.norm1c(memory_cat)

        # return memory_cat_p

        return memory_cat


class AREModule(LightningModule):
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

    # REQUIRED_FIELD = ["image", "question", "answer_label", "fact_label", "boxes", "image_id", "positive_map"]
    # VALIDATION_FIELD = ["answer", "answer_label", "fact_label", "orig_size", "image_id", "boxes"]
    # GROUNDING_FIELD = ["orig_size", "image_id", "boxes"]

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.build_model()
        self.init_weight()
        self.evaluator = AREEvaluator(self.hparams.evaluate)

    def init_weight(self) -> None:
        from torch import nn
        if getattr(self.hparams, "is_init_parameters", False):
            [nn.init.xavier_uniform_(p) for p in self.parameters() if p.dim() > 1]
            # for p in self.parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_uniform_(p)

    def build_model(self):  # TODO
        self.encoder = build_encoder(cfg=self.hparams.model.are_encoder)
        self.explaining_decoder = build_decoder(cfg=self.hparams.model.explaining_decoder)
        self.reasoning_decoder = build_decoder(cfg=self.hparams.model.reasoning_decoder)

    def step(self, batch: Any) -> Tuple[Dict, Dict]:
        are_encoder_rst = self.encoder(batch["image"], batch["question"])
        grd_rst = self.explaining_decoder(are_encoder_rst["explaining_encoder"], batch)  # object detect
        reasoning_decoder_rst = self.reasoning_decoder(batch, are_encoder_rst["reasoning_encoder"])
        return grd_rst, reasoning_decoder_rst

    def training_step(self, batch: Any, batch_idx: int) -> Dict:
        _batch = {k: v for k, v in batch.items() if k in AIVQA.dataset_field}
        grd_output, reasoning_output = self.step(_batch)

        # output
        loss_output = grd_output
        loss_output.update(reasoning_output)
        loss_output = {k: v for k, v in loss_output.items() if k.startswith("loss")}
        loss = torch.sum(torch.vstack([v for v in loss_output.values()]))
        loss_output.update({"loss_sum": loss})

        # import pprint  # TODO
        # pprint.pprint({"loss_sum": loss})
        for name, val in loss_output.items():
            if name == "loss_sum":
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=True)
            else:
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:
        _batch = {k: v for k, v in batch.items() if k in AIVQA.dataset_field}
        grd_output, reasoning_output = self.step(_batch)

        # output
        output = {k: v for k, v in batch.items() if k in AIVQA.validation_field}
        output.update(grd_output)
        output.update(reasoning_output)

        return output

    @staticmethod
    def merge_data(outputs: List[Any]) -> Dict:  # todo
        rst = {}

        # add_tensor: Callable = lambda k, v: torch.cat([rst[k], v]) if k in rst else v
        # add_str: Callable = lambda k, v: rst[k].extend(v) if k in rst else rst[k] = v

        def add_tensor(k, v):
            if k in rst:
                rst[k] = torch.cat([rst[k], v])
            else:
                rst[k] = v

        def add_str(k, v):
            if k in rst:
                rst[k].extend(v)
            else:
                rst[k] = v

        for output in outputs:
            for k, v in output.items():
                if isinstance(v, list):
                    if isinstance(v[0], str):
                        add_str(k, v)
                    elif isinstance(v[0], torch.Tensor):
                        add_tensor(k, torch.vstack(v))
                elif isinstance(v, torch.Tensor):
                    add_tensor(k, v)

        return rst

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        data = self.merge_data(outputs)
        predict_grd = {"pred_logics": data.pop("pred_logics"), "pred_boxes": data.pop("pred_boxes")}
        metrics = self.evaluator(predict_event=data.pop("predict_event"),
                                 predict_answer=data.pop("predict_answer"),
                                 predict_grd=predict_grd,
                                 target=data)
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

        from src.optimizer import build_optimizer

        model_params = self.set_module_lr()
        optimizer_cfg = dict(self.hparams.optimizer_params)
        optimizer_cfg["params"] = model_params
        return build_optimizer(optimizer_cfg)

    def set_module_lr(self) -> Union[None, List[Dict]]:
        param_cfg = self.hparams.optimizer_params.pop("paramwise_cfg", None)
        if param_cfg is None:
            return None

        model_key = []
        model_parameters = []

        # set encoder an explaining_decoder-grd_decoder  learning rates
        for name, val in param_cfg.items():
            module_name: str = val.pop('name_in_model', name)
            for key, lr in val.items():
                m_key = f'{module_name}.{key}'
                model_key.append(m_key)
                params = [p for n, p in self.named_parameters() if m_key in n and p.requires_grad]
                model_parameters.append({"params": params, "lr": lr.lr})

        is_include: Callable = lambda _name: any([k in _name for k in model_key])
        other_params = [p for n, p in self.named_parameters() if not is_include(n) and p.requires_grad]
        model_parameters.append({"params": other_params, "lr": self.hparams.optimizer_params.lr})

        return model_parameters

    def load_pretrain_weight(self, file_path: str):
        self.load_state_dict(torch.load(file_path)["state_dict"], strict=True)

    test_step = validation_step
    test_epoch_end = validation_epoch_end


class AREModule1(LightningModule):
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

    # REQUIRED_FIELD = ["image", "question", "answer_label", "fact_label", "boxes", "image_id", "positive_map"]
    # VALIDATION_FIELD = ["answer", "answer_label", "fact_label", "orig_size", "image_id", "boxes"]
    # GROUNDING_FIELD = ["orig_size", "image_id", "boxes"]

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.build_model()
        self.init_weight()
        self.evaluator = AREEvaluator1(self.hparams.evaluate)

    def init_weight(self) -> None:
        from torch import nn
        if getattr(self.hparams, "is_init_parameters", False):
            [nn.init.xavier_uniform_(p) for p in self.parameters() if p.dim() > 1]

    def build_model(self):  # TODO
        self.encoder = build_encoder(cfg=self.hparams.model.are_encoder)
        self.explaining_decoder = build_decoder(cfg=self.hparams.model.explaining_decoder)
        self.reasoning_decoder = build_decoder(cfg=self.hparams.model.reasoning_decoder)

        self.add_grd_no_gate = getattr(self.hparams.model, "add_grd_no_gate", False)
        if self.add_grd_no_gate:
            input_feat_size, output_feat_size, dropout = 256, 2, 0.1
            num_heads = 4
            self.answer_attention_fun: Callable = CrossAttention(input_feat_size, num_heads, dropout)
            self.event_attention_fun: Callable = CrossAttention(input_feat_size, num_heads, dropout)

    def add_grd_fusion(self, are_encoder_rst, grd_rst):

        def get_grd_label_score():
            pred_logics = grd_rst.get("pred_logics")
            prob = F.softmax(pred_logics, -1)
            scores = 1 - prob[:, :, -1]
            prob_obj_idx = scores.argmax(dim=-1)
            return torch.vstack([prob[i, idx, :] for i, idx in enumerate(prob_obj_idx)])

        def get_grd_label_score_topk(topk=3):
            pred_logics = grd_rst.get("pred_logics")
            prob = F.softmax(pred_logics, -1)
            scores = 1 - prob[:, :, -1]
            _, prob_obj_topk_idx = torch.topk(scores, topk)

            grd_label_score = []
            for idx, topk_idx in enumerate(prob_obj_topk_idx):
                grd_label_score.append(torch.reshape(prob[idx, topk_idx, :], (-1,)).unsqueeze(0))
            grd_label_score = torch.vstack(grd_label_score)

            return grd_label_score

        if self.add_grd_no_gate:
            grd_label_feat = get_grd_label_score()
            event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
            event_feat_attn = self.event_attention_fun(query=grd_label_feat.unsqueeze(1),
                                                       key=event_feat.unsqueeze(1),
                                                       value=event_feat.unsqueeze(1),
                                                       key_padding_mask=None)
            are_encoder_rst['reasoning_encoder']['event_feature'] = event_feat_attn.squeeze(1)

            knowledge_feat = are_encoder_rst['reasoning_encoder']['knowledge_feature']
            knowledge_feat_attn = self.answer_attention_fun(query=grd_label_feat.unsqueeze(1),
                                                            key=knowledge_feat.unsqueeze(1),
                                                            value=knowledge_feat.unsqueeze(1),
                                                            key_padding_mask=None)
            are_encoder_rst['reasoning_encoder']['knowledge_feature'] = knowledge_feat_attn.squeeze(1)

    def step(self, batch: Any) -> Tuple[Dict, Dict, torch.Tensor]:
        are_encoder_rst = self.encoder(batch["image"], batch["question"])
        grd_rst = self.explaining_decoder(are_encoder_rst["explaining_encoder"], batch)  # object detect
        gate_logics = self.add_grd_fusion(are_encoder_rst, grd_rst)
        reasoning_decoder_rst = self.reasoning_decoder(batch, are_encoder_rst["reasoning_encoder"])
        return grd_rst, reasoning_decoder_rst, gate_logics

    def training_step(self, batch: Any, batch_idx: int) -> Dict:
        _batch = {k: v for k, v in batch.items() if k in AIVQA.dataset_field}
        grd_output, reasoning_output, gate_logics = self.step(_batch)

        # output
        loss_output = grd_output
        loss_output.update(reasoning_output)
        loss_output = {k: v for k, v in loss_output.items() if k.startswith("loss")}
        loss = torch.sum(torch.vstack([v for v in loss_output.values()]))
        loss_output.update({"loss_sum": loss})

        for name, val in loss_output.items():
            if name == "loss_sum":
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=True)
            else:
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:
        _batch = {k: v for k, v in batch.items() if k in AIVQA.dataset_field}
        grd_output, reasoning_output, _ = self.step(_batch)

        # output
        output = {k: v for k, v in batch.items() if k in AIVQA.validation_field}
        output.update(grd_output)
        output.update(reasoning_output)

        return output

    @staticmethod
    def merge_data(outputs: List[Any]) -> Dict:  # todo
        rst = {}

        def add_tensor(k, v):
            if k in rst:
                rst[k] = torch.cat([rst[k], v])
            else:
                rst[k] = v

        def add_str(k, v):
            if k in rst:
                rst[k].extend(v)
            else:
                rst[k] = v

        for output in outputs:
            for k, v in output.items():
                if isinstance(v, list):
                    if isinstance(v[0], str):
                        add_str(k, v)
                    elif isinstance(v[0], torch.Tensor):
                        add_tensor(k, torch.vstack(v))
                elif isinstance(v, torch.Tensor):
                    add_tensor(k, v)

        return rst

    def validation_epoch_end(self, outputs: List[Any]) -> None:

        data = self.merge_data(outputs)
        predict_grd = {"pred_logics": data.pop("pred_logics"), "pred_boxes": data.pop("pred_boxes")}

        metrics = self.evaluator(predict_event=data.pop("predict_event"),
                                 predict_answer=data.pop("predict_answer"),
                                 predict_relation=data.pop("predict_relation", None),
                                 predict_grd=predict_grd,
                                 target=data)

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

        from src.optimizer import build_optimizer

        model_params = self.set_module_lr()
        optimizer_cfg = dict(self.hparams.optimizer_params)
        optimizer_cfg["params"] = model_params
        return build_optimizer(optimizer_cfg)

    def set_module_lr(self) -> Union[None, List[Dict]]:
        param_cfg = self.hparams.optimizer_params.pop("paramwise_cfg", None)
        if param_cfg is None:
            return None

        model_key = []
        model_parameters = []

        # set encoder an explaining_decoder-grd_decoder  learning rates
        for name, val in param_cfg.items():
            module_name: str = val.pop('name_in_model', name)
            for key, lr in val.items():
                m_key = f'{module_name}.{key}'
                model_key.append(m_key)
                params = [p for n, p in self.named_parameters() if m_key in n and p.requires_grad]
                model_parameters.append({"params": params, "lr": lr.lr})

        is_include: Callable = lambda _name: any([k in _name for k in model_key])
        other_params = [p for n, p in self.named_parameters() if not is_include(n) and p.requires_grad]
        model_parameters.append({"params": other_params, "lr": self.hparams.optimizer_params.lr})

        return model_parameters

    def load_pretrain_weight(self, file_path: str):
        self.load_state_dict(torch.load(file_path)["state_dict"], strict=False)

    test_step = validation_step
    test_epoch_end = validation_epoch_end
