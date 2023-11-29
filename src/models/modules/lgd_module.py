import torch
from typing import Any, List, Union, Dict, Callable, Tuple, Optional
from pytorch_lightning import LightningModule
from src import utils
from ..evaluations import AREEvaluator
from src.datamodules.dataset_field import AIVQA
from src.models.builder import build_encoder, build_decoder
from torch.nn import functional as F
from torch import nn, Tensor
from loguru import logger

py_logger = utils.get_logger(__name__)

from stable_diffusion.ldm.util import instantiate_from_config

class LGDModule(LightningModule): 
    """
    LGDModule: latent guided diffusion
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

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.build_model()
        self.init_weight()
        self.evaluator = AREEvaluator(self.hparams.evaluate)

        self.right_label = 0  # del

    def init_weight(self) -> None:
        from torch import nn
        if getattr(self.hparams, "is_init_parameters", False):
            [nn.init.xavier_uniform_(p) for p in self.parameters() if p.dim() > 1]
            # for p in self.parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_uniform_(p)

    def build_model(self):  # TODO
        self.llm_encoder = self._instantiate_llm_encoder(self.hparams.model.llm_encoder)  # llm with adpater
        self.img_generator = self._instantiate_image_decder(self.hparams.model.img_generator)
        self.latent_sapce_adapter= self._instantiate_latent_sapce_adapter(self.hparams.model.latent_space_adapter)
        a = 1

    
    def _instantiate_llm_encoder(self,cfg):
        from copy import deepcopy
        model = instantiate_from_config(cfg)
        return model


    def _instantiate_image_decder(self,cfg):
        pass

    def _instantiate_latent_sapce_adapter(self,cfg):
        pass
        

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

        if self.has_grd_fusion:
            # fact_fusion_token_embedding_repeats = fact_fusion_token_embedding.unsqueeze(1).repeat(1,100,1)
            # fusion_feats = self.fusion_fc(torch.cat([fact_fusion_token_embedding_repeats, fact_fusion_token_embedding_repeats, outputs_class[-1]], dim=-1))
            event_feat = are_encoder_rst['reasoning_encoder']['event_feature'].unsqueeze(1).repeat(1, 100, 1)
            event_fusion_grd = torch.cat([event_feat, event_feat, grd_rst['pred_logics']], dim=-1)
            idx = _get_src_permutation_idx(grd_rst['match_indices'])
            are_encoder_rst['reasoning_encoder']['event_feature'] = self.grd_fusion(event_fusion_grd)[idx]

        if self.add_grd_feature:
            grd_label_feat = get_grd_label_score()

            event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
            event_fusion_grd = torch.cat([event_feat, grd_label_feat], dim=-1)
            are_encoder_rst['reasoning_encoder']['event_feature'] = self.grd_event_fusion(event_fusion_grd)

            knowledge_feat = are_encoder_rst['reasoning_encoder']['knowledge_feature']
            knowledge_fusion_grd = torch.cat([knowledge_feat, grd_label_feat], dim=-1)
            are_encoder_rst['reasoning_encoder']['knowledge_feature'] = self.grd_answer_fusion(knowledge_fusion_grd)

        if self.add_grd_feature_gate:
            grd_label_feat = get_grd_label_score()

            gate_feat = are_encoder_rst['reasoning_encoder']['gate_feature'].mean(dim=0)
            gate_logics = self.gate_fun(gate_feat)
            gate_weight = F.softmax(gate_logics, dim=-1)

            grd_weight_feat1 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(gate_weight[:, 0])])
            event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
            event_fusion_grd = torch.cat([event_feat, grd_weight_feat1], dim=-1)
            are_encoder_rst['reasoning_encoder']['event_feature'] = self.grd_event_fusion(event_fusion_grd)

            grd_weight_feat2 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(gate_weight[:, 1])])
            knowledge_feat = are_encoder_rst['reasoning_encoder']['knowledge_feature']
            knowledge_fusion_grd = torch.cat([knowledge_feat, grd_weight_feat2], dim=-1)
            are_encoder_rst['reasoning_encoder']['knowledge_feature'] = self.grd_answer_fusion(knowledge_fusion_grd)

        if self.add_grd_feature_gate_attention:
            grd_label_feat = get_grd_label_score()
            gate_feat = are_encoder_rst['reasoning_encoder']['gate_feature'].mean(dim=0)
            gate_logics = self.gate_fun(gate_feat)
            gate_weight = F.softmax(gate_logics, dim=-1)

            grd_weight_feat1 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(gate_weight[:, 0])])
            grd_weight_feat2 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(gate_weight[:, 1])])

            event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
            # forward(self, query, key, value, key_padding_mask):
            event_feat_attn = self.event_attention_fun(query=grd_weight_feat1.unsqueeze(1),
                                                       key=event_feat.unsqueeze(1),
                                                       value=event_feat.unsqueeze(1),
                                                       key_padding_mask=None)
            are_encoder_rst['reasoning_encoder']['event_feature'] = event_feat_attn.squeeze(1)

            knowledge_feat = are_encoder_rst['reasoning_encoder']['knowledge_feature']
            knowledge_feat_attn = self.answer_attention_fun(query=grd_weight_feat2.unsqueeze(1),
                                                            key=knowledge_feat.unsqueeze(1),
                                                            value=knowledge_feat.unsqueeze(1),
                                                            key_padding_mask=None)
            are_encoder_rst['reasoning_encoder']['knowledge_feature'] = knowledge_feat_attn.squeeze(1)

        if self.add_grd_no_gate:
            grd_label_feat = get_grd_label_score()
            event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
            # forward(self, query, key, value, key_padding_mask):
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

        if self.add_grd_feature_gate_attention_gate_fun:  # no mean
            grd_label_feat = get_grd_label_score()
            gate_feat = are_encoder_rst['reasoning_encoder']['gate_feature'].mean(dim=0)
            gate_logics = self.gate_fun(gate_feat)
            gate_weight = F.softmax(gate_logics, dim=-1)

            # is_head_tail:0  is_head:1 is_tail:2 is_no_head_tail:3
            _gate_weight = torch.zeros([gate_weight.shape[0], 2], dtype=gate_weight.dtype, device=gate_weight.device)
            arg_max = gate_weight.argmax(dim=-1)

            if False:
                for idx, max_idx in enumerate(arg_max):
                    if max_idx.item() == 0:
                        _gate_weight[idx][0] = gate_weight[idx][0] / 2 + gate_weight[idx][1]
                        _gate_weight[idx][1] = gate_weight[idx][0] / 2 + gate_weight[idx][2]
                    elif max_idx.item() == 3:
                        _gate_weight[idx][0] = gate_weight[idx][1]  # head
                        _gate_weight[idx][1] = gate_weight[idx][2]  # tail
                    else:
                        _gate_weight[idx][0] = gate_weight[idx][1]
                        _gate_weight[idx][1] = gate_weight[idx][2]
                logger.warning(f"_gate_weight:{_gate_weight}")
                logger.warning(f"gate_weight:{gate_weight}")

                grd_weight_feat1 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(_gate_weight[:, 0])])
                grd_weight_feat2 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(_gate_weight[:, 1])])

                event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
                # forward(self, query, key, value, key_padding_mask):
                event_feat_attn = self.event_attention_fun(query=grd_weight_feat1.unsqueeze(1),
                                                           key=event_feat.unsqueeze(1),
                                                           value=event_feat.unsqueeze(1),
                                                           key_padding_mask=None)
                are_encoder_rst['reasoning_encoder']['event_feature'] = event_feat_attn.squeeze(1)

                knowledge_feat = are_encoder_rst['reasoning_encoder']['knowledge_feature']
                knowledge_feat_attn = self.answer_attention_fun(query=grd_weight_feat2.unsqueeze(1),
                                                                key=knowledge_feat.unsqueeze(1),
                                                                value=knowledge_feat.unsqueeze(1),
                                                                key_padding_mask=None)
                are_encoder_rst['reasoning_encoder']['knowledge_feature'] = knowledge_feat_attn.squeeze(1)
                return gate_logics
            else:
                _gate_weight = torch.ones([gate_weight.shape[0], 2], dtype=gate_weight.dtype, device=gate_weight.device)
                grd_weight_feat1 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(_gate_weight[:, 0])])
                grd_weight_feat2 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(_gate_weight[:, 1])])

                event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
                # forward(self, query, key, value, key_padding_mask):
                event_feat_attn = self.event_attention_fun(query=grd_weight_feat1.unsqueeze(1),
                                                           key=event_feat.unsqueeze(1),
                                                           value=event_feat.unsqueeze(1),
                                                           key_padding_mask=None)

                knowledge_feat = are_encoder_rst['reasoning_encoder']['knowledge_feature']
                knowledge_feat_attn = self.answer_attention_fun(query=grd_weight_feat2.unsqueeze(1),
                                                                key=knowledge_feat.unsqueeze(1),
                                                                value=knowledge_feat.unsqueeze(1),
                                                                key_padding_mask=None)

                event_feat_attn = event_feat_attn.squeeze(1)
                knowledge_feat_attn = knowledge_feat_attn.squeeze(1)

                for idx, max_idx in enumerate(arg_max):
                    if max_idx.item() == 0:
                        pass
                    elif max_idx.item() == 1:  # head
                        knowledge_feat_attn[idx] = knowledge_feat[idx]
                    elif max_idx.item() == 2:  # taild
                        event_feat_attn[idx] = event_feat[idx]
                    else:
                        knowledge_feat_attn[idx] = knowledge_feat[idx]
                        event_feat_attn[idx] = event_feat[idx]

                are_encoder_rst['reasoning_encoder']['event_feature'] = event_feat_attn
                are_encoder_rst['reasoning_encoder']['knowledge_feature'] = knowledge_feat_attn

            return gate_logics

        if self.add_grd_multi_objects:
            grd_label_feat = get_grd_label_score_topk(self.topk_objects)
            grd_label_feat = self.multi_obj_fun(grd_label_feat)  # Feature_reszie

            gate_feat = are_encoder_rst['reasoning_encoder']['gate_feature'].mean(dim=0)
            gate_logics = self.gate_fun(gate_feat)
            gate_weight = F.softmax(gate_logics, dim=-1)

            grd_weight_feat1 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(gate_weight[:, 0])])
            grd_weight_feat2 = torch.vstack([grd_label_feat[i, :] * w for i, w in enumerate(gate_weight[:, 1])])

            event_feat = are_encoder_rst['reasoning_encoder']['event_feature']
            # forward(self, query, key, value, key_padding_mask):
            event_feat_attn = self.event_attention_fun(query=grd_weight_feat1.unsqueeze(1),
                                                       key=event_feat.unsqueeze(1),
                                                       value=event_feat.unsqueeze(1),
                                                       key_padding_mask=None)
            are_encoder_rst['reasoning_encoder']['event_feature'] = event_feat_attn.squeeze(1)

            knowledge_feat = are_encoder_rst['reasoning_encoder']['knowledge_feature']
            knowledge_feat_attn = self.answer_attention_fun(query=grd_weight_feat2.unsqueeze(1),
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

        if self.fixed_grd:
            remove_key = []
            for k, v in grd_output.items():
                if k.startswith("loss"):
                    remove_key.append(k)
            for k in remove_key:
                grd_output.pop(k)
        # output
        loss_output = grd_output
        loss_output.update(reasoning_output)

        if self.add_grd_feature_gate_attention_gate_fun:
            gate_label_loss = self.gate_fun_loss(gate_logics, torch.vstack(batch["gate_label"]))
            loss_output["loss_gate_label"] = gate_label_loss * 0.5

        loss_output = {k: v for k, v in loss_output.items() if k.startswith("loss")}
        loss = torch.sum(torch.vstack([v for v in loss_output.values()]))
        loss_output.update({"loss_sum": loss})

        # import pprint  # TODO
        # pprint.pprint({"loss_sum": loss})
        # pprint.pprint(f"loss_output:{loss_output}")
        for name, val in loss_output.items():
            if name == "loss_sum":
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=True)
            else:
                self.log(f"train/{name}", val, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:
        _batch = {k: v for k, v in batch.items() if k in AIVQA.dataset_field}
        grd_output, reasoning_output, _ = self.step(_batch)

        # debug
        # gate_weight = F.softmax(_, dim=-1)
        # gate_label = torch.vstack(batch["gate_label"])
        # for idx in range(gate_weight.shape[0]):
        #     logger.error(f"gate_weight[{idx}]=:{gate_weight[idx]}")
        #     logger.error(f"gate_label[{idx}]=:{gate_label[idx]}")
        #     if gate_weight[idx].argmax() == gate_label[idx].argmax():
        #         self.right_label += 1

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

        # debug
        # logger.error(f"right_label={self.right_label}  prob:{self.right_label / len(outputs)}")

        data = self.merge_data(outputs)
        predict_grd = {"pred_logics": data.pop("pred_logics"),
                       "pred_boxes": data.pop("pred_boxes")}

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
        # import hydra
        # from omegaconf import open_dict, OmegaConf
        # OmegaConf.create()
        from src.optimizer import build_optimizer

        # optim_type: str = self.hparams.optimizer_params.pop("type", "Adam")
        # assert optim_type == "Adam"
        # kwargs = dict(self.hparams.optimizer_params)
        # kwargs["params"] = model_parameters
        #
        # return torch.optim.Adam(**kwargs)

        model_params = self.set_module_lr()
        optimizer_cfg = dict(self.hparams.optimizer_params)
        optimizer_cfg["params"] = model_params
        return build_optimizer(optimizer_cfg)

    # def set_module_lr(self) -> Union[None, List[Dict]]:
    #     param_cfg = self.hparams.optimizer_params.pop("paramwise_cfg", None)
    #     if param_cfg is None:
    #         return None
    #
    #     model_key = []
    #     model_parameters = []
    #     # encoder
    #     for key in param_cfg.are_encoder.keys():
    #         if key == "visual_encoder":
    #             _key = '.'.join([key, "encode"])
    #             m_key = f"encoder.{_key}"
    #         else:
    #             m_key = f"encoder.{key}"
    #
    #         model_key.append(m_key)
    #         m_lr = param_cfg.are_encoder[key].lr
    #         params = [p for n, p in self.named_parameters() if m_key in n and p.requires_grad]
    #         model_parameters.append({"params": params, "lr": m_lr})
    #
    #     # explaining_decoder-grd_decoder
    #     for key in param_cfg.explaining_decoder.keys():
    #         m_key = f"explaining_decoder.{key}"
    #         model_key.append(m_key)
    #         m_lr = param_cfg.explaining_decoder[key].lr
    #         params = [p for n, p in self.named_parameters() if m_key in n and p.requires_grad]
    #         model_parameters.append({"params": params, "lr": m_lr})
    #
    #     is_include: Callable = lambda name: any([k in name for k in model_key])
    #     other_params = [p for n, p in self.named_parameters() if not is_include(n) and p.requires_grad]
    #     model_parameters.append({"params": other_params, "lr": self.hparams.optimizer_params.lr})
    #
    #     return model_parameters

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
    # def test_step(self, batch: Any, batch_idx: int) -> Dict:
    #     return self.validation_step(batch, batch_idx)
    #
    # def test_epoch_end(self, outputs: List[Any]) -> None:
    #     return self.validation_epoch_end(outputs)
