import torch
from torch import nn
from omegaconf import DictConfig
from typing import Dict, List, Callable
from src.models.builder import build_loss, HEADS
import hydra
from collections import OrderedDict


@HEADS.register_module()
class PairwiseHeader(nn.Module):

    def __init__(self, feat_proj: DictConfig, loss_cfg: DictConfig):
        super(PairwiseHeader, self).__init__()
        self.feat_proj = hydra.utils.instantiate(feat_proj)
        self.loss_fn = build_loss(loss_cfg)

    def forward(self, title_feature: torch.Tensor, ingredients_feature: torch.Tensor, csi_feature: torch.Tensor,
                labels: torch.Tensor) -> Dict:
        feat = torch.cat([title_feature + ingredients_feature, csi_feature]).contiguous()
        predict = self.feat_proj(feat)
        loss, _, _ = self.loss_fn(predict, labels)
        return {"TripletLoss": loss}


@HEADS.register_module()
class ShuffleHeader(nn.Module):

    def __init__(self, csi_header: DictConfig, instruction_header: DictConfig, mix_instruction_csi_header: DictConfig):
        super(ShuffleHeader, self).__init__()
        build_header: Callable = lambda cfg: nn.Sequential(
            OrderedDict(feat_proj=hydra.utils.instantiate(cfg.feat_proj), loss_fn=hydra.utils.instantiate(cfg.loss_cfg))
        )
        self.csi_header: nn.Sequential = build_header(csi_header)
        self.instruction_header: nn.Sequential = build_header(instruction_header)
        self.mix_instruction_csi_header: nn.Sequential = build_header(mix_instruction_csi_header)

    def forward(self, csi_info: Dict, instruction_info: Dict) -> Dict:
        if any(v is None for v in csi_info.values()) or any(v is None for v in instruction_info.values()):
            loss_0 = torch.tensor(0, dtype=torch.float, device=self.csi_header.feat_proj.weight.device)
            csi_loss, ins_loss, mix_csi_ins_loss = loss_0, loss_0, loss_0
        else:
            csi_shuffle_feat = csi_info["csi_shuffle_features_mean"]
            csi_shuffle_label = csi_info["csi_shuffle_features_label"]
            ins_shuffle_feat = instruction_info["instruction_shuffle_features_mean"]
            ins_shuffle_label = instruction_info["instruction_shuffle_features_label"]

            csi_shuffle_feat = torch.stack(csi_shuffle_feat, dim=0)
            ins_shuffle_feat = torch.stack(ins_shuffle_feat, dim=0)

            csi_loss = self.calculate_csi_loss(csi_shuffle_feat, csi_shuffle_label)
            ins_loss = self.calculate_instruction_loss(ins_shuffle_feat, ins_shuffle_label)
            mix_csi_ins_loss = self.calculate_mix_instruction_csi_loss(csi_feat=csi_shuffle_feat,
                                                                       csi_label=csi_shuffle_label,
                                                                       ins_feat=ins_shuffle_feat,
                                                                       ins_label=ins_shuffle_label)

        # output
        output = {
            "csi_shuffle_loss": csi_loss,
            "instruction_shuffle_loss": ins_loss,
            "mix_csi_instruction_shuffle_loss": mix_csi_ins_loss
        }
        return output

    def calculate_csi_loss(self, csi_feat: torch.Tensor, csi_label: List) -> torch.Tensor:
        predict = self.csi_header.feat_proj(csi_feat)
        label = torch.tensor(csi_label, dtype=predict.dtype, device=predict.device)
        return self.csi_header.loss_fn(predict.squeeze(dim=-1), label)

    def calculate_instruction_loss(self, ins_feat: torch.Tensor, ins_label: List) -> torch.Tensor:
        predict = self.instruction_header.feat_proj(ins_feat)
        label = torch.tensor(ins_label, dtype=predict.dtype, device=predict.device)
        return self.instruction_header.loss_fn(predict.squeeze(dim=-1), label)

    def calculate_mix_instruction_csi_loss(self, csi_feat: torch.Tensor, csi_label: List, ins_feat: torch.Tensor,
                                           ins_label: List) -> torch.Tensor:
        feat = torch.cat((ins_feat, csi_feat), dim=1)
        predict = self.mix_instruction_csi_header.feat_proj(feat)
        label = torch.tensor(csi_label) | torch.tensor(ins_label)
        label = torch.tensor(label, dtype=predict.dtype, device=predict.device)
        return self.mix_instruction_csi_header.loss_fn(predict.squeeze(dim=-1), label)
