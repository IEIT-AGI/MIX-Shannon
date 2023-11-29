import torch
from torch import nn
from omegaconf import DictConfig
from typing import Dict, List, Any, Tuple, Callable
from src.models.builder import POSITION_EMBEDDING  # todo
from collections import OrderedDict
import hydra


@POSITION_EMBEDDING.register_module()
class AttentionEmbedding(nn.Module):
    def __init__(self, img_feat_proj: DictConfig, attention_module: DictConfig):
        super(AttentionEmbedding, self).__init__()
        self.img_feat_proj = hydra.utils.instantiate(img_feat_proj)

        create_attention_layer: Callable = lambda fc, af: nn.Sequential(OrderedDict(fc_fun=fc, activate_fn=af))
        self.attn_all_layer_names = []
        for layer, layer_args in attention_module.items():
            if layer_args.pop("is_last_layer", False):
                activate_fn = nn.Softmax(dim=0)
                self.attn_last_layer_dim = layer_args.out_features
            else:
                activate_fn = nn.ReLU(inplace=True)
            setattr(self, f"attention_{layer}", create_attention_layer(nn.Linear(**layer_args), activate_fn))
            self.attn_all_layer_names.append(f"attention_{layer}")

    def forward(self, img_feature: torch.Tensor, recipe_img_pos: List) -> List:
        img_feature = self.img_feat_proj(img_feature)
        img_features = [img_feature[p[0]:p[1]] for p in recipe_img_pos]

        return [self.attention_forward(img_feat) for img_feat in img_features]

    def attention_forward(self, feature: torch.Tensor) -> torch.Tensor:  # todo att_feat dims -> I could not understand
        att_feat = torch.zeros((self.attn_last_layer_dim, feature.shape[1]), dtype=feature.dtype, device=feature.device)
        att_feat[:len(feature)] = feature

        for attn_layer in self.attn_all_layer_names[:-1]:
            att_feat = getattr(self, attn_layer)(att_feat)
        att_feat = att_feat.t().contiguous()

        attn_last_layer = getattr(self, self.attn_all_layer_names[-1])
        fc_fun: Callable = attn_last_layer.fc_fun
        softmax_fun: Callable = attn_last_layer.activate_fn

        att_feat = fc_fun(att_feat).t().contiguous()
        return feature + feature * softmax_fun(att_feat[:len(feature)])
