from omegaconf import DictConfig
from torch import nn
from src.models.builder import ENCODERS, build_backbone, build_position_embedding
from typing import Dict, Callable, Tuple, List
from src.utils.misc import NestedTensor
from collections import OrderedDict
import hydra


@ENCODERS.register_module()
class VisualEncoder(nn.Module):
    def __init__(self, backbone: DictConfig, position_embedding: DictConfig, visual_proj: DictConfig):
        super(VisualEncoder, self).__init__()
        self.encode: nn.Sequential = self._build_model(backbone, position_embedding)
        self.visual_proj: Callable = hydra.utils.instantiate(visual_proj)

    @staticmethod
    def _build_model(backbone: DictConfig, position_embedding: DictConfig) -> nn.Sequential:
        model = nn.Sequential(OrderedDict(backbone=build_backbone(backbone),
                                          position_embedding=build_position_embedding(position_embedding)))
        model.num_channels = model.backbone.num_channels
        return model

    def _encode(self, image: NestedTensor) -> Tuple[List, List]:
        hs = self.encode.backbone(image)
        feature, position = [], []
        for n, x in hs.items():
            feature.append(x)
            position.append(self.encode.position_embedding(x).to(x.tensors.dtype))
        return feature, position

    def forward(self, image: NestedTensor) -> Dict:
        feature, position = self._encode(image)
        feat, mask = feature[-1].decompose()
        position_embed = position[-1]

        feat = self.visual_proj(feat).flatten(2).permute(2, 0, 1)
        position_embed = position_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        return {"feature": feat, "position": position_embed, "mask": mask}
