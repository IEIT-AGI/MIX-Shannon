import torch
from torch import nn
from torch.nn.utils import rnn
from omegaconf import DictConfig
from typing import Dict, List, Any, Tuple
import random
import numpy as np
import dgl
import hydra
from ..builder import ENCODERS, build_position_embedding


@ENCODERS.register_module()
class RecipeEncoder(nn.Module):
    def __init__(self,
                 ins2ing: DictConfig,
                 ing2ins: DictConfig,
                 tf_embedding: DictConfig,
                 text_lstm: DictConfig,
                 text_lstm_proj: DictConfig,
                 title_feat_proj: DictConfig,
                 is_shuffle: bool = False,
                 n_iter: int = 1):
        super(RecipeEncoder, self).__init__()

        self.n_iter = n_iter
        self.is_shuffle = is_shuffle

        from src.models.GAT import WSWGAT
        self.ins2ing_GAT_obj = WSWGAT(**ins2ing)  # WSWGAT  instructions to ingredients  encoder? todo
        self.ing2ins_GAT_obj = WSWGAT(**ing2ins)  # WSWGAT  ingredients to instructions

        self.tf_embed = hydra.utils.instantiate(tf_embedding)
        self.text_lstm = hydra.utils.instantiate(text_lstm)
        self.text_lstm_proj = hydra.utils.instantiate(text_lstm_proj)
        self.title_feat_proj = hydra.utils.instantiate(title_feat_proj)

    def forward(self,
                instruction_feature: torch.Tensor,
                ingredients_feature: torch.Tensor,
                title_feature: torch.Tensor,
                heterogeneous_graph: dgl.DGLGraph,
                recipe_img_pos: List) -> Dict:
        self.init_heterogeneous_graph(heterogeneous_graph=heterogeneous_graph,
                                      instruction_feature=instruction_feature,
                                      ingredients_feature=ingredients_feature)
        instruction_node = self.GAT_forward(heterogeneous_graph=heterogeneous_graph,
                                            instruction_feature=instruction_feature,
                                            ingredients_feature=ingredients_feature)
        instruction_temporal_info = self.encode_text_temporal_info(instruction_node=instruction_node,
                                                                   recipe_img_pos=recipe_img_pos)

        # header : concatenate: title_feat + instruction_feat
        title_feature = self.title_feat_proj(title_feature)
        instruction_temporal_info["title_feature"] = title_feature

        # output
        output = {}
        for key, val in instruction_temporal_info.items():
            if key.startswith("text"):
                key = key.replace("text", "instruction")
            output[key] = val

        return output

    def init_heterogeneous_graph(self,
                                 heterogeneous_graph: dgl.DGLGraph,
                                 instruction_feature: torch.Tensor,
                                 ingredients_feature: torch.Tensor) -> None:
        # nodes initialization
        ins_node_id = heterogeneous_graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        heterogeneous_graph.nodes[ins_node_id].data["sent_embedding"] = instruction_feature

        ing_node_id = heterogeneous_graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        heterogeneous_graph.nodes[ing_node_id].data["sent_embedding"] = ingredients_feature

        # edges initialization
        edge_id = heterogeneous_graph.filter_edges(lambda edges: edges.data["dtype"] == 0)
        etf = heterogeneous_graph.edges[edge_id].data["tffrac"]
        heterogeneous_graph.edges[edge_id].data["tfidfembed"] = self.tf_embed(etf)

    def GAT_forward(self,
                    heterogeneous_graph: dgl.DGLGraph,
                    instruction_feature: torch.Tensor,
                    ingredients_feature: torch.Tensor) -> Any:  # GAT: graphical attention network

        instruction_node_state = self.ing2ins_GAT_obj(heterogeneous_graph, ingredients_feature, instruction_feature)
        for _ in range(self.n_iter):
            ingredients_node_state = self.ins2ing_GAT_obj(heterogeneous_graph,
                                                          ingredients_feature,
                                                          instruction_node_state)
            instruction_node_state = self.ing2ins_GAT_obj(heterogeneous_graph,
                                                          ingredients_node_state,
                                                          instruction_node_state)
        return instruction_node_state

    @staticmethod
    def shuffle_text(text_features: List) -> Tuple[List, List]:
        shuffle_features, shuffle_features_label = [], []
        for idx in range(len(text_features)):
            if random.random() > 0.5 and len(text_features[idx]) >= 2:
                feat_len, _ = text_features[idx].shape
                _idx = np.arange(feat_len)
                ratio = int(0.3 * feat_len / 2)
                ratio = 1 if ratio == 0 else ratio
                swap_num = random.sample(set(_idx), ratio * 2)
                for i in range(int(len(swap_num) / 2)):
                    j = 2 * i
                    _idx[swap_num[j + 1]], _idx[swap_num[j]] = _idx[swap_num[j]], _idx[swap_num[j + 1]]
                tmp = text_features[idx].clone()
                shuffle_features.append(tmp[_idx])
                shuffle_features_label.append(1)

            else:
                shuffle_features.append(text_features[idx].clone())
                shuffle_features_label.append(0)

        return shuffle_features, shuffle_features_label

    def encode_text_temporal_info(self,
                                  instruction_node: torch.Tensor,
                                  recipe_img_pos: List) -> Dict:
        text_features = [instruction_node[pos[0]:pos[1]] for pos in recipe_img_pos]

        lstm_feature = self.extract_text_temporal_feature(text_features)
        shuffle_features, shuffle_features_label, _lstm_shuffle_features = None, None, None
        if self.is_shuffle:
            shuffle_features, shuffle_features_label = self.shuffle_text(text_features)
            _lstm_shuffle_features = self.extract_text_temporal_feature(shuffle_features)

        # output
        lstm_feature_mean, lstm_features, lstm_shuffle_features_mean = [], [], [] if self.is_shuffle else None
        for pos in recipe_img_pos:
            lstm_features.append(lstm_feature[pos[0]:pos[1]])
            lstm_feature_mean.append(lstm_feature[pos[0]:pos[1]].mean(0))
            if self.is_shuffle:
                lstm_shuffle_features_mean.append(_lstm_shuffle_features[pos[0]:pos[1]].mean(0))

        output = {"text_feature_mean": lstm_feature_mean,
                  "text_features": lstm_features,
                  "text_shuffle_features_mean": lstm_shuffle_features_mean,
                  "text_shuffle_features_label": shuffle_features_label
                  }
        return output

    def extract_text_temporal_feature(self, text_features: List) -> torch.Tensor:
        seq_pad = rnn.pad_sequence(text_features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(seq_pad,
                                              lengths=[len(feat) for feat in text_features],
                                              batch_first=True,
                                              enforce_sorted=False)

        lstm_output, _ = self.text_lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[idx][:unpacked_len[idx]] for idx in range(len(unpacked))]
        feature = self.text_lstm_proj(torch.cat(lstm_embedding, dim=0))
        return feature


@ENCODERS.register_module()
class CSIEncoder(nn.Module):
    """
    CSIEncoder: cooking steps image encoder
    """

    def __init__(self,
                 csi_lstm: DictConfig,
                 csi_lstm_proj: DictConfig,
                 AEN: DictConfig,
                 is_shuffle: bool = False):
        super(CSIEncoder, self).__init__()
        self.is_shuffle = is_shuffle

        self.csi_lstm = hydra.utils.instantiate(csi_lstm)
        self.csi_lstm_proj = hydra.utils.instantiate(csi_lstm_proj)

        self.attention_embedding_network = build_position_embedding(AEN)

    def forward(self, img_feature: torch.Tensor, recipe_img_pos: List) -> Dict:
        csi_features = self.AEN_forward(img_feature, recipe_img_pos)
        return self.encode_csi_temporal_info(csi_features, recipe_img_pos)

    def AEN_forward(self, img_feature: torch.Tensor, recipe_img_pos: List) -> List:  # AEN: attention_embedding network
        return self.attention_embedding_network(img_feature, recipe_img_pos)

    def encode_csi_temporal_info(self,
                                 csi_features: List[torch.Tensor],
                                 recipe_img_pos: List) -> Dict:

        lstm_feature = self.extract_csi_temporal_feature(csi_features)
        shuffle_features, shuffle_features_label, _lstm_shuffle_features = None, None, None
        if self.is_shuffle:
            shuffle_features, shuffle_features_label = self.shuffle_img(csi_features)
            _lstm_shuffle_features = self.extract_csi_temporal_feature(shuffle_features)

        # output
        lstm_feature_mean, lstm_features, lstm_shuffle_features_mean = [], [], [] if self.is_shuffle else None
        for pos in recipe_img_pos:
            lstm_features.append(lstm_feature[pos[0]:pos[1]])
            lstm_feature_mean.append(lstm_feature[pos[0]:pos[1]].mean(0))
            if self.is_shuffle:
                lstm_shuffle_features_mean.append(_lstm_shuffle_features[pos[0]:pos[1]].mean(0))

        output = {"csi_feature_mean": lstm_feature_mean,
                  "csi_features": lstm_features,
                  "csi_shuffle_features_mean": lstm_shuffle_features_mean,
                  "csi_shuffle_features_label": shuffle_features_label
                  }
        return output

    def extract_csi_temporal_feature(self, csi_features: List) -> torch.Tensor:
        seq_pad = rnn.pad_sequence(csi_features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(seq_pad,
                                              lengths=[len(feat) for feat in csi_features],
                                              batch_first=True,
                                              enforce_sorted=False)  # TODO
        lstm_output, _ = self.csi_lstm(lstm_input)

        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[idx][:unpacked_len[idx]] for idx in range(len(unpacked))]
        feature = self.csi_lstm_proj(torch.cat(lstm_embedding, dim=0))
        return feature

    @staticmethod
    def shuffle_img(csi_features):
        return RecipeEncoder.shuffle_text(csi_features)
