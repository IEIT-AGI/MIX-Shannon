from torch import nn, Tensor
from typing import Optional
from src.models.model_funcs import _get_activation_fn, _get_clones
from src.models.builder import ENCODERS, build_encoder
from omegaconf import DictConfig


@ENCODERS.register_module()
class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn_c = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_s = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_c = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_s = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1c = nn.Linear(d_model, dim_feedforward)
        self.linear1s = nn.Linear(d_model, dim_feedforward)
        self.linear2c = nn.Linear(dim_feedforward, d_model)
        self.linear2s = nn.Linear(dim_feedforward, d_model)

        self.norm1c = nn.LayerNorm(d_model)
        self.norm1s = nn.LayerNorm(d_model)
        self.norm2c = nn.LayerNorm(d_model)
        self.norm2s = nn.LayerNorm(d_model)
        self.norm3c = nn.LayerNorm(d_model)
        self.norm3s = nn.LayerNorm(d_model)
        self.dropout1c = nn.Dropout(dropout)
        self.dropout1s = nn.Dropout(dropout)
        self.dropout2c = nn.Dropout(dropout)
        self.dropout2s = nn.Dropout(dropout)
        self.dropout3c = nn.Dropout(dropout)
        self.dropout3s = nn.Dropout(dropout)
        self.dropout4c = nn.Dropout(dropout)
        self.dropout4s = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward(self, memory_cat, pos_cat, pmask_cat, memory_rs, pmask_rs, pos_rs):
        q_c = k_c = self.with_pos_embed(memory_cat, pos_cat)
        memory_cat_p, _ = self.self_attn_c(query=q_c, key=k_c, value=memory_cat, key_padding_mask=pmask_cat)
        memory_cat = memory_cat + self.dropout1c(memory_cat_p)
        memory_cat = self.norm1c(memory_cat)

        q_rs = k_rs = self.with_pos_embed(memory_rs, pos_rs)
        memory_rs_p, _ = self.self_attn_s(query=q_rs, key=k_rs, value=memory_rs, key_padding_mask=pmask_rs)
        memory_rs = memory_rs + self.dropout1s(memory_rs_p)
        memory_rs = self.norm1s(memory_rs)

        memory_cat_p, _ = self.cross_attn_c(query=self.with_pos_embed(memory_cat, pos_cat),
                                            key=self.with_pos_embed(memory_rs, pos_rs), value=memory_rs,
                                            key_padding_mask=pmask_rs)
        out_memory_cat = memory_cat + self.dropout2c(memory_cat_p)
        out_memory_cat = self.norm2c(out_memory_cat)

        memory_rs_p, _ = self.cross_attn_s(query=self.with_pos_embed(memory_rs, pos_rs),
                                           key=self.with_pos_embed(memory_cat, pos_cat), value=memory_cat,
                                           key_padding_mask=pmask_cat)
        out_memory_rs = memory_rs + self.dropout2s(memory_rs_p)
        out_memory_rs = self.norm2s(out_memory_rs)

        # FFN
        out_memory_cat_p = self.linear2c(self.dropout3c(self.activation(self.linear1c(out_memory_cat))))
        out_memory_cat = out_memory_cat + self.dropout4c(out_memory_cat_p)
        out_memory_cat = self.norm3c(out_memory_cat)

        out_memory_rs_p = self.linear2s(self.dropout3s(self.activation(self.linear1s(out_memory_rs))))
        out_memory_rs = out_memory_rs + self.dropout4s(out_memory_rs_p)
        out_memory_rs = self.norm3s(out_memory_rs)

        return out_memory_cat, out_memory_rs


# @ENCODER.register_module()
# class CrsEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers, norm_c=None, norm_s=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm_c = norm_c
#         self.norm_s = norm_s
#
#     def forward(self, memory_cat, pos_cat, pmask_cat, memory_rs, pmask_rs, pos_rs):
#         output_memory_cat = memory_cat
#         output_memory_rs = memory_rs
#         for layer in self.layers:
#             output_memory_cat, output_memory_rs = layer(output_memory_cat, pos_cat, pmask_cat, output_memory_rs,
#                                                         pmask_rs, pos_rs)
#         if self.norm_c is not None:
#             output_memory_cat = self.norm_c(output_memory_cat)
#         if self.norm_s is not None:
#             output_memory_rs = self.norm_s(output_memory_rs)
#         return output_memory_cat, output_memory_rs


@ENCODERS.register_module()
class CrsEncoder(nn.Module):
    def __init__(self,
                 encoder_layer: DictConfig,
                 num_layers: int,
                 norm_c: DictConfig = None,
                 norm_s: DictConfig = None):
        super().__init__()
        # self.layers = _get_clones(encoder_layer, num_layers)
        # self.layers = build_encoder(encoder_layer)
        self.layers = _get_clones(build_encoder(encoder_layer), num_layers)
        self.num_layers = num_layers

        import hydra
        self.norm_c = hydra.utils.instantiate(norm_c)
        self.norm_s = hydra.utils.instantiate(norm_s)

    def forward(self, memory_cat, pos_cat, pmask_cat, memory_rs, pmask_rs, pos_rs):
        output_memory_cat = memory_cat
        output_memory_rs = memory_rs
        for layer in self.layers:
            output_memory_cat, output_memory_rs = layer(output_memory_cat, pos_cat, pmask_cat, output_memory_rs,
                                                        pmask_rs, pos_rs)
        if self.norm_c is not None:
            output_memory_cat = self.norm_c(output_memory_cat)
        if self.norm_s is not None:
            output_memory_rs = self.norm_s(output_memory_rs)
        return output_memory_cat, output_memory_rs


def build_crs_decoder(d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu"):
    encoder_layer = CrossTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    encoder_norm_c = nn.LayerNorm(d_model)
    encoder_norm_s = nn.LayerNorm(d_model)
    return CrsEncoder(encoder_layer, num_encoder_layers, encoder_norm_c, encoder_norm_s)
