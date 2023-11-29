from torch import nn, Tensor
from typing import Optional
from src.models.model_funcs import _get_activation_fn, _get_clones
import torch

from src.models.builder import DECODERS, build_decoder
from omegaconf import DictConfig


@DECODERS.register_module()
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(self, tgt, memory, text_memory, tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None, text_memory_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2, tgt_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2, tgt_weights = self.cross_attn_image(query=self.with_pos_embed(tgt, query_pos),
                                                  key=self.with_pos_embed(memory, pos), value=memory,
                                                  attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward_pre(self, tgt, memory, text_memory, tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None, text_memory_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        assert False, "not implemented yet"
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, text_memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                text_memory_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, text_memory, tgt_mask, memory_mask, text_memory_key_padding_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


# @DECODER.register_module()
# class GrdDecoder(nn.Module):
#     def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.return_intermediate = return_intermediate
#
#     def forward(self, tgt, memory, text_memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
#                 text_memory_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         output = tgt
#         intermediate = []
#         for layer in self.layers:
#             output = layer(output, memory, text_memory=text_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
#                            text_memory_key_padding_mask=text_memory_key_padding_mask,
#                            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
#                            pos=pos, query_pos=query_pos)
#             if self.return_intermediate:
#                 intermediate.append(self.norm(output))
#
#         if self.norm is not None:
#             output = self.norm(output)
#             if self.return_intermediate:
#                 intermediate.pop()
#                 intermediate.append(output)
#         if self.return_intermediate:
#             return torch.stack(intermediate)
#         return output

@DECODERS.register_module()
class GrdDecoder(nn.Module):
    def __init__(self,
                 decoder_layer: DictConfig,
                 num_layers: int,
                 is_norm: bool = True,
                 norm: Optional[DictConfig] = None,
                 return_intermediate: bool = False):
        super().__init__()
        # self.layers = _get_clones(decoder_layer, num_layers)
        # self.layers = build_decoder(decoder_layer)
        self.layers = _get_clones(build_decoder(decoder_layer), num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        if is_norm and norm:
            import hydra
            self.norm = hydra.utils.instantiate(norm)
        else:
            self.norm = None

    def forward(self, tgt, memory, text_memory, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                text_memory_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, text_memory=text_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           text_memory_key_padding_mask=text_memory_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


def build_grd_decoder(d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu",
                      normalize_before=False, return_intermediate=False):
    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    decoder_norm = nn.LayerNorm(d_model)
    return GrdDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate)
