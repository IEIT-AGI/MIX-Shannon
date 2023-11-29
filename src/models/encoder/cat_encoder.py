from src.models.model_funcs import *
from src.models.model_funcs import _get_activation_fn, _get_clones
from src.models.builder import ENCODERS, build_encoder
from omegaconf import DictConfig


@ENCODERS.register_module()
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# @ENCODER.register_module()
# class CatEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.norm = norm
#
#     def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         output = src
#         for layer in self.layers:
#             output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
#         if self.norm is not None:
#             output = self.norm(output)
#         return output

@ENCODERS.register_module()
class CatEncoder(nn.Module):
    def __init__(self, encoder_layer: DictConfig,
                 num_layers: int,
                 is_before_norm: bool,
                 norm: Optional[DictConfig] = None):
        super().__init__()
        # self.layers = _get_clones(encoder_layer, num_layers)
        self.layers = _get_clones(build_encoder(encoder_layer), num_layers)
        self.num_layers = num_layers
        if is_before_norm and norm:
            import hydra
            self.norm = hydra.utils.instantiate(norm)
        else:
            self.norm = None

    def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


def build_cat_encoder(d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu",
                      normalize_before=False):
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    return CatEncoder(encoder_layer, num_encoder_layers, encoder_norm)
