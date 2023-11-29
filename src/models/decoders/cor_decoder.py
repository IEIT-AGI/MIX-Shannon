from torch import nn, Tensor
from typing import Optional
from src.models.model_funcs import _get_activation_fn, _get_clones
import torch
from src.models.builder import DECODERS, build_decoder, build_position_embedding
from omegaconf import DictConfig


# @DECODERS.register_module()
# class GenerateTransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before
#
#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos
#
#     def forward(self, src, src_mask: Optional[Tensor] = None, pmask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None):
#         q = k = self.with_pos_embed(src, pos)
#         src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=pmask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src


# @DECODER.register_module()
# class CorDecoder(nn.Module):
#     def __init__(self, d_model, encoder_layer, num_layers, tokenizer, word_embeddings, token_pos_eb):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.tokenizer = tokenizer
#         self.word_embeddings = word_embeddings
#         self.word_size, self.d_embeddings = word_embeddings.weight.shape
#         self.d_model = d_model
#         self.linear1 = nn.Linear(self.d_embeddings, self.d_model)
#         self.linear2 = nn.Linear(self.d_model, self.word_size)
#         self.linear3 = nn.Linear(self.d_model, 2)
#         self.token_pos_eb = token_pos_eb
#
#     def forward(self, memory_cs, pmask_cs, pos_rs, emb_ts=None, pmask_ts=None, pos_ts=None):
#         if emb_ts is not None:
#             assert pmask_ts is not None and pos_ts is not None
#             return self.forward_train(memory_cs, pmask_cs, pos_rs, emb_ts, pmask_ts, pos_ts)
#         else:
#             assert pmask_ts is None and pos_ts is None
#             return self.forward_test(memory_cs, pmask_cs, pos_rs)
#
#     def forward_train(self, memory_cs, pmask_cs, pos_rs, emb_ts, pmask_ts, pos_ts):
#         memory_ts = self.linear1(emb_ts)
#         tok_c, _, _ = memory_cs.shape
#         tok_t, _, _ = memory_ts.shape
#         mask_cor = self.get_pcor_mask(tok_c, tok_t).to(memory_cs.device)
#         pmask_cor = torch.cat([pmask_cs, pmask_ts], dim=1)
#         memory_cor = torch.cat([memory_cs, memory_ts], dim=0)
#         pos_cor = torch.cat([pos_rs, pos_ts], dim=0)
#
#         output = memory_cor
#         for layer in self.layers:
#             output = layer(output, src_mask=mask_cor, pmask=pmask_cor, pos=pos_cor)
#         output = output[-tok_t:]
#         token_prob = self.linear2(output)
#         change_prob = self.linear3(output)
#         return token_prob, change_prob
#
#     def forward_test(self, memory_cs, pmask_cs, pos_rs):
#         device = memory_cs.device
#         bs = memory_cs.shape[1]
#         token_ids_lst = torch.zeros([bs, 1], dtype=torch.long, device=device)
#         while sum([self.tokenizer.eos_token_id in token_ids for token_ids in token_ids_lst]) < bs and \
#                 token_ids_lst.shape[1] < len(memory_cs) * 2:
#             emb_ts = self.word_embeddings(token_ids_lst).transpose(0, 1)
#             memory_ts = self.linear1(emb_ts)
#             tok_c, _, _ = memory_cs.shape
#             tok_t, _, _ = memory_ts.shape
#             pmask_ts = torch.zeros_like(token_ids_lst, dtype=torch.bool, device=pmask_cs.device)
#             pos_ts = self.token_pos_eb(torch.arange(tok_t).to(device), bs)
#             mask_cor = self.get_pcor_mask(tok_c, tok_t).to(device)
#             pmask_cor = torch.cat([pmask_cs, pmask_ts], dim=1)
#             memory_cor = torch.cat([memory_cs, memory_ts], dim=0)
#             pos_cor = torch.cat([pos_rs, pos_ts], dim=0)
#             output = memory_cor
#             for layer in self.layers:
#                 output = layer(output, src_mask=mask_cor, pmask=pmask_cor, pos=pos_cor)
#             output = output[-1:]
#             output = self.linear2(output)
#             last_token_ids = output[-1, :].argmax(1)
#             token_ids_lst = torch.cat([token_ids_lst, last_token_ids.unsqueeze(1)], dim=1)
#         return token_ids_lst
#
#     def get_pcor_mask(self, l1, l2):
#         mask_pcor = torch.zeros([l1 + l2, l1 + l2], dtype=torch.bool)
#         mask_pcor[:l1, :l1] = torch.zeros([l1, l1], dtype=torch.bool)
#         mask_pcor[:l1, l1:] = torch.ones([l1, l2], dtype=torch.bool)
#         mask_pcor[l1:, :l1] = torch.zeros([l2, l1], dtype=torch.bool)
#         mask_pcor[l1:, l1:] = torch.tensor((1 - (torch.tril(torch.ones([l2, l2])))).numpy(), dtype=torch.bool)
#         return mask_pcor


# def __init__(self, d_model: int,
#              encoder_layer: DictConfig,
#              num_layers: int,
#              tokenizer: DictConfig,
#              word_embedding: DictConfig,
#              token_position_embedding: DictConfig):
#     super().__init__()
#     # self.layers = _get_clones(encoder_layer, num_layers)
#
#     self.num_layers = num_layers
#     self.tokenizer = tokenizer
#     self.d_model = d_model
#
#     self.layers = build_decoder(encoder_layer)
#     self.token_pos_eb = build_tokenizer(token_position_embedding)
#     self.word_embeddings = build_embedding(word_embedding)
#
#     self.word_size, self.d_embeddings = self.word_embeddings.weight.shape
#     self.linear1 = nn.Linear(self.d_embeddings, self.d_model)
#     self.linear2 = nn.Linear(self.d_model, self.word_size)

@DECODERS.register_module()
class CorDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 encoder_layer: DictConfig,
                 num_layers: int,
                 tokenizer: DictConfig,
                 word_embedding: DictConfig,
                 token_pos_eb: DictConfig):
        super().__init__()
        # self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = _get_clones(build_decoder(encoder_layer), num_layers)
        self.token_pos_eb = build_position_embedding(token_pos_eb)

        import hydra
        self.tokenizer = hydra.utils.instantiate(tokenizer)
        self.word_embeddings = hydra.utils.instantiate(word_embedding).embeddings.word_embeddings
        self.word_size, self.d_embeddings = self.word_embeddings.weight.shape

        self.linear1 = nn.Linear(self.d_embeddings, self.d_model)
        self.linear2 = nn.Linear(self.d_model, self.word_size)
        self.linear3 = nn.Linear(self.d_model, 2)

    def forward(self, memory_cs, pmask_cs, pos_rs, emb_ts=None, pmask_ts=None, pos_ts=None):
        if emb_ts is not None:
            assert pmask_ts is not None and pos_ts is not None
            return self.forward_train(memory_cs, pmask_cs, pos_rs, emb_ts, pmask_ts, pos_ts)
        else:
            assert pmask_ts is None and pos_ts is None
            return self.forward_test(memory_cs, pmask_cs, pos_rs)

    def forward_train(self, memory_cs, pmask_cs, pos_rs, emb_ts, pmask_ts, pos_ts):
        memory_ts = self.linear1(emb_ts)
        tok_c, _, _ = memory_cs.shape
        tok_t, _, _ = memory_ts.shape
        mask_cor = self.get_pcor_mask(tok_c, tok_t).to(memory_cs.device)
        pmask_cor = torch.cat([pmask_cs, pmask_ts], dim=1)
        memory_cor = torch.cat([memory_cs, memory_ts], dim=0)
        pos_cor = torch.cat([pos_rs, pos_ts], dim=0)

        output = memory_cor
        for layer in self.layers:
            output = layer(output, src_mask=mask_cor, pmask=pmask_cor, pos=pos_cor)
        output = output[-tok_t:]
        token_prob = self.linear2(output)
        change_prob = self.linear3(output)
        return token_prob, change_prob

    def forward_test(self, memory_cs, pmask_cs, pos_rs):
        device = memory_cs.device
        bs = memory_cs.shape[1]
        token_ids_lst = torch.zeros([bs, 1], dtype=torch.long, device=device)
        while sum([self.tokenizer.eos_token_id in token_ids for token_ids in token_ids_lst]) < bs and \
                token_ids_lst.shape[1] < len(memory_cs) * 2:
            emb_ts = self.word_embeddings(token_ids_lst).transpose(0, 1)
            memory_ts = self.linear1(emb_ts)
            tok_c, _, _ = memory_cs.shape
            tok_t, _, _ = memory_ts.shape
            pmask_ts = torch.zeros_like(token_ids_lst, dtype=torch.bool, device=pmask_cs.device)
            pos_ts = self.token_pos_eb(torch.arange(tok_t).to(device), bs)
            mask_cor = self.get_pcor_mask(tok_c, tok_t).to(device)
            pmask_cor = torch.cat([pmask_cs, pmask_ts], dim=1)
            memory_cor = torch.cat([memory_cs, memory_ts], dim=0)
            pos_cor = torch.cat([pos_rs, pos_ts], dim=0)
            output = memory_cor
            for layer in self.layers:
                output = layer(output, src_mask=mask_cor, pmask=pmask_cor, pos=pos_cor)
            output = output[-1:]
            output = self.linear2(output)
            last_token_ids = output[-1, :].argmax(1)
            token_ids_lst = torch.cat([token_ids_lst, last_token_ids.unsqueeze(1)], dim=1)
        return token_ids_lst

    def get_pcor_mask(self, l1, l2):
        mask_pcor = torch.zeros([l1 + l2, l1 + l2], dtype=torch.bool)
        mask_pcor[:l1, :l1] = torch.zeros([l1, l1], dtype=torch.bool)
        mask_pcor[:l1, l1:] = torch.ones([l1, l2], dtype=torch.bool)
        mask_pcor[l1:, :l1] = torch.zeros([l2, l1], dtype=torch.bool)
        mask_pcor[l1:, l1:] = torch.tensor((1 - (torch.tril(torch.ones([l2, l2])))).numpy(), dtype=torch.bool)
        return mask_pcor


def build_cor_decoder(d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu",
                      tokenizer=None, text_encoder=None, token_pos_eb=None):
    word_embeddings = text_encoder.embeddings.word_embeddings
    encoder_layer = GenerateTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    return CorDecoder(d_model, encoder_layer, num_encoder_layers, tokenizer, word_embeddings, token_pos_eb)
