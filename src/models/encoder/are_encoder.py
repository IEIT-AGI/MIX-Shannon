from typing import List, Optional, Union, Dict, Callable
import torch
from torch import nn
from omegaconf import DictConfig
from src.utils.misc import NestedTensor
from src.models.builder import ENCODERS, build_encoder, build_models
import hydra


@ENCODERS.register_module()
class AREEncoder(nn.Module):
    def __init__(self,
                 is_pass_pos: bool,
                 answer_token: DictConfig,
                 event_token: DictConfig,
                 tokenizer: DictConfig,
                 feature_resizer: DictConfig,
                 visual_encoder: DictConfig,
                 text_encoder: DictConfig,
                 reasoning_encoder: DictConfig,
                 grounding_encoder: DictConfig
                 ):
        super(AREEncoder, self).__init__()

        # Disables passing the positional encodings to each attention layers
        self.is_pass_pos = is_pass_pos

        # self.answer_token = build_embedding(answer_token)
        # self.event_token = build_embedding(event_token)
        self.answer_token = hydra.utils.instantiate(answer_token)
        self.event_token = hydra.utils.instantiate(event_token)

        # self.tokenizer = build_tokenizer(tokenizer)
        self.tokenizer = hydra.utils.instantiate(tokenizer)

        # self.text_encoder = build_text_encoder(text_encoder)
        self.text_encoder = hydra.utils.instantiate(text_encoder)
        self.feat_resizer = build_models(feature_resizer)
        self.visual_encoder = build_encoder(visual_encoder)
        self.reasoning_encoder = build_encoder(reasoning_encoder)
        self.explaining_encoder = build_encoder(grounding_encoder)

    @staticmethod
    def get_register_fn(register_fn: Dict, cfg: DictConfig, default_type: str) -> Callable:
        fn_type = cfg.pop("type", default_type)
        return register_fn.get(fn_type)

    def forward(self, image: NestedTensor, question: List) -> Dict:
        visual_encode_rst: Dict = self.visual_encoder(image)
        reasoning_encode_rst: Dict = self._encode_event_knowledge(visual_encode_rst, question)
        grd_encoder_rst: Dict = self._encode_grounding(visual_encode_rst, question)

        # output
        return {"explaining_encoder": grd_encoder_rst, "reasoning_encoder": reasoning_encode_rst}

    def _encode_question_for_reasoning(self, question: List) -> Dict:
        q_token = self.tokenizer.batch_encode_plus(question, padding="longest", return_tensors="pt")
        q_token = q_token.to(self.text_encoder.device)
        q_text_encoder = self.text_encoder.embeddings.word_embeddings(q_token.data["input_ids"]).transpose(0, 1)

        q_text_mask = q_token.attention_mask.ne(1).bool()
        q_text_encoder = self.feat_resizer(q_text_encoder)
        q_text_pos = torch.zeros_like(q_text_encoder)

        # output
        output = {"feature": q_text_encoder,
                  "position": q_text_pos,
                  "mask": q_text_mask}
        return output

    # TODO function name _encode_answer_event_base_img_question
    def _encode_event_knowledge(self, visual_info: Dict, question: List) -> Dict:

        batch_size = len(question)
        device = self.event_token.weight.device
        question_info: Dict = self._encode_question_for_reasoning(question)
        event_token = self.event_token.weight.view(1, 1, -1).repeat(1, batch_size, 1)
        answer_token = self.answer_token.weight.view(1, 1, -1).repeat(1, batch_size, 1)

        feat = [event_token, answer_token, visual_info["feature"], question_info["feature"]]
        mask = [torch.zeros(batch_size, 2).bool().to(device), visual_info["mask"], question_info["mask"]]
        position = [torch.zeros(2, batch_size, visual_info["feature"].shape[-1], device=device),
                    visual_info["position"], question_info["position"]]

        concat_feat = torch.cat(feat, dim=0)
        concat_mask = torch.cat(mask, dim=1)
        concat_position = torch.cat(position, dim=0)

        # cross encoder
        reason_cross_feat = self.reasoning_encoder(src=concat_feat,
                                                   src_key_padding_mask=concat_mask,
                                                   pos=concat_position)
        event_feat = reason_cross_feat[0]
        knowledge_feat = reason_cross_feat[1]
        relation_feat = reason_cross_feat[2]

        gate_feat = reason_cross_feat[3:]

        # output
        output = {"event_feature": event_feat,
                  "knowledge_feature": knowledge_feat,  # todo answer_feature is better
                  "relation_feature": relation_feat,
                  "gate_feature": gate_feat}

        return output

    def _encode_question_for_grd(self, question: List) -> Dict:
        q_token = self.tokenizer.batch_encode_plus(question, padding="longest", return_tensors="pt")
        q_token = q_token.to(self.text_encoder.device)
        q_text_original_encoder = self.text_encoder(**q_token)

        q_text_encoder = q_text_original_encoder.last_hidden_state.transpose(0, 1)
        q_text_encoder = self.feat_resizer(q_text_encoder)
        q_text_mask = q_token.attention_mask.ne(1).bool()
        q_text_pos = torch.zeros_like(q_text_encoder)

        # output
        output = {"resize_feature": q_text_encoder,
                  "original_feature": q_text_original_encoder,
                  "position": q_text_pos,
                  "mask": q_text_mask,
                  "token": q_token}
        return output

    def _encode_grounding(self, visual_info: Dict, question: List) -> Dict:

        def is_add_position(weight: float = 0.1) -> Dict:  # TODO why?
            # query_embed = self.query_embed.weight
            # query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
            # if self.is_pass_pos_and_query:
            #     tgt = torch.zeros_like(query_embed)
            # else:
            #     src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            if self.is_pass_pos:
                return visual_info
            else:
                from copy import deepcopy
                _output = deepcopy(visual_info)
                _output["feature"] += _output["position"] * weight
                _output["position"] = None
                return _output

        visual_info: Dict = is_add_position()
        grd_question_info: Dict = self._encode_question_for_grd(question)
        feat = [visual_info["feature"], grd_question_info["resize_feature"]]
        mask = [visual_info["mask"], grd_question_info["mask"]]
        position = [visual_info["position"], grd_question_info["position"]]

        concat_feat = torch.cat(feat, dim=0)
        concat_mask = torch.cat(mask, dim=1)
        concat_pos = torch.cat(position, dim=0)

        # cross encoder
        grd_cross_feat = self.explaining_encoder(src=concat_feat, src_key_padding_mask=concat_mask, pos=concat_pos)
        grd_question_feat = grd_cross_feat[-len(grd_question_info["resize_feature"]):]

        # output
        output = {f"question_{k}": v for k, v in grd_question_info.items()}
        output["grounding_question_feature"] = grd_question_feat  # img_memory_grd
        output["grounding_encoder_feature"] = grd_cross_feat  # text_memory_grd
        output["question_pool_feature"] = grd_question_info["original_feature"].pooler_output  # text_pooled_op
        output["visual_pool_feature"] = grd_cross_feat[0]  # img_pooled_op
        output["grounding_mask"] = concat_mask
        output["grounding_position"] = concat_pos

        return output
