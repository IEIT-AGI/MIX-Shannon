from src.models.builder import ENCODERS, build_encoder, build_models, build_position_embedding
from torch import nn
from omegaconf import DictConfig
from src.utils.misc import NestedTensor
from typing import List, Optional, Callable, Dict
import torch
import copy


@ENCODERS.register_module()
class FCTRBaselineEncoder(nn.Module):

    def __init__(self, image_encoder: DictConfig, text_encoder: DictConfig, multimodal_fusion: DictConfig,
                 token_pos_eb: DictConfig):
        super(FCTRBaselineEncoder, self).__init__()
        self.image_encoder: Callable = build_encoder(image_encoder)
        self.text_encoder: Callable = build_encoder(text_encoder)
        self.multimodal_fusion: Callable = build_encoder(multimodal_fusion)
        self.token_pos_eb = build_position_embedding(token_pos_eb)
        self.word_embeddings = copy.deepcopy(self.text_encoder.encoder.embeddings.word_embeddings)

    def forward(self, images: NestedTensor, raw_sentences: List, target_sentences: Optional[List] = None) -> Dict:
        if self.training:
            return self.forward_train(images, raw_sentences, target_sentences)
        else:
            return self.forward_test(images, raw_sentences)

    def forward_train(self, images: NestedTensor, raw_sentences: List, target_sentences: Optional[List] = None) -> Dict:
        output = self.forward_test(images, raw_sentences)
        output["target_sentence"] = self.text_encoder(target_sentences)

        return output

    def forward_test(self, images: NestedTensor, raw_sentences: List) -> Dict:
        visual_feat = self.image_encoder(images)
        raw_sent_feat = self.text_encoder(raw_sentences)
        aligned_feat = self._multimodal_fusion_process(img_feat=visual_feat, text_feat=raw_sent_feat)

        output = {
            "visual": visual_feat,
            "raw_sentence": raw_sent_feat,
            "aligned_feat": aligned_feat,
            "token_pos_embedding_fun": self.token_pos_eb,
            "word_embeddings_fun": self.word_embeddings
        }

        return output

    def _multimodal_fusion_process(self, img_feat: Dict, text_feat: Dict) -> Dict:
        feature_cat = torch.cat([img_feat.get("feature"), text_feat.get("feature")], dim=0)
        padding_mask_cat = torch.cat([img_feat.get("mask"), text_feat.get("padding_mask")], dim=1)
        position_cat = torch.cat([img_feat.get("position"), torch.zeros_like(text_feat.get("feature"))], dim=0)
        feature = self.multimodal_fusion(src=feature_cat, src_key_padding_mask=padding_mask_cat, pos=position_cat)
        output = {
            "feature": feature,
            "position": position_cat,
            "padding_mask": padding_mask_cat,
        }
        return output


@ENCODERS.register_module()
class FCTREncoder(nn.Module):

    def __init__(
        self,
        image_encoder: DictConfig,
        text_encoder: DictConfig,
        multimodal_fusion: DictConfig,
        attention_interaction: DictConfig,  # attentional interaction module
        token_pos_eb: DictConfig,
    ):
        super(FCTREncoder, self).__init__()

        self.image_encoder: Callable = build_encoder(image_encoder)
        self.text_encoder: Callable = build_encoder(text_encoder)
        self.multimodal_fusion: Callable = build_encoder(multimodal_fusion)
        self.attention_interaction: Callable = build_encoder(attention_interaction)
        self.token_pos_eb = build_position_embedding(token_pos_eb)

    def forward(self, images: NestedTensor, raw_sentences: List, target_sentences: Optional[List] = None) -> Dict:
        if self.training:
            return self.forward_train(images, raw_sentences, target_sentences)
        else:
            return self.forward_test(images, raw_sentences)

    def forward_train(self, images: NestedTensor, raw_sentences: List, target_sentences: Optional[List] = None) -> Dict:
        output = self.forward_test(images, raw_sentences)
        output["target_sentence"] = self.text_encoder(target_sentences)

        return output

    def forward_test(self, images: NestedTensor, raw_sentences: List) -> Dict:
        visual_feat = self.image_encoder(images)
        raw_sent_feat = self.text_encoder(raw_sentences)
        aligned_feat = self._multimodal_fusion_process(img_feat=visual_feat, text_feat=raw_sent_feat)
        aim_rst = self._attentional_interaction_process(aligned_feat, text_feat=raw_sent_feat)

        output = {
            "visual": visual_feat,
            "raw_sentence": raw_sent_feat,
            "aligned_feat": aligned_feat,
            "attentional_interaction": aim_rst
        }

        return output

    def _multimodal_fusion_process(self, img_feat: Dict, text_feat: Dict) -> Dict:
        feature_cat = torch.cat([img_feat.get("feature"), text_feat.get("feature")], dim=0)
        padding_mask_cat = torch.cat([img_feat.get("mask"), text_feat.get("padding_mask")], dim=1)
        position_cat = torch.cat([img_feat.get("position"), torch.zeros_like(text_feat.get("feature"))], dim=0)
        feature = self.multimodal_fusion(src=feature_cat, src_key_padding_mask=padding_mask_cat, pos=position_cat)
        output = {
            "feature": feature,
            "position": position_cat,
            "padding_mask": padding_mask_cat,
        }
        return output

    def _attentional_interaction_process(self, aligned_feat: Dict, text_feat: Dict) -> Dict:
        device = aligned_feat['feature'].device
        tok_r, bs, _ = text_feat.get("feature").shape
        pos_rs = self.token_pos_eb(torch.arange(tok_r).to(device), bs)
        explanation_signal, correction_signal = self.attention_interaction(memory_cat=aligned_feat["feature"],
                                                                           pos_cat=aligned_feat["position"],
                                                                           pmask_cat=aligned_feat["padding_mask"],
                                                                           memory_rs=text_feat["feature"],
                                                                           pmask_rs=text_feat["padding_mask"],
                                                                           pos_rs=pos_rs)
        # output = {"rationale_attn": rationale_attn,
        #           "correction_attn": correction_attn,
        #           "position_embed": pos_rs}

        output = {
            "explanation_signal": explanation_signal,
            "correction_signal": correction_signal,
            "position_embed": pos_rs
        }
        return output


@ENCODERS.register_module()
class FCTRTextEncoder(nn.Module):

    def __init__(self, tokenizer: DictConfig, encoder: DictConfig, feature_resizer: DictConfig):
        super(FCTRTextEncoder, self).__init__()

        import hydra
        from transformers import PreTrainedTokenizerBase
        self.tokenizer: PreTrainedTokenizerBase = hydra.utils.instantiate(tokenizer)
        self.encoder: PreTrainedTokenizerBase = hydra.utils.instantiate(encoder)

        self.feature_resizer = build_models(feature_resizer)

    def forward(self, sentence: List, padding="longest", return_tensors="pt") -> Dict:
        sent_token = self.tokenizer.batch_encode_plus(sentence, padding=padding, return_tensors=return_tensors)
        sent_token = sent_token.to(self.encoder.device)
        sent_encode = self.encoder(**sent_token)
        sent_feature = sent_encode.last_hidden_state.transpose(0, 1)
        sent_padding_mask = sent_token.attention_mask.ne(1).bool()
        sent_feature = self.feature_resizer(sent_feature)

        output = {"token": sent_token, "feature": sent_feature, "padding_mask": sent_padding_mask}

        return output
