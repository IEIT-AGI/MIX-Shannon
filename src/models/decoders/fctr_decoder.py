import hydra.utils
from torch import nn
from src.models.builder import DECODERS, build_decoder, build_encoder, build_head
from omegaconf import DictConfig
from typing import Dict, List, Optional, Callable, Union, Tuple
import torch
from collections import OrderedDict
from transformers import BartTokenizer


@DECODERS.register_module()
class EMLMLayer(nn.Module):
    def __init__(self, input_dim: int, mid_dim: int, word_size: int):
        super(EMLMLayer, self).__init__()
        self.first_linear = nn.Linear(input_dim, mid_dim)
        self.current_linear = nn.Linear(mid_dim, word_size)
        self.before_linear = nn.Linear(mid_dim, word_size)

    def forward(self, memory_cs) -> Tuple[torch.Tensor, torch.Tensor]:
        mid_rst = self.first_linear(memory_cs)
        tokens_current_prob = self.current_linear(mid_rst)
        tokens_before_prob = self.before_linear(mid_rst)
        return tokens_current_prob, tokens_before_prob


@DECODERS.register_module()
class EMLMTask(nn.Module):  # todo
    def __init__(self, decoder: DictConfig, header: DictConfig, tokenizer: DictConfig):
        super(EMLMTask, self).__init__()
        self.decoder = build_decoder(decoder)
        self.header = build_head(header)
        self.tokenizer = hydra.utils.instantiate(tokenizer)

    def forward(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict) -> Dict:
        memory_cs = aim_rst.get("correction_signal") + raw_sentences.get("feature")
        current_token_prob, before_tokens_prob = self.decoder(memory_cs)
        output = self.header(current_token_prob,
                             before_tokens_prob,
                             raw_sentences,
                             target_sentences,
                             self.tokenizer)
        return output


@DECODERS.register_module()
class RationaleGenerationTask(nn.Module):
    def __init__(self, decoder: DictConfig, header: DictConfig):
        super(RationaleGenerationTask, self).__init__()
        self.decoder = build_decoder(decoder)
        self.header = build_head(header)

    def forward(self,
                raw_sentences: Dict,
                aim_rst: Dict,
                aligned_feat: Dict,
                target_sentences: Optional[Dict] = None,
                rationales: Optional[List] = None):
        if self.training:
            return self.forward_train(raw_sentences, aim_rst, aligned_feat, target_sentences, rationales)
        else:
            return self.forward_test(raw_sentences, aim_rst, aligned_feat)

    def _decode_train(self, raw_sentence, aim_rst, aligned_feat, rationales):
        memory_rat_attn, pos_rs = aim_rst["explanation_signal"], aim_rst["position_embed"]
        memory_rs, pmask_rs = raw_sentence["feature"], raw_sentence["padding_mask"]
        pmask_cat, pos_cat = aligned_feat["padding_mask"], aligned_feat["position"]

        device = raw_sentence['feature'].device
        tokenized_xs = self.decoder.tokenizer.batch_encode_plus(rationales, padding="longest", return_tensors="pt").to(
            device)
        tok_x = tokenized_xs.data["input_ids"].shape[1]
        tokens_xs = tokenized_xs.data["input_ids"]
        pos_xs = self.decoder.token_pos_eb(torch.arange(tok_x).to(device), len(rationales))

        memory_rat = torch.cat([memory_rat_attn, memory_rs], dim=0)
        pmask_rat = torch.cat([pmask_cat, pmask_rs], dim=1)
        pos_rat = torch.cat([pos_cat, pos_rs], dim=0)
        pmask_xs = tokenized_xs.attention_mask.ne(1).bool()
        emb_xs = self.decoder.word_embeddings(tokens_xs).transpose(0, 1)
        tokens_prob_xs = self.decoder(memory_rat, pmask_rat, pos_rat, emb_xs, pmask_xs, pos_xs)

        output = {"tokens_prob": tokens_prob_xs,
                  "rationales_token": tokenized_xs}

        return output

    def _generate_caption(self, raw_sentence, aim_rst, aligned_feat):
        memory_rat_attn, pos_rs = aim_rst["explanation_signal"], aim_rst["position_embed"]
        memory_rs, pmask_rs = raw_sentence["feature"], raw_sentence["padding_mask"]
        pmask_cat, pos_cat = aligned_feat["padding_mask"], aligned_feat["position"]

        # tokenized_xs = self.tokenizer.batch_encode_plus(rationales, padding="longest", return_tensors="pt").to(
        #     self.device)
        # tok_x = tokenized_xs.data["input_ids"].shape[1]
        # tokens_xs = tokenized_xs.data["input_ids"]
        # pos_xs = self.token_pos_eb(torch.arange(tok_x).to(self.device), len(rationales))
        device = raw_sentence['feature'].device
        bs = pmask_cat.shape[0]
        tokens_xs = torch.zeros([bs, 1], dtype=torch.long, device=device)

        memory_rat = torch.cat([memory_rat_attn, memory_rs], dim=0)
        pmask_rat = torch.cat([pmask_cat, pmask_rs], dim=1)
        pos_rat = torch.cat([pos_cat, pos_rs], dim=0)

        sum_eos: Callable = lambda tokens: sum([self.decoder.tokenizer.eos_token_id in tk_ids for tk_ids in tokens])
        while sum_eos(tokens_xs) < bs and tokens_xs.shape[1] < 64:
            pos_xs = self.decoder.token_pos_eb(torch.arange(tokens_xs.shape[1], device=device), bs)
            pmask_xs = torch.zeros_like(tokens_xs, dtype=torch.bool, device=device)
            emb_xs = self.decoder.word_embeddings(tokens_xs).transpose(0, 1)
            tokens_prob_xs = self.decoder(memory_rat, pmask_rat, pos_rat, emb_xs, pmask_xs, pos_xs)
            tokens_xs_add = tokens_prob_xs.argmax(-1).t()[:, -1:]
            tokens_xs = torch.cat([tokens_xs, tokens_xs_add], dim=1)

        return tokens_xs

    @staticmethod
    def convert_token(tokens: torch.Tensor, tokenizer: BartTokenizer) -> List:  # todo
        predict_sentences = []
        for tk in tokens:
            tk_ids = [tk_id for tk_id in tk.tolist() if tk_id != tokenizer.bos_token_id]
            predict_sentence = tokenizer.decode(tk_ids).split(tokenizer.eos_token)[0]
            predict_sentences.append(predict_sentence)

        return predict_sentences

    def forward_train(self,
                      raw_sentences: Dict,
                      aim_rst: Dict,
                      aligned_feat: Dict,
                      target_sentences: Optional[Dict] = None,
                      rationales: Optional[List] = None):
        decode_rst = self._decode_train(raw_sentences, aim_rst, aligned_feat, rationales)
        return self.header(raw_sentences, target_sentences, decode_rst)

    def forward_test(self,
                     raw_sentences: Dict,
                     aim_rst: Dict,
                     aligned_feat: Dict):
        caption_token = self._generate_caption(raw_sentences, aim_rst, aligned_feat)
        predict_sentences = self.convert_token(caption_token, self.decoder.tokenizer)

        # output
        return {"caption_token": caption_token,
                "predict_sentences": predict_sentences}


@DECODERS.register_module()
class ExpressionCorrectionTask(nn.Module):
    def __init__(self, decoder: DictConfig, header: DictConfig):
        super(ExpressionCorrectionTask, self).__init__()
        self.decoder = build_decoder(decoder)
        self.header = build_head(header)

    def forward(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Optional[Dict] = None):
        if self.training:
            return self.forward_train(raw_sentences, aim_rst, target_sentences)
        else:
            return self.forward_test(raw_sentences, aim_rst)

    def _decode_train(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict):
        pmask_cs = raw_sentences["padding_mask"]  # cross padding mask

        # Denoised Feauture is equal to correction signal and noisy feature
        memory_cs = aim_rst["correction_signal"] + raw_sentences["feature"]

        tok_t, bs, _ = target_sentences["feature"].shape
        tokens_ts = target_sentences["token"].data["input_ids"]  # target sentence token
        padding_mask_ts = target_sentences["padding_mask"]

        pos_ts = self.decoder.token_pos_eb(torch.arange(tokens_ts.shape[1]).to(pmask_cs.device), bs)
        emb_ts = self.decoder.word_embeddings(tokens_ts).transpose(0, 1)
        pos_rs = aim_rst["position_embed"]

        token_prob, change_prob = self.decoder(memory_cs=memory_cs,
                                               pmask_cs=pmask_cs,
                                               pos_rs=pos_rs,
                                               emb_ts=emb_ts,
                                               pmask_ts=padding_mask_ts[:, :len(pos_ts)],
                                               pos_ts=pos_ts)
        output = {"cross_token_prob": token_prob,
                  "token_change_prob": change_prob,
                  "corrected_feature": memory_cs}  # todo corrected_feature

        return output

    def _generate_caption(self, raw_sentences: Dict, aim_rst: Dict):
        pmask_cs = raw_sentences["padding_mask"]  # cross padding mask
        device = pmask_cs.device
        # Denoised Feauture is equal to correction signal and noisy feature
        memory_cs = aim_rst["correction_signal"] + raw_sentences["feature"]
        bs = pmask_cs.shape[0]
        tokens_ts = torch.zeros([bs, 1], dtype=torch.long, device=device)

        sum_eos: Callable = lambda tokens: sum([self.decoder.tokenizer.eos_token_id in tk_ids for tk_ids in tokens])
        while sum_eos(tokens_ts) < bs and tokens_ts.shape[1] < len(memory_cs) * 2:
            pos_ts = self.decoder.token_pos_eb(torch.arange(tokens_ts.shape[1]).to(device), bs)
            emb_ts = self.decoder.word_embeddings(tokens_ts).transpose(0, 1)
            pos_rs = aim_rst["position_embed"]
            pmask_ts = torch.zeros_like(tokens_ts, dtype=torch.bool, device=device)

            token_prob, change_prob = self.decoder(memory_cs=memory_cs,
                                                   pmask_cs=pmask_cs,
                                                   pos_rs=pos_rs,
                                                   emb_ts=emb_ts,
                                                   pmask_ts=pmask_ts[:, :len(pos_ts)],
                                                   pos_ts=pos_ts)
            tokens_ts_add = token_prob.argmax(-1).t()[:, -1:]
            tokens_ts = torch.cat([tokens_ts, tokens_ts_add], dim=1)

        return tokens_ts, memory_cs

    def forward_train(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict) -> Dict:
        decode_rst = self._decode_train(raw_sentences, aim_rst, target_sentences)
        return self.header(raw_sentences, aim_rst, target_sentences, decode_rst)

    def forward_test(self, raw_sentences: Dict, aim_rst: Dict) -> Dict:
        caption_token, corrected_feature = self._generate_caption(raw_sentences, aim_rst)
        predict_sentences = RationaleGenerationTask.convert_token(caption_token, self.decoder.tokenizer)

        return {"caption_token": caption_token,
                "corrected_feature": corrected_feature,
                "predict_sentences": predict_sentences}


@DECODERS.register_module()
class GroundingTask(nn.Module):
    def __init__(self,
                 encoder: Union[DictConfig, nn.Module],
                 decoder: Union[DictConfig, nn.Module],
                 header: DictConfig,
                 query_embed: DictConfig):
        super(GroundingTask, self).__init__()

        self.model = self._build_model(encoder, decoder)
        self.header = build_head(header)
        self.query_embed = hydra.utils.instantiate(query_embed)

    @staticmethod
    def _build_model(encoder: Union[DictConfig, nn.Module],
                     decoder: Union[DictConfig, nn.Module]) -> nn.Sequential:
        _encoder = build_encoder(encoder) if isinstance(encoder, DictConfig) else encoder
        _decoder = build_decoder(decoder) if isinstance(decoder, DictConfig) else decoder
        return nn.Sequential(OrderedDict(encoder=_encoder, decoder=_decoder))

    def forward(self, visual_feat: Dict, aim_rst: Dict, raw_sentences: Dict, target: Optional[Dict] = None) -> Dict:
        if self.training:
            return self.forward_train(visual_feat, aim_rst, raw_sentences, target)
        else:
            return self.forward_test(visual_feat, aim_rst, raw_sentences)

    def forward_train(self, visual_feat: Dict, aim_rst: Dict, raw_sentences: Dict, target: Dict) -> Dict:
        grd_tf = self._grounding_transform(visual_feat, aim_rst, raw_sentences)
        return self.header(raw_sentences, grd_tf, target)

    def forward_test(self, visual_feat: Dict, aim_rst: Dict, raw_sentences: Dict) -> Dict:
        grd_tf = self._grounding_transform(visual_feat, aim_rst, raw_sentences)
        return self.header(raw_sentences, grd_tf)

    def _grounding_transform(self, visual_feat: Dict, aim_rst: Dict, raw_sentences: Dict) -> Dict:
        memory_img = visual_feat["feature"]
        pmask_img = visual_feat["mask"]
        pos_img = visual_feat["position"]

        pos_grd = self.query_embed.weight
        pos_grd = pos_grd.unsqueeze(1).repeat(1, pmask_img.shape[0], 1)
        memory_grd = torch.zeros_like(pos_grd, device=pmask_img.device)

        memory_cs = aim_rst["correction_signal"] + raw_sentences["feature"]
        pmask_rs = raw_sentences["padding_mask"]

        # 6. grounding decoder
        #    1)multimodal fusion
        memory_cot = torch.cat([memory_img, memory_cs], dim=0)
        pmask_cot = torch.cat([pmask_img, pmask_rs], dim=1)
        pos_cot = torch.cat([pos_img, torch.zeros_like(memory_cs)], dim=0)
        memory_cot = self.model.encoder(memory_cot, src_key_padding_mask=pmask_cot, pos=pos_cot)
        memory_cs2 = memory_cot[-len(memory_cs):]

        # 2) grounding decoder
        # grd decoder
        grounding_result = self.model.decoder(memory_grd, memory_cot, memory_cs2,
                                              memory_key_padding_mask=pmask_cot,
                                              text_memory_key_padding_mask=pmask_rs, pos=pos_cot,
                                              query_pos=pos_grd)
        grounding_result = grounding_result.transpose(1, 2)

        ouput = {"grounding_result": grounding_result,
                 "concatenated_feature": memory_cs2}
        return ouput


@DECODERS.register_module()
class FCTRDecoder(nn.Module):
    def __init__(self,
                 correction_task: DictConfig,
                 rationale_task: DictConfig,
                 grounding_task: DictConfig,
                 emlm_task: DictConfig = None):
        super(FCTRDecoder, self).__init__()
        self.exp_cor_task = build_decoder(correction_task)
        self.rationale_task = build_decoder(rationale_task)
        self.grd_task = build_decoder(grounding_task)
        self.emlm_task = build_decoder(emlm_task) if emlm_task else None  # Elastic Mask language modeling

    def forward(self, fctr_encoder_result: Dict, rationales: Optional[List] = None, target: Optional[Dict] = None):
        if self.training:
            return self.forward_train(fctr_encoder_result, rationales, target)
        else:
            return self.forward_test(fctr_encoder_result)

    def forward_train(self, fctr_encoder_result: Dict, rationales: List, target: Dict):
        correction_rst = self.exp_cor_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                           aim_rst=fctr_encoder_result["attentional_interaction"],
                                           target_sentences=fctr_encoder_result["target_sentence"])

        rationale_rst = self.rationale_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                            aim_rst=fctr_encoder_result["attentional_interaction"],
                                            aligned_feat=fctr_encoder_result["aligned_feat"],
                                            target_sentences=fctr_encoder_result["target_sentence"],
                                            rationales=rationales)
        grd_rst = self.grd_task(visual_feat=fctr_encoder_result["visual"],
                                aim_rst=fctr_encoder_result["attentional_interaction"],
                                raw_sentences=fctr_encoder_result["raw_sentence"],
                                target=target)

        # add elastic mask language model task
        if self.emlm_task:
            emel_rst = self.emlm_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                      aim_rst=fctr_encoder_result["attentional_interaction"],
                                      target_sentences=fctr_encoder_result["target_sentence"])

        # output
        update_dict: Callable = lambda task, dict_data: {f"{task}_{k}": v for k, v in dict_data.items()}
        output = update_dict("expression_correction_task", correction_rst)
        output.update(update_dict("rationale_task", rationale_rst))
        [output.update(update_dict(f"grounding_task_{k}", v)) for k, v in grd_rst.items()]

        # ad elastic mask language model loss
        if self.emlm_task:
            output.update(update_dict("elastic_mask_language_task", emel_rst))

        return output

    def forward_test(self, fctr_encoder_result: Dict) -> Dict:
        correction_rst = self.exp_cor_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                           aim_rst=fctr_encoder_result["attentional_interaction"])
        rationale_rst = self.rationale_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                            aim_rst=fctr_encoder_result["attentional_interaction"],
                                            aligned_feat=fctr_encoder_result["aligned_feat"])
        grd_rst = self.grd_task(visual_feat=fctr_encoder_result["visual"],
                                aim_rst=fctr_encoder_result["attentional_interaction"],
                                raw_sentences=fctr_encoder_result["raw_sentence"])

        # outout
        return {"expression_correction": correction_rst,
                "rationale": rationale_rst,
                "grounding": grd_rst}
