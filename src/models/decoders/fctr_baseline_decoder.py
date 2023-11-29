import hydra.utils
from torch import nn
from src.models.builder import DECODERS, build_decoder, build_encoder, build_head
from omegaconf import DictConfig
from typing import Dict, List, Optional, Callable, Union, Tuple
import torch
from collections import OrderedDict
from transformers import BartTokenizer


@DECODERS.register_module()
class RationaleGenerationBaselineTask(nn.Module):
    def __init__(self, decoder: DictConfig, header: DictConfig):
        super(RationaleGenerationBaselineTask, self).__init__()
        self.decoder = build_decoder(decoder)
        self.header = build_head(header)

    def forward(self,
                raw_sentences: Dict,
                aligned_feat: Dict,
                token_pos_embedding_fun: Callable,
                word_embeddings_fun: Callable,
                target_sentences: Optional[Dict] = None,
                rationales: Optional[List] = None):
        if self.training:
            return self.forward_train(raw_sentences, aligned_feat, token_pos_embedding_fun,
                                      word_embeddings_fun, target_sentences, rationales)
        else:
            return self.forward_test(raw_sentences, aligned_feat, token_pos_embedding_fun,
                                     word_embeddings_fun)

    def _decode_train(self,
                      aligned_feat,
                      token_pos_embedding_fun,
                      word_embeddings_fun,
                      rationales):

        memory_rat = aligned_feat["feature"]
        pmask_rat = aligned_feat["padding_mask"]
        pos_rat = aligned_feat["position"]
        device = aligned_feat['feature'].device

        tokenized_xs = self.decoder.tokenizer.batch_encode_plus(rationales, padding="longest", return_tensors="pt").to(
            device)
        tok_x = tokenized_xs.data["input_ids"].shape[1]
        tokens_xs = tokenized_xs.data["input_ids"]

        pos_xs = token_pos_embedding_fun(torch.arange(tok_x).to(device), len(rationales))
        pmask_xs = tokenized_xs.attention_mask.ne(1).bool()
        emb_xs = word_embeddings_fun(tokens_xs).transpose(0, 1)
        tokens_prob_xs = self.decoder(memory_rat, pmask_rat, pos_rat, emb_xs, pmask_xs, pos_xs)

        output = {"tokens_prob": tokens_prob_xs,
                  "rationales_token": tokenized_xs}

        return output

    def _generate_caption(self, aligned_feat, token_pos_embedding_fun,
                          word_embeddings_fun):

        memory_rat = aligned_feat["feature"]
        pmask_rat = aligned_feat["padding_mask"]
        pos_rat = aligned_feat["position"]

        device = memory_rat.device
        bs = pmask_rat.shape[0]
        tokens_xs = torch.zeros([bs, 1], dtype=torch.long, device=device)

        sum_eos: Callable = lambda tokens: sum([self.decoder.tokenizer.eos_token_id in tk_ids for tk_ids in tokens])
        while sum_eos(tokens_xs) < bs and tokens_xs.shape[1] < 64:
            # pos_xs = self.decoder.token_pos_eb(torch.arange(tokens_xs.shape[1], device=device), bs)
            pos_xs = token_pos_embedding_fun(torch.arange(tokens_xs.shape[1], device=device), bs)
            pmask_xs = torch.zeros_like(tokens_xs, dtype=torch.bool, device=device)
            # emb_xs = self.decoder.word_embeddings(tokens_xs).transpose(0, 1)
            emb_xs = word_embeddings_fun(tokens_xs).transpose(0, 1)
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
                      aligned_feat: Dict,
                      token_pos_embedding_fun: Callable,
                      word_embeddings_fun: Callable,
                      target_sentences: Optional[Dict] = None,
                      rationales: Optional[List] = None):
        decode_rst = self._decode_train(aligned_feat, token_pos_embedding_fun,
                                        word_embeddings_fun, rationales)
        return self.header(raw_sentences, target_sentences, decode_rst)

    def forward_test(self,
                     raw_sentences: Dict,
                     aligned_feat: Dict,
                     token_pos_embedding_fun: Callable,
                     word_embeddings_fun: Callable
                     ):
        caption_token = self._generate_caption(aligned_feat,
                                               token_pos_embedding_fun,
                                               word_embeddings_fun)
        predict_sentences = self.convert_token(caption_token, self.decoder.tokenizer)

        # output
        return {"caption_token": caption_token,
                "predict_sentences": predict_sentences}


@DECODERS.register_module()
class ExpressionCorrectionBaselineTask(nn.Module):
    def __init__(self, decoder: DictConfig, header: DictConfig):
        super(ExpressionCorrectionBaselineTask, self).__init__()
        self.decoder = build_decoder(decoder)
        self.header = build_head(header)

    def forward(self,
                raw_sentences: Dict,
                aligned_feat: Dict,
                token_pos_embedding_fun: Callable,
                word_embeddings_fun: Callable,
                target_sentences: Optional[Dict] = None):
        if self.training:
            return self.forward_train(raw_sentences,
                                      aligned_feat,
                                      token_pos_embedding_fun,
                                      word_embeddings_fun,
                                      target_sentences)
        else:
            return self.forward_test(raw_sentences,
                                     aligned_feat,
                                     token_pos_embedding_fun,
                                     word_embeddings_fun)

    def _decode_train(self,
                      aligned_feat: Dict,
                      token_pos_embedding_fun: Callable,
                      word_embeddings_fun: Callable,
                      target_sentences: Dict):

        memory_cs = aligned_feat["feature"]
        pos_rs = aligned_feat["position"]
        pmask_cs = aligned_feat["padding_mask"]

        tok_t, bs, _ = target_sentences["feature"].shape
        tokens_ts = target_sentences["token"].data["input_ids"]  # target sentence token
        padding_mask_ts = target_sentences["padding_mask"]

        # pos_ts = self.decoder.token_pos_eb(torch.arange(tokens_ts.shape[1]).to(pmask_cs.device), bs)
        # emb_ts = self.decoder.word_embeddings(tokens_ts).transpose(0, 1)
        pos_ts = token_pos_embedding_fun(torch.arange(tokens_ts.shape[1]).to(pmask_cs.device), bs)
        emb_ts = word_embeddings_fun(tokens_ts).transpose(0, 1)

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

    def _generate_caption(self,
                          aligned_feat: Dict,
                          token_pos_embedding_fun: Callable,
                          word_embeddings_fun: Callable):

        memory_cs = aligned_feat["feature"]
        pos_rs = aligned_feat["position"]
        pmask_cs = aligned_feat["padding_mask"]
        device = pmask_cs.device

        bs = pmask_cs.shape[0]
        tokens_ts = torch.zeros([bs, 1], dtype=torch.long, device=device)

        sum_eos: Callable = lambda tokens: sum([self.decoder.tokenizer.eos_token_id in tk_ids for tk_ids in tokens])
        while sum_eos(tokens_ts) < bs and tokens_ts.shape[1] < len(memory_cs) * 2:
            # pos_ts = self.decoder.token_pos_eb(torch.arange(tokens_ts.shape[1]).to(device), bs)
            # emb_ts = self.decoder.word_embeddings(tokens_ts).transpose(0, 1)
            pos_ts = token_pos_embedding_fun(torch.arange(tokens_ts.shape[1]).to(device), bs)
            emb_ts = word_embeddings_fun(tokens_ts).transpose(0, 1)
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

    def forward_train(self,
                      raw_sentences: Dict,
                      aligned_feat: Dict,
                      token_pos_embedding_fun: Callable,
                      word_embeddings_fun: Callable,
                      target_sentences: Dict) -> Dict:
        decode_rst = self._decode_train(aligned_feat, token_pos_embedding_fun, word_embeddings_fun,
                                        target_sentences)
        return self.header(raw_sentences, None, target_sentences, decode_rst)

    def forward_test(self,
                     raw_sentences: Dict,
                     aligned_feat: Dict,
                     token_pos_embedding_fun: Callable,
                     word_embeddings_fun: Callable) -> Dict:
        caption_token, corrected_feature = self._generate_caption(aligned_feat, token_pos_embedding_fun,
                                                                  word_embeddings_fun)
        predict_sentences = RationaleGenerationBaselineTask.convert_token(caption_token, self.decoder.tokenizer)

        return {"caption_token": caption_token,
                "corrected_feature": corrected_feature,
                "predict_sentences": predict_sentences}


@DECODERS.register_module()
class GroundingBaselineTask(nn.Module):
    def __init__(self, decoder: Union[DictConfig, nn.Module],
                 header: DictConfig,
                 query_embed: DictConfig):
        super(GroundingBaselineTask, self).__init__()

        self.model = self._build_model(decoder)
        self.header = build_head(header)
        self.query_embed = hydra.utils.instantiate(query_embed)

    @staticmethod
    def _build_model(decoder: Union[DictConfig, nn.Module]) -> nn.Sequential:
        _decoder = build_decoder(decoder) if isinstance(decoder, DictConfig) else decoder
        return nn.Sequential(OrderedDict(decoder=_decoder))

    def forward(self, visual_feat: Dict, aligned_feat: Dict, raw_sentences: Dict,
                target: Optional[Dict] = None) -> Dict:
        if self.training:
            return self.forward_train(visual_feat, aligned_feat, raw_sentences, target)
        else:
            return self.forward_test(visual_feat, aligned_feat, raw_sentences)

    def forward_train(self, visual_feat: Dict, aligned_feat: Dict, raw_sentences: Dict, target: Dict) -> Dict:
        grd_tf = self._grounding_transform(visual_feat, aligned_feat, raw_sentences)
        return self.header(raw_sentences, grd_tf, target)

    def forward_test(self, visual_feat: Dict, aligned_feat: Dict, raw_sentences: Dict) -> Dict:
        grd_tf = self._grounding_transform(visual_feat, aligned_feat, raw_sentences)
        return self.header(raw_sentences, grd_tf)

    def _grounding_transform(self, visual_feat: Dict, aligned_feat: Dict, raw_sentences: Dict) -> Dict:
        # memory_img = visual_feat["feature"]
        # pos_img = visual_feat["position"]

        pmask_img = visual_feat["mask"]
        pos_grd = self.query_embed.weight
        pos_grd = pos_grd.unsqueeze(1).repeat(1, pmask_img.shape[0], 1)
        memory_grd = torch.zeros_like(pos_grd, device=pmask_img.device)

        pmask_rs = raw_sentences["padding_mask"]
        memory_cot = aligned_feat["feature"]
        pmask_cot = aligned_feat["padding_mask"]
        pos_cot = aligned_feat["position"]

        memory_cs2 = memory_cot[-len(raw_sentences["feature"]):]

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
class FCTRBaselineDecoder(nn.Module):
    def __init__(self,
                 correction_task: DictConfig,
                 rationale_task: DictConfig,
                 grounding_task: DictConfig):
        super(FCTRBaselineDecoder, self).__init__()
        self.exp_cor_task = build_decoder(correction_task)
        self.rationale_task = build_decoder(rationale_task)
        self.grd_task = build_decoder(grounding_task)

    def forward(self, fctr_encoder_result: Dict, rationales: Optional[List] = None, target: Optional[Dict] = None):
        if self.training:
            return self.forward_train(fctr_encoder_result, rationales, target)
        else:
            return self.forward_test(fctr_encoder_result)

    def forward_train(self, fctr_encoder_result: Dict, rationales: List, target: Dict):
        correction_rst = self.exp_cor_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                           aligned_feat=fctr_encoder_result["aligned_feat"],
                                           token_pos_embedding_fun=fctr_encoder_result['token_pos_embedding_fun'],
                                           word_embeddings_fun=fctr_encoder_result['word_embeddings_fun'],
                                           target_sentences=fctr_encoder_result["target_sentence"])

        rationale_rst = self.rationale_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                            aligned_feat=fctr_encoder_result["aligned_feat"],
                                            token_pos_embedding_fun=fctr_encoder_result['token_pos_embedding_fun'],
                                            word_embeddings_fun=fctr_encoder_result['word_embeddings_fun'],
                                            target_sentences=fctr_encoder_result["target_sentence"],
                                            rationales=rationales)
        grd_rst = self.grd_task(visual_feat=fctr_encoder_result["visual"],
                                aligned_feat=fctr_encoder_result["aligned_feat"],
                                raw_sentences=fctr_encoder_result["raw_sentence"],
                                target=target)

        # output
        update_dict: Callable = lambda task, dict_data: {f"{task}_{k}": v for k, v in dict_data.items()}
        output = update_dict("expression_correction_task", correction_rst)
        output.update(update_dict("rationale_task", rationale_rst))
        [output.update(update_dict(f"grounding_task_{k}", v)) for k, v in grd_rst.items()]

        return output

    def forward_test(self, fctr_encoder_result: Dict) -> Dict:
        correction_rst = self.exp_cor_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                           aligned_feat=fctr_encoder_result["aligned_feat"],
                                           token_pos_embedding_fun=fctr_encoder_result['token_pos_embedding_fun'],
                                           word_embeddings_fun=fctr_encoder_result['word_embeddings_fun'],
                                           )
        rationale_rst = self.rationale_task(raw_sentences=fctr_encoder_result["raw_sentence"],
                                            aligned_feat=fctr_encoder_result["aligned_feat"],
                                            token_pos_embedding_fun=fctr_encoder_result['token_pos_embedding_fun'],
                                            word_embeddings_fun=fctr_encoder_result['word_embeddings_fun'],
                                            )
        grd_rst = self.grd_task(visual_feat=fctr_encoder_result["visual"],
                                aligned_feat=fctr_encoder_result["aligned_feat"],
                                raw_sentences=fctr_encoder_result["raw_sentence"])

        # outout
        return {"expression_correction": correction_rst,
                "rationale": rationale_rst,
                "grounding": grd_rst}
