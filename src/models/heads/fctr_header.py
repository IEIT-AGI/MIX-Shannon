from torch import nn
from omegaconf import DictConfig, OmegaConf
import torch
from typing import Dict, Callable, List, Optional, Tuple, Any
from src.models.builder import HEADS, build_head, build_matcher, build_loss, build_models
import hydra
from transformers import PreTrainedTokenizerBase
from collections import OrderedDict
from torch.nn import functional as F
import copy


def get_focus(tokens_r_: List,
              tokens_t_: List,
              tokens_x_: Optional[List] = None,
              tokenizer: PreTrainedTokenizerBase = None):
    if tokenizer is None:
        import os
        model_path = f"/home/{os.getlogin()}/.cache/torch/hub/transformers/roberta-base"
        tokenizer_cfg = OmegaConf.create({
            "_target_": "transformers.RobertaTokenizerFast.from_pretrained",
            "pretrained_model_name_or_path": model_path
        })  # todo
        tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    min_len = min(len(tokens_r_), len(tokens_t_))
    for ks in range(min_len):
        if tokens_r_[ks] != tokens_t_[ks]:
            break
    for ke in range(min_len):
        if tokens_r_[-1 - ke] != tokens_t_[-1 - ke]:
            break
    focus_ts = [ks, min(len(tokens_t_) - 1, max(len(tokens_t_) - ke, ks + 1))]
    tokens_fr_ = tokens_r_[focus_ts[0]:min(len(tokens_r_) - 1, max(len(tokens_r_) - ke, ks + 1))]
    tokens_ft_ = tokens_t_[focus_ts[0]:focus_ts[1]]

    ts_fo = [k for k in list(range(focus_ts[0], focus_ts[1])) if tokens_t_[k] not in tokenizer.all_special_ids]

    if tokens_x_ is None:
        return torch.tensor(ts_fo)
    else:
        tokens_s = [tok for tok in tokens_fr_ + tokens_ft_ if tok not in tokenizer.all_special_ids]
        xs_fo = [k for k in range(len(tokens_x_)) if tokens_x_[k] in tokens_s]
        return torch.tensor(ts_fo), torch.tensor(xs_fo)


def get_change_label(tokens_r_, tokens_t_):
    min_len = min(len(tokens_r_), len(tokens_t_))
    for ks in range(min_len):
        if tokens_r_[ks] != tokens_t_[ks]:
            break
    for ke in range(min_len):
        if tokens_r_[-1 - ke] != tokens_t_[-1 - ke]:
            break
    focus_ts = [ks, min(len(tokens_t_) - 1, max(len(tokens_t_) - ke, ks))]
    labels = torch.tensor([0 for _ in range(len(tokens_t_))])
    labels[focus_ts[0]:focus_ts[1]] = 1
    return labels


def get_change_info_from_sentence(raw_sentences: Dict,
                                  target_sentences: Dict,
                                  rationale_sentences: Optional["BatchEncoding"] = None) -> Tuple:
    raw_token = raw_sentences["token"]
    target_token = target_sentences["token"]
    device = raw_token['input_ids'].device

    if rationale_sentences is None:
        target_focus = []
        change_labels = []
        for r_tk, t_tk in zip(raw_token["input_ids"], target_token["input_ids"]):
            r_tk = r_tk.tolist()
            t_rk = t_tk.tolist()
            t_focus = get_focus(r_tk, t_tk)
            change_labels.append(get_change_label(r_tk, t_rk))
            target_focus.append(t_focus)
        change_labels = torch.stack(change_labels).t().to(device)
        return change_labels, target_focus
    else:
        rationale_token = rationale_sentences
        target_focus = []
        rationale_focus = []
        change_labels = []
        for r_tk, t_tk, rl_tk in zip(raw_token["input_ids"], target_token["input_ids"], rationale_token["input_ids"]):
            r_tk = r_tk.tolist()
            t_tk = t_tk.tolist()
            rl_tk = rl_tk.tolist()
            t_focus, rl_focus = get_focus(r_tk, t_tk, rl_tk)
            change_labels.append(get_change_label(r_tk, t_tk))
            target_focus.append(t_focus)
            rationale_focus.append(rl_focus)

        change_labels = torch.stack(change_labels).t().to(device)
        return change_labels, target_focus, rationale_focus


@HEADS.register_module()
class FCTRGroundingHeader(nn.Module):

    def __init__(self, grd_loss: DictConfig, grd_proj: DictConfig):
        super(FCTRGroundingHeader, self).__init__()
        self.grd_loss: Callable = build_loss(grd_loss)
        self.model = self._build_model(grd_proj)

    @staticmethod
    def _build_model(cfg: DictConfig) -> nn.Sequential:
        class_embed = hydra.utils.instantiate(cfg.class_embed)
        bbox_embed = build_models(cfg.bbox_embed)
        contrastive_align_img_proj = hydra.utils.instantiate(cfg.contrastive_align_img_proj)
        contrastive_align_text_proj = hydra.utils.instantiate(cfg.contrastive_align_text_proj)

        return nn.Sequential(
            OrderedDict(class_embed=class_embed,
                        bbox_embed=bbox_embed,
                        contrastive_align_img_proj=contrastive_align_img_proj,
                        contrastive_align_text_proj=contrastive_align_text_proj))

    def forward(self, raw_sentences: Dict, grd_rst: Dict, grd_target: Optional[Dict] = None):
        if self.training:
            return self.forward_train(raw_sentences, grd_rst, grd_target)
        else:
            return self.forward_test(raw_sentences, grd_rst)

    def forward_train(self, raw_sentences: Dict, grd_rst: Dict, grd_target: Dict) -> Dict:
        header_rst = self.forward_test(raw_sentences, grd_rst)
        grd_loss = {}
        for layer_idx, predict in header_rst.items():
            grd_loss[f"grd_header_{layer_idx}"] = self.grd_loss(predict, grd_target)

        return grd_loss

    def forward_test(self, raw_sentences: Dict, grd_rst: Dict) -> Dict:
        raw_sent_token = raw_sentences["token"]
        memory_cs2 = grd_rst["concatenated_feature"]  # multimodal fusion(grounding encoder) result
        grd_rst = grd_rst["grounding_result"]  # grounding decoder result

        # grounding header output
        obj_cls = self.model.class_embed(grd_rst)
        obj_coord = self.model.bbox_embed(grd_rst).sigmoid()

        # aux output
        align_img = self.model.contrastive_align_img_proj(grd_rst)
        align_text = self.model.contrastive_align_text_proj(memory_cs2)

        proj_queries = F.normalize(align_img, p=2, dim=-1)
        proj_tokens = F.normalize(align_text.transpose(0, 1), p=2, dim=-1)

        output = {}
        if self.training:
            for idx, cls, bbox, query in zip(range(obj_cls.shape[0]), obj_cls, obj_coord, proj_queries):
                output[f"layer{idx}"] = {
                    "pred_logics": cls,
                    "pred_boxes": bbox,
                    "proj_queries": query,
                    "proj_tokens": proj_tokens,
                    "tokenized": raw_sent_token
                }

            # grd_decode_inter_layer_outputs: List = []
            # for cls, bbox, query in zip(obj_cls[:-1], obj_coord[:-1], proj_queries[:-1]):
            #     grd_decode_inter_layer_outputs.append({"pred_logits": cls,
            #                                            "pred_boxes": bbox,
            #                                            "proj_queries": query,
            #                                            "proj_tokens": proj_tokens,
            #                                            "tokenized": raw_sent_token
            #                                            })
            #
            # grd_header_output = {"object_class": obj_cls,
            #                      "object_bbox": obj_coord
            #                      }
            #
            # output = {"grd_decode_inter_layer_outputs": grd_decode_inter_layer_outputs,
            #           "grounding_header_output": grd_header_output}
        else:
            # output["pred_logits"] = obj_cls[-1]
            # output["pred_boxes"] = obj_coord[-1]
            # output["proj_queries"] = proj_queries[-1]
            # output["proj_tokens"] = proj_tokens
            # output["tokenized"] = raw_sent_token

            pred_logits = obj_cls[-1]
            pred_boxes = obj_coord[-1]

            prob = F.softmax(pred_logits, -1)
            scores, labels = prob[..., :-1].max(-1)
            labels = torch.ones_like(labels)
            scores = 1 - prob[:, :, -1]

            output["pred_scores"] = scores
            output["pred_labels"] = labels
            output["pred_boxes"] = pred_boxes

        return output


@HEADS.register_module()
class FCTRRationaleHeader(nn.Module):

    def __init__(self, caption_loss: DictConfig, caption_loss_weight: Optional[float] = 1.0):
        super(FCTRRationaleHeader, self).__init__()

        self.caption_loss: Callable = build_loss(caption_loss)
        self.caption_loss_weight = caption_loss_weight

    def forward(self, raw_sentences: Dict, target_sentences: Dict, rationale_decode_rst: Dict) -> Dict:
        _, _, rationale_focus = get_change_info_from_sentence(raw_sentences, target_sentences,
                                                              rationale_decode_rst["rationales_token"])
        caption_loss: Dict = self.caption_loss(predict=rationale_decode_rst["tokens_prob"],
                                               target=rationale_decode_rst["rationales_token"],
                                               target_focus=rationale_focus)
        caption_loss = {k: v * self.caption_loss_weight for k, v in caption_loss.items()}

        return caption_loss


@HEADS.register_module()
class EMLMHeader(nn.Module):

    def __init__(self, loss_fun: DictConfig, loss_weight: Optional[float] = 1.0):
        super(EMLMHeader, self).__init__()
        self.loss_fun: Callable = hydra.utils.instantiate(loss_fun)  # cross_entropy
        self.loss_weight = loss_weight

        from difflib import Differ
        self.differ: Differ = Differ()

    def forward(self, current_token_prob: torch.Tensor, before_tokens_prob: torch.Tensor, raw_sentences: Dict,
                target_sentences: Dict, tokenizer: 'tokenizer') -> Dict:
        current_tokens, current_mask, before_tokens, before_mask = self._get_labels(raw_sentences.get("token"),
                                                                                    target_sentences.get("token"),
                                                                                    tokenizer)

        # labels
        _current_tokens = torch.tensor(current_tokens).reshape(-1)
        _before_tokens = torch.tensor(before_tokens).reshape(-1)

        _current_mask = torch.tensor(current_mask).reshape(-1)
        _before_mask = torch.tensor(before_mask).reshape(-1)

        # prob
        _current_token_prob = current_token_prob.reshape(-1, current_token_prob.shape[-1]).contiguous()
        _before_token_prob = before_tokens_prob.reshape(-1, before_tokens_prob.shape[-1]).contiguous()

        # loss
        current_tokens_loss = self._calculate_loss(target_tokens=_current_tokens,
                                                   prob_tokens=_current_token_prob,
                                                   mask=_current_mask)

        before_tokens_loss = self._calculate_loss(target_tokens=_before_tokens,
                                                  prob_tokens=_before_token_prob,
                                                  mask=_before_mask)
        loss = current_tokens_loss + before_tokens_loss
        return {"emlm": loss * self.loss_weight}  # elastic_mask_language_loss

    def _calculate_loss(self, target_tokens, prob_tokens, mask) -> torch.Tensor:

        def get_position() -> torch.Tensor:
            import random
            sample = random.sample(torch.where(mask > 0)[0].tolist(), int(len(mask) * 0.2))
            pos = sample + torch.where(mask > 1)[0].tolist()
            return torch.tensor(list(set(pos)), dtype=torch.long)

        positions = get_position()
        tgt_tokens = target_tokens[positions]
        _prob_tokens = prob_tokens[positions]

        return self.loss_fun(_prob_tokens, tgt_tokens.to(_prob_tokens.device))

    def _get_labels(self, raw_sent_token, target_sent_token, tokenizer):
        current_tokens_list = []
        before_tokens_list = []
        current_mask_list = []
        before_mask_list = []
        pad_token_str = str(tokenizer.pad_token_id)
        for idx in range(len(raw_sent_token)):
            raw_str = list(map(str, copy.deepcopy(raw_sent_token[idx].ids)))
            tar_str = list(map(str, copy.deepcopy(target_sent_token[idx].ids)))
            diff = list(map(lambda x: x.strip(), list(self.differ.compare(raw_str, tar_str))))
            diff = [di for di in diff if not (di in ["+ 1", "- 1"])]
            current_tokens_str = copy.deepcopy(raw_str)
            before_tokens_str = ["0"] + copy.deepcopy(raw_str)[:-1]
            changes_remove = [di.strip("-") for di in diff if di.startswith("-")]
            changes_add = [di.strip("+") for di in diff if di.startswith("+")]
            change_start = 0
            for idx_t, t in enumerate(diff):
                if t.startswith("+") or t.startswith("-"):
                    change_start = idx_t
                    break
            if len(changes_remove) > len(changes_add):
                changes_add = copy.deepcopy([pad_token_str] * (len(changes_remove) - len(changes_add)) + changes_add)

            current_tokens_str[change_start:change_start + len(changes_remove)] = changes_add[len(changes_add) -
                                                                                              len(changes_remove):]
            before_tokens_str[change_start + 1:change_start + len(changes_remove) +
                              1] = changes_add[len(changes_add) - len(changes_remove):]

            current_tokens = list(map(int, current_tokens_str))
            before_tokens = list(map(int, before_tokens_str))

            current_mask = copy.deepcopy([0] * len(current_tokens))
            current_mask[current_tokens.index(0) + 1:current_tokens.index(2)] = copy.deepcopy(
                [1] * (current_tokens.index(2) - current_tokens.index(0) - 1))
            current_mask[change_start:change_start + len(changes_remove)] = copy.deepcopy([2] * len(changes_remove))

            before_mask = copy.deepcopy([0] * len(before_tokens))
            before_mask[current_tokens.index(0) + 2:current_tokens.index(2) + 1] = copy.deepcopy(
                [1] * (current_tokens.index(2) - current_tokens.index(0) - 1))
            before_mask[change_start + 1:change_start + len(changes_remove) + 1] = copy.deepcopy([2] *
                                                                                                 len(changes_remove))

            current_tokens_list.append(current_tokens)
            before_tokens_list.append(before_tokens)

            current_mask_list.append(current_mask)
            before_mask_list.append(before_mask)

        return current_tokens_list, current_mask_list, before_tokens_list, before_mask_list


@HEADS.register_module()
class FCTRCorrectionHeader(nn.Module):

    def __init__(self,
                 caption_loss: DictConfig,
                 modify_words_loss: DictConfig,
                 feature_loss: DictConfig,
                 caption_loss_weight: Optional[float] = 1.0,
                 modify_words_loss_weight: Optional[float] = 1.0,
                 feature_loss_weight: Optional[float] = 1.0):
        super(FCTRCorrectionHeader, self).__init__()

        self.caption_loss: Callable = build_loss(caption_loss)
        self.modify_words_loss: Callable = hydra.utils.instantiate(modify_words_loss)
        self.feature_loss: Callable = hydra.utils.instantiate(feature_loss)

        self.caption_loss_weight = caption_loss_weight
        self.modify_words_loss_weight = modify_words_loss_weight
        self.feature_loss_weight = feature_loss_weight

    def forward(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict, correction_decode_rst: Dict):
        if self.training:
            return self.forward_train(raw_sentences, aim_rst, target_sentences, correction_decode_rst)
        else:
            return self.forward_test(raw_sentences, aim_rst, target_sentences, correction_decode_rst)

    def forward_train(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict,
                      correction_decode_rst: Dict) -> Dict:
        return self._calculate_loss(raw_sentences, aim_rst, target_sentences, correction_decode_rst)

    def forward_test(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict, correction_decode_rst: Dict):
        pass

    def _calculate_feature_loss(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict) -> torch.Tensor:
        memory_rs, pmask_rs = raw_sentences["feature"], raw_sentences["padding_mask"]
        memory_ts, pmask_ts = target_sentences["feature"], target_sentences["padding_mask"]

        correction_signal = aim_rst["correction_signal"]

        raw_feature = memory_rs * ~pmask_rs.t().unsqueeze(-1)
        target_feature = memory_ts * ~pmask_ts.t().unsqueeze(-1)
        correction_feature = correction_signal * ~pmask_rs.t().unsqueeze(-1)

        transpose: Callable = lambda feat: feat.transpose(0, 1).sum(1)
        return self.feature_loss(transpose(correction_feature), transpose(target_feature) - transpose(raw_feature))

    def _calculate_modify_words_loss(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _predict = predict[:-1]
        _target = target[1:]

        _input = _predict.view(-1, _predict.shape[-1]).contiguous()
        return self.modify_words_loss(_input, _target.reshape(-1))

    def _calculate_loss(self, raw_sentences: Dict, aim_rst: Dict, target_sentences: Dict,
                        correction_decode_rst: Dict) -> Dict:
        change_labels, target_sent_focus = get_change_info_from_sentence(raw_sentences, target_sentences)
        caption_loss: Dict = self.caption_loss(predict=correction_decode_rst["cross_token_prob"],
                                               target=target_sentences["token"],
                                               target_focus=target_sent_focus)
        modify_words_loss = self._calculate_modify_words_loss(predict=correction_decode_rst["token_change_prob"],
                                                              target=change_labels)
        caption_feature_loss = self._calculate_feature_loss(raw_sentences, aim_rst, target_sentences)

        caption_loss = {k: v * self.caption_loss_weight for k, v in caption_loss.items()}
        modify_words_loss *= self.modify_words_loss_weight
        caption_feature_loss *= self.feature_loss_weight

        # output
        output = {"modify_words_loss": modify_words_loss, "caption_feature_loss": caption_feature_loss}
        output.update(caption_loss)
        return output
