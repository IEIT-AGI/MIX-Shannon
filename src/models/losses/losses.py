import torch
from torch import nn
from omegaconf import DictConfig
from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from torch.nn import functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any
from src.models.builder import LOSSES


def get_src_permutation_idx(indices: List) -> Tuple[List, List]:  # todo low
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx.tolist(), src_idx.tolist()


@LOSSES.register_module()
class L1GIoULoss(nn.Module):
    def __init__(self, loss_l1_weight: float = 1, loss_giou_weight: float = 1):
        super(L1GIoULoss, self).__init__()
        self.loss_l1_weight = loss_l1_weight
        self.loss_giou_weight = loss_giou_weight

    def forward(self, pred_bbox: torch.Tensor,
                gt_bbox: List,
                indices: List) -> Dict:
        idx = get_src_permutation_idx(indices)
        _pred_bbox = pred_bbox[idx]

        tgt_boxes = [gt_box[i] for gt_box, (_, i) in zip(gt_bbox, indices)]
        tgt_boxes = torch.cat(tgt_boxes, dim=0)

        loss_bbox = F.l1_loss(_pred_bbox, tgt_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() * self.loss_l1_weight

        _pred_bbox = box_cxcywh_to_xyxy(_pred_bbox)
        tgt_boxes = box_cxcywh_to_xyxy(tgt_boxes)
        loss_giou = 1 - torch.diag(generalized_box_iou(_pred_bbox, tgt_boxes))
        loss_giou = loss_giou.sum() * self.loss_giou_weight

        return {"loss_bbox_L1": loss_bbox, "loss_bbox_giou": loss_giou}


@LOSSES.register_module()
class SoftTokenLoss(nn.Module):
    def __init__(self, eos_coef: float = 0.1, loss_weight: float = 1):
        super(SoftTokenLoss, self).__init__()
        self.eos_coef = eos_coef  # relative classification weight applied to the no-object category
        self.loss_weight = loss_weight

    @staticmethod
    def _get_gt_idx(gt_bbox: List,
                    indices: List) -> torch.Tensor:
        # todo
        offset = 0
        tgt_idx = []
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(gt_bbox[i])
        tgt_idx = torch.cat(tgt_idx)
        return tgt_idx

    def forward(self, pred_logic: torch.Tensor,
                gt_bbox: List,
                indices: List,
                positive_map: torch.Tensor) -> Dict:
        logics = pred_logic.log_softmax(-1)
        src_idx = get_src_permutation_idx(indices)

        tgt_idx = self._get_gt_idx(gt_bbox, indices)

        target_sim = torch.zeros_like(logics)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = positive_map[tgt_idx]
        loss_ce = -(logics * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum()
        loss_ce *= self.loss_weight

        return {"loss_label_ce": loss_ce}


@LOSSES.register_module()
class FCTRCaptionLoss(nn.Module):
    def __init__(self,
                 token_loss: DictConfig,
                 focus_loss: DictConfig,
                 token_loss_weight: Optional[float] = 1.0,
                 focus_loss_weight: Optional[float] = 2.0):
        super(FCTRCaptionLoss, self).__init__()
        self.token_loss_weight = token_loss_weight
        self.focus_loss_weight = focus_loss_weight

        import hydra
        self.token_loss: Callable = hydra.utils.instantiate(token_loss)
        self.focus_loss: Callable = hydra.utils.instantiate(focus_loss)

    def forward(self, predict: torch.Tensor, target: Any, target_focus: List):
        token_loss = self._calculate_token_loss(predict, target)
        focus_loss = self._calculate_focus_loss(predict, target, target_focus)

        token_loss *= self.token_loss_weight
        focus_loss *= self.focus_loss_weight

        return {"caption_token_loss": token_loss,
                "caption_focus_loss": focus_loss}  # todo "caption_Loss": token_loss + focus_loss,

    def _calculate_token_loss(self, predict: torch.Tensor, target: Any) -> torch.Tensor:
        _predict = predict[:-1]
        _target = target.data["input_ids"].t()[1:]
        _input = _predict.view(-1, _predict.shape[-1]).contiguous()

        return self.token_loss(_input, _target.reshape(-1))

    def _calculate_focus_loss(self, predict: torch.Tensor, target: Any, focus: List) -> torch.Tensor:
        _predict = predict[:-1]
        _target = target.data["input_ids"].t()[1:]

        _predict_focus = []
        _target_focus = []
        for idx, fo in enumerate(focus):
            if len(fo) == 0:
                continue
            _predict_focus.append(_predict[fo - 1, idx])
            _target_focus.append(_target[fo - 1, idx])

        if len(_target_focus) > 0:
            _input = torch.cat(_predict_focus)
            _target = torch.cat(_target_focus)
            return self.focus_loss(_input.contiguous(), _target)
        else:
            return torch.tensor(0, device=_predict.device)


@LOSSES.register_module()
class CardinalityLoss(nn.Module):
    def __init__(self, loss_weight: float = 1):
        super(CardinalityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_fun: Callable = F.l1_loss

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> Dict:
        pred = (predict.argmax(-1) != predict.shape[-1] - 1).sum(1)
        _tgt = torch.tensor(data=[len(gt_bb) for gt_bb in target],
                            device=predict.device,
                            dtype=predict.dtype)
        loss = self.loss_fun(pred.float(), _tgt.float()) * self.loss_weight
        return {"cardinality_loss": loss}


@LOSSES.register_module()
class ContrastiveAlignLoss(nn.Module):
    def __init__(self,
                 temperature: float = 1,
                 box_to_token_loss_weight: float = 1,
                 token_to_box_loss_weight: float = 1):
        super(ContrastiveAlignLoss, self).__init__()
        self.box_to_token_loss_weight = box_to_token_loss_weight
        self.token_to_box_loss_weight = token_to_box_loss_weight
        self.temperature = temperature

    def forward(self, predict: Dict, tgt_positive_map: List, indices: List) -> Dict:
        def calculate_logics():
            img_embedding = predict["proj_queries"]
            text_embedding = predict["proj_tokens"]

            return torch.matmul(img_embedding, text_embedding.transpose(-1, -2))

        logics = (calculate_logics() / self.temperature)
        positive_map = self._get_positive_map(predict["tokenized"],
                                              torch.zeros(logics.shape, dtype=torch.bool),
                                              indices,
                                              tgt_positive_map)

        positive_map = positive_map.to(logics.device)
        positive_logics = -logics.masked_fill(~positive_map, 0)
        negative_logics = logics

        # box_to_token_loss
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logics.sum(2)
        neg_term = negative_logics.logsumexp(2)
        nb_pos = positive_map.sum(2) + 1e-6
        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()
        box_to_token_loss *= self.box_to_token_loss_weight

        # tokens_to_boxes_loss
        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logics.sum(1)
        neg_term = negative_logics.logsumexp(1)
        nb_pos = positive_map.sum(1) + 1e-6
        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tokens_to_boxes_loss *= self.token_to_box_loss_weight

        # loss_sum
        loss = (box_to_token_loss + tokens_to_boxes_loss) / 2
        return {"contrastive_align_loss": loss}

    def _get_positive_map(self,
                          tokenized: "BatchEncoding",
                          positive_map: torch.Tensor,
                          indices: List,
                          tokens_positive: List) -> torch.Tensor:

        for i, ((idx_src, idx_tgt), tokens) in enumerate(zip(indices, tokens_positive)):
            cur_tokens = [tokens[j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)

                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None

                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[i, idx_src[j], beg_pos: end_pos + 1].fill_(True)

        return positive_map


@LOSSES.register_module()
class FCTRGroundingLoss(nn.Module):
    def __init__(self,
                 label_loss: DictConfig,
                 bbox_loss: DictConfig,
                 cardinality_loss: DictConfig,
                 contrastive_align_loss: DictConfig,
                 matcher: DictConfig):
        super(FCTRGroundingLoss, self).__init__()

        from src.models.builder import build_loss, build_matcher
        self.label_loss: Callable = build_loss(label_loss)
        self.bbox_loss: Callable = build_loss(bbox_loss)
        self.cardinality_loss: Callable = build_loss(cardinality_loss)
        self.contrastive_align_loss: Callable = build_loss(contrastive_align_loss)
        self.matcher: Callable = build_matcher(matcher)

    def forward(self, predict: Dict, target: Dict) -> Dict:
        _tgt = {"boxes": target["boxes"], "positive_map": target["positive_map_raw"]}
        indices: List = self.matcher(predict, _tgt)

        label_loss = self._calculate_label_loss(predict, target, indices)
        bbox_loss = self._calculate_bbox_loss(predict, target, indices)
        cardinality_loss = self._calculate_cardinality_loss(predict, target)
        contrastive_align_loss = self._calculate_contrastive_align_loss(predict, target, indices)

        num_boxes = torch.tensor(len(_tgt["boxes"]), device=_tgt["positive_map"].device)
        divide_num_boxes: Callable = lambda loss_dict: {k: v / num_boxes for k, v in loss_dict.items()}

        output = {}
        output.update(divide_num_boxes(label_loss))
        output.update(divide_num_boxes(bbox_loss))
        output.update(cardinality_loss)
        output.update(divide_num_boxes(contrastive_align_loss))
        return output

    def _calculate_cardinality_loss(self, predict: Dict, target: Dict) -> Dict:
        pred_logics = predict["pred_logics"]
        gt_bbox = target["boxes"]
        return self.cardinality_loss(pred_logics, gt_bbox)

    def _calculate_label_loss(self, predict: Dict, target: Dict, matcher_indices: List) -> Dict:
        return self.label_loss(predict["pred_logics"],
                               target["boxes"],
                               matcher_indices,
                               target["positive_map_raw"])

    def _calculate_bbox_loss(self, predict: Dict, target: Dict, matcher_indices: List) -> Dict:
        return self.bbox_loss(predict["pred_boxes"],
                              target["boxes"],
                              matcher_indices)

    def _calculate_contrastive_align_loss(self, predict: Dict, target: Dict, matcher_indices: List) -> Dict:
        return self.contrastive_align_loss(predict,
                                           target['tokens_positive_raw'],
                                           matcher_indices)
