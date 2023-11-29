from typing import Dict, List, Callable, Tuple
import numpy as np
from omegaconf import DictConfig
import torch
from torch.nn import functional as F
from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .cider import CIDEr


class CorrectionEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cider_eval: CIDEr = CIDEr(**cfg.cider)

    def __call__(self, cache: Dict, predict_name: str, target_name: str) -> Tuple[Dict, Dict]:
        cider_avg, cider_score = self.cider_eval(cache, predict_name, target_name)
        contribute: Dict = self._compute_contribute(cache)
        cider_scores, hit_scores, max_f_scores_idx = self._get_cider_and_hit_score(cider_score, contribute)

        # output
        average_val: Callable = lambda data_dict: float(np.mean(list(data_dict.values())))
        eval_rst = {"cider_score": average_val(cider_scores),
                    "hit_scores": average_val(hit_scores)}
        return eval_rst, max_f_scores_idx

    def _compute_contribute(self, cache: Dict) -> Dict:
        cor_score_dict = {}
        for name, cor_info in cache.items():
            cor_score_dict[name] = [self.get_hit_score(change_info) for change_info in cor_info["change_infos"]]
        return cor_score_dict

    @staticmethod
    def get_hit_score(hit_dict: Dict) -> Dict:
        label, pred = hit_dict.get("label"), hit_dict.get("pred")

        if len(pred) == 0 or len(label) == 0:
            eq_val = {"f_score": 1.0, "precision": 1.0, "recall": 1.0}
            no_eq_val = {"f_score": 0.0, "precision": 0.0, "recall": 0.0}
            return eq_val if len(pred) == len(label) else no_eq_val
        else:
            correct = [h for h in pred if h in label]
            recall = float(len(correct)) / len(label)
            precision = float(len(correct)) / len(pred)
            f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            return {"f_score": f_score, "precision": precision, "recall": recall}

    @staticmethod
    def _get_cider_and_hit_score(cider_scores: Dict, hit_scores: Dict) -> Tuple[Dict, Dict, Dict]:
        _cider_scores = {}
        _hit_scores = {}
        max_f_scores_idxs = {}
        for name, hit_score in hit_scores.items():
            idx = np.array([hs["f_score"] for hs in hit_score]).argmax()
            _hit_scores[name] = hit_score[idx]["f_score"]
            _cider_scores[name] = cider_scores[name][idx]
            max_f_scores_idxs[name] = idx

        return _cider_scores, _hit_scores, max_f_scores_idxs


class RationaleEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cider_eval: CIDEr = CIDEr(**cfg.cider)

    def __call__(self, cache: Dict, predict_name: str, target_name: str) -> Dict:
        cider_avg, cider_score = self.cider_eval(cache, predict_name, target_name)
        score = float(np.mean((list(cider_score.values()))))

        return {"rationale_cider_score": score}


class FRECGrounding:
    def __init__(self, cfg: DictConfig):
        self.rank = cfg.get("rank", (1, 5, 10))
        self.rank = eval(self.rank) if isinstance(self.rank, str) else self.rank
        self.iou_thresh = cfg.get("iou_thresh", 0.5)

    @staticmethod
    def convert(predict: Dict, target_size: torch.Tensor) -> List:
        pred_bbox = predict["pred_boxes"]
        boxes = box_cxcywh_to_xyxy(pred_bbox)
        img_h, img_w = target_size.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        scores = predict["pred_scores"]
        labels = predict["pred_labels"]

        assert len(scores) == len(labels) == len(boxes)

        # output
        return [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

    def _calculate_grd_eval(self, predict: List, target: Dict, max_f_score_idx: Dict) -> Dict:
        grd_cache = {}
        for pred, img_id, orig_size, gt_box, f_score_idx in zip(predict, target["name"], target["orig_size"],
                                                                target["boxes"], max_f_score_idx.values()):
            sorted_scores_boxes = sorted(zip(pred["scores"].tolist(), pred["boxes"].tolist()), reverse=True)
            sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])

            target_bbox = gt_box.tolist()  # todo what?
            src_h, src_w = orig_size.tolist()
            convert_gt_bbox = [[bbox[0] * src_w - bbox[2] * src_w / 2,
                                bbox[1] * src_h - bbox[3] * src_h / 2,
                                bbox[0] * src_w + bbox[2] * src_w / 2,
                                bbox[1] * src_h + bbox[3] * src_h / 2]
                               for bbox in target_bbox]
            gious = generalized_box_iou(sorted_boxes, torch.as_tensor(convert_gt_bbox))
            gious = gious[:, f_score_idx]
            grd_cache[img_id] = {k: int(max(gious[:k]) >= self.iou_thresh) for k in self.rank}

        grd_score = {k: 0.0 for k in self.rank}
        cnt = 0.0
        for name in list(grd_cache):
            cnt += 1
            for k in self.rank:
                grd_score[k] += grd_cache[name][k]
        return {"grounding_score": {f"recall@{k}": grd_score[k] / cnt for k in self.rank}}

    def __call__(self, predict: Dict, target: Dict, max_f_score_idx: Dict) -> Dict:
        predict: List = self.convert(predict, target["orig_size"])
        return self._calculate_grd_eval(predict, target, max_f_score_idx)


class FRECEvaluator:

    def __init__(self, cfg: DictConfig):
        self.rationale_eval = RationaleEvaluator(cfg.rationale_eval)
        self.correction_eval = CorrectionEvaluator(cfg.correction_eval)
        self.grd_eval = FRECGrounding(cfg.grounding_eval)

    def __call__(self, data: Dict) -> Dict:
        rationale_eval_rst = self._calculate_rationale_eval(data)
        correction_eval_rst, max_score_idx = self._calculate_correction_eval(data)
        grd_eval_rst = self._calculate_grounding_eval(data, max_score_idx)

        # output
        return {"rationale": rationale_eval_rst,
                "correction:": correction_eval_rst,
                "grounding": grd_eval_rst}

    def _calculate_rationale_eval(self, data: Dict) -> Dict:
        target_field = ["rationale_list"]
        predict_field = "predict_sentences"

        def get_rationale_data() -> Dict:
            rst = {}
            for k, v in data.items():
                rst[k] = {tgt_k: v[tgt_k] for tgt_k in target_field}
                rst[k].update(v["predict"].get("rationale"))
            return rst

        rationale_data = get_rationale_data()
        return self.rationale_eval(rationale_data, predict_field, target_field[0])

    @staticmethod
    def get_change(src_sent, target_sent):
        from difflib import Differ
        differ = Differ()
        diffs = list(differ.compare(src_sent.split(), target_sent.split()))
        changes = []
        c = []
        for diff in diffs:
            if diff.startswith("+") or diff.startswith("-"):
                c.append(diff)
            elif len(c) > 0:
                changes.append(c)
                c = []
        if len(c) > 0:
            changes.append(c)
        change_infos = []
        for c in changes:
            w_minus = [w.split(" ")[-1] for w in c if w.startswith("-")]
            w_add = [w.split(" ")[-1] for w in c if w.startswith("+")]
            if len(w_minus) == len(w_add):
                change_infos.extend([f"{mi}->{ad}" for mi, ad in zip(w_minus, w_add)])
            else:
                minus = " ".join(w_minus)
                add = " ".join(w_add)
                change_infos.append(f"{minus}->{add}")
        return change_infos

    def _calculate_correction_eval(self, data: Dict) -> Tuple[Dict, Dict]:
        target_field = ["cor_sent_list", "raw_sent"]
        predict_field = "predict_sentences"

        def get_change_infos(raw_sentence: str, target_sentence: List, predict_sentences: str) -> List:
            change_info: Callable = lambda raw_sent, target_sent: self.get_change(raw_sent, target_sent)
            change_pred = change_info(raw_sentence, predict_sentences)
            return [{"label": change_info(raw_sentence, tar_sent), "pred": change_pred} for tar_sent in target_sentence]

        def get_correction_data() -> Dict:
            rst = {}
            for k, v in data.items():
                rst[k] = {tgt_k: v[tgt_k] for tgt_k in target_field}
                expression_correction_rst = v["predict"].get("expression_correction")
                rst[k].update(expression_correction_rst)
                change_infos = get_change_infos(raw_sentence=rst[k].get(target_field[1]),
                                                target_sentence=rst[k].get(target_field[0]),
                                                predict_sentences=expression_correction_rst.get(predict_field))
                rst[k].update({"change_infos": change_infos})
            return rst

        correction_data = get_correction_data()
        return self.correction_eval(correction_data, predict_field, target_field[0])

    def _calculate_grounding_eval(self, data: Dict, max_score_idx: Dict):

        def get_predict_data() -> Dict:
            predicts = {k: v['predict']['grounding'] for k, v in data.items()}
            _predict = {}
            for name, pred in predicts.items():
                for k, v in pred.items():
                    if k in _predict:
                        _predict[k].append(v)
                    else:
                        _predict[k] = [v]

            rst = {}
            for k, v in _predict.items():
                rst[k] = torch.stack(v, dim=0)
            return rst

        def get_target_data() -> Dict:
            target_field = ["orig_size", "boxes", "name"]
            _target = {k: {v_k: v[v_k] for v_k in target_field} for k, v in data.items()}
            _tgt = {}
            for name, pred in _target.items():
                for k, v in pred.items():
                    if k in _tgt:
                        _tgt[k].append(v)
                    else:
                        _tgt[k] = [v]
            _tgt["orig_size"] = torch.stack(_tgt["orig_size"], dim=0)
            return _tgt

        # predict,target and max_score_idx are all derived from data , so they are in the same order
        target = get_target_data()
        predict = get_predict_data()

        return self.grd_eval(predict, target, max_score_idx)
