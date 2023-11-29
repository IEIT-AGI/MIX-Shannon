import torch
import numpy as np

from .cider import Cider
# from utils import box_ops, dist
from src.utils import box_ops, dist
from typing import Union, Dict, List


class LiefCocoEvaluator(object):
    def __init__(self, k=(1, 5, 10), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        self.k = k
        self.thresh_iou = thresh_iou
        self.labels = {}
        self.predictions = {}

    def accumulate(self):
        pass

    def update(self, predictions: Union[List, Dict]) -> None:
        if isinstance(predictions, List):
            for p in predictions:
                self.predictions.update(p)
        elif isinstance(predictions, Dict):
            self.predictions.update(predictions)
        else:
            assert TypeError('predictions is not list or dict type')

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def compute_contribute(self, cor_cache):
        cor_score_dict = {}
        for name, cor_info in cor_cache.items():
            cor_score_dict[name] = [self.get_hit_score(change_info) for change_info in cor_info["change_infos"]]
        return cor_score_dict

    def get_hit_score(self, hit_dict):
        label = hit_dict["label"]
        pred = hit_dict["pred"]
        if len(pred) == 0 or len(label) == 0:
            if len(pred) != len(label):
                return {"f_score": 0.0, "precision": 0.0, "recall": 0.0}
            else:
                return {"f_score": 1.0, "precision": 1.0, "recall": 1.0}
        correct = [h for h in pred if h in label]
        recall = float(len(correct)) / len(label)
        precision = float(len(correct)) / len(pred)
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return {"f_score": f_score, "precision": precision, "recall": recall}

    def summarize(self):
        if not dist.is_main_process():
            return None
        grd_cache = {}
        cor_cache = {}
        rat_cache = {}
        split_cache = {}
        for anno_name, prediction in self.predictions.items():
            data_src = anno_name.split("|")[-1].split("_")[0]
            assert prediction is not None
            assert len(prediction["boxes_target"]) == len(prediction["cor_sent_list"])
            sorted_scores_boxes = sorted(zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True)
            sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
            target_bboxes = prediction["boxes_target"].tolist()
            src_h, src_w = prediction["orig_size"].tolist()
            converted_bboxes = [[target_bbox[0] * src_w - target_bbox[2] * src_w / 2,
                                 target_bbox[1] * src_h - target_bbox[3] * src_h / 2,
                                 target_bbox[0] * src_w + target_bbox[2] * src_w / 2,
                                 target_bbox[1] * src_h + target_bbox[3] * src_h / 2]
                                for target_bbox in target_bboxes]
            gious = box_ops.generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bboxes))

            grd_cache[anno_name] = gious
            cor_cache[anno_name] = {"raw": prediction["raw_sent"], "cor": prediction["correction_sent_pred"],
                                    "tars": prediction["cor_sent_list"], "change_infos": prediction["change_infos"]}
            rat_cache[anno_name] = {"rat": prediction["rationale_sent_pred"], "tars": prediction["rationale_list"]}
            split_cache[anno_name] = data_src
        # name1 = list(grd_cache)[0]
        # grd_cache[name1] = grd_cache[name1].repeat(1, 2)
        # cor_cache[name1]["tars"] = [cor_cache[name1]["tars"][0], "woman with white backpack"]

        # calculate rationale captioning score by using cider score
        # rat_cache["test"] = {"rat": "a girl is sitting on the chair", "tars": ["a girl is sitting on the chair", "the girl is on the grass", "a dog is next to the girl"][:]}
        rat_scorer = Cider(score_mode="max")
        rat_cider_avg, rat_cider_dict = rat_scorer.compute_score(rat_cache, pred_name="rat", targets_name="tars")
        rat_score_dict = rat_cider_dict

        # calculate correction captioning contributions by using cider score
        # cor_cache["test"] = {"raw": "the girl sitting on the chair", "cor": "the baby sitting on the chair", "tars":["the boy sitting on the chair", "the girl standing on the chair"]}
        cor_scorer = Cider(score_mode="all")
        _, raw_cider_dict = cor_scorer.compute_score(cor_cache, pred_name="raw", targets_name="tars")
        _, cor_cider_dict = cor_scorer.compute_score(cor_cache, pred_name="cor", targets_name="tars")
        cor_contribute_dict = self.compute_contribute(cor_cache)
        # cor_cider_contribute_dict = {}
        # for name in raw_cider_dict:
        #     cor_cider_contribute_dict[name] = np.array(cor_cider_dict[name]) - np.array(raw_cider_dict[name])

        # calculate correction captioning contributions and refering grounding by using gious
        cor_score_text_dict = {}
        cor_score_reason_dict = {}
        grd_rcl_dict = {}
        for name in grd_cache:
            cor_scores_for_name = cor_contribute_dict[name]
            idx = np.array([sco["f_score"] for sco in cor_scores_for_name]).argmax()
            cor_score_reason_dict[name] = cor_contribute_dict[name][idx]["f_score"]
            cor_score_text_dict[name] = cor_cider_dict[name][idx]
            grd_score_for_name = grd_cache[name][:, idx]
            grd_rcl_dict[name] = {k: int(max(grd_score_for_name[:k]) >= self.thresh_iou) for k in self.k}

        # calculate all scores
        grd_rcl_dict_refcoco = {}
        grd_rcl_dict_refcocop = {}
        grd_rcl_dict_refcocog = {}
        cor_score_text_dict_refcoco = {}
        cor_score_text_dict_refcocop = {}
        cor_score_text_dict_refcocog = {}
        cor_score_reason_dict_refcoco = {}
        cor_score_reason_dict_refcocop = {}
        cor_score_reason_dict_refcocog = {}
        rat_score_dict_refcoco = {}
        rat_score_dict_refcocop = {}
        rat_score_dict_refcocog = {}

        for name in split_cache:
            if split_cache[name] == "refcoco":
                grd_rcl_dict_refcoco[name] = grd_rcl_dict[name]
                cor_score_text_dict_refcoco[name] = cor_score_text_dict[name]
                cor_score_reason_dict_refcoco[name] = cor_score_reason_dict[name]
                rat_score_dict_refcoco[name] = rat_score_dict[name]
            if split_cache[name] == "refcocop":
                grd_rcl_dict_refcocop[name] = grd_rcl_dict[name]
                cor_score_text_dict_refcocop[name] = cor_score_text_dict[name]
                cor_score_reason_dict_refcocop[name] = cor_score_reason_dict[name]
                rat_score_dict_refcocop[name] = rat_score_dict[name]
            if split_cache[name] == "refcocog":
                grd_rcl_dict_refcocog[name] = grd_rcl_dict[name]
                cor_score_text_dict_refcocog[name] = cor_score_text_dict[name]
                cor_score_reason_dict_refcocog[name] = cor_score_reason_dict[name]
                rat_score_dict_refcocog[name] = rat_score_dict[name]
        results = {
            "refcoco": get_result(grd_rcl_dict_refcoco, cor_score_text_dict_refcoco, cor_score_reason_dict_refcoco,
                                  rat_score_dict_refcoco, ks=self.k),
            "refcocop": get_result(grd_rcl_dict_refcocop, cor_score_text_dict_refcocop, cor_score_reason_dict_refcocop,
                                   rat_score_dict_refcocop, ks=self.k),
            "refcocog": get_result(grd_rcl_dict_refcocog, cor_score_text_dict_refcocog, cor_score_reason_dict_refcocog,
                                   rat_score_dict_refcocog, ks=self.k),
            "liefcoco": get_result(grd_rcl_dict, cor_score_text_dict, cor_score_reason_dict, rat_score_dict, ks=self.k)
        }
        print(str(results))
        return results


def get_result(grd_rcl_dict, cor_score_text_dict, cor_score_reason_dict, rat_score_dict, ks=[1, 5, 10]):
    assert len(grd_rcl_dict) == len(cor_score_text_dict) == len(cor_score_reason_dict) == len(rat_score_dict)
    grd_score = {k: 0.0 for k in ks}
    cnt = 0.0
    for name in list(grd_rcl_dict):
        cnt += 1
        for k in ks:
            grd_score[k] += grd_rcl_dict[name][k]
    for k in ks:
        grd_score[k] /= cnt
    cor_score_text = float(np.mean(list(cor_score_text_dict.values())))
    cor_score_reason = float(np.mean(list(cor_score_reason_dict.values())))
    rat_score = float(np.mean(list(rat_score_dict.values())))
    results = {
        "grounding_score": grd_score,
        "correction_cider_score": cor_score_text,
        "correction_hit_score": cor_score_reason,
        "rationale_cider_score": rat_score
    }
    return results
