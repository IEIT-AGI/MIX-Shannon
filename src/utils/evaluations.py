import torch
import torch.nn.functional as F
import copy
from torch import nn
from difflib import Differ

# from utils import box_ops, dist
from src.utils import box_ops, dist


class PostProcess(nn.Module):
    def __init__(self, tokenizer=None):
        super(PostProcess, self).__init__()
        self.tokenizer = tokenizer
        self.differ = Differ()

    @torch.no_grad()
    def forward(self, outputs, target_sizes, infos=None):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        labels = torch.ones_like(labels)
        scores = 1 - prob[:, :, -1]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        assert len(scores) == len(labels) == len(boxes)
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        if "pred_isfinal" in outputs:
            is_final = outputs["pred_isfinal"].sigmoid()
            scores_refexp = scores * is_final.view_as(scores)
            assert len(results) == len(scores_refexp)
            for i in range(len(results)):
                results[i]["scores_refexp"] = scores_refexp[i]

        if "caption_correction" in outputs and "caption_rationale" in outputs:
            correction_sents_pred = []
            for token_ids_out in outputs["caption_correction"]:
                token_ids_out = token_ids_out.tolist()
                token_ids_out = [token_id_out for token_id_out in token_ids_out if
                                 token_id_out != self.tokenizer.bos_token_id]
                correction_sent_pred = self.tokenizer.decode(token_ids_out).split(self.tokenizer.eos_token)[0]
                correction_sents_pred.append(correction_sent_pred)
            rationale_sents_pred = []
            for token_ids_out in outputs["caption_rationale"]:
                token_ids_out = token_ids_out.tolist()
                token_ids_out = [token_id_out for token_id_out in token_ids_out if
                                 token_id_out != self.tokenizer.bos_token_id]
                rationale_sent_pred = self.tokenizer.decode(token_ids_out).split(self.tokenizer.eos_token)[0]
                rationale_sents_pred.append(rationale_sent_pred)

            for k, (correction_sent_pred, rationale_sent_pred) in enumerate(
                    zip(correction_sents_pred, rationale_sents_pred)):
                results[k]["correction_sent_pred"] = correction_sent_pred
                results[k]["rationale_sent_pred"] = rationale_sent_pred

            if infos is not None:
                res = {}
                for info, result in zip(infos, results):
                    name = info["name"]
                    res[name] = result
                    res[name]["name"] = name
                    res[name]["boxes_target"] = info["boxes"]
                    res[name]["orig_size"] = info["orig_size"]
                    raw_sent = info["raw_sent"]
                    cor_sent = result["correction_sent_pred"]
                    tar_sents = info["cor_sent_list"]
                    change_pred = self.get_change(raw_sent, cor_sent)
                    res[name]["change_infos"] = [{"label": self.get_change(raw_sent, tar_sent), "pred": change_pred} for
                                                 tar_sent in tar_sents]
                    res[name]["raw_sent"] = raw_sent
                    res[name]["cor_sent_list"] = tar_sents
                    res[name]["rationale_list"] = info["rationale_list"]
                return res
        return results

    def get_change(self, src_sent, target_sent):
        diffs = list(self.differ.compare(src_sent.split(), target_sent.split()))
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


class RefExpEvaluator(object):
    def __init__(self, k=(1, 5, 10), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        self.k = k
        self.thresh_iou = thresh_iou
        self.labels = {}
        self.predictions = {}

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if not dist.is_main_process():
            return None
        dataset2score = {"refcoco": {k: 0.0 for k in self.k}, "refcocop": {k: 0.0 for k in self.k},
                         "refcocog": {k: 0.0 for k in self.k}}
        dataset2count = {"refcoco": 0.0, "refcocop": 0.0, "refcocog": 0.0}
        score_dict = {k: 0.0 for k in self.k}
        count = 0.0

        info_cnt = {}
        for anno_name, prediction in self.predictions.items():
            if "raw" not in prediction:
                data_src = anno_name.split("|")[-1].split("_")[0]
                assert prediction is not None
                sorted_scores_boxes = sorted(zip(prediction["scores"].tolist(), prediction["boxes"].tolist()),
                                             reverse=True)
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                target_bbox = prediction["boxes_target"][0].tolist()
                src_h, src_w = prediction["orig_size"].tolist()
                converted_bbox = [
                    target_bbox[0] * src_w - target_bbox[2] * src_w / 2,
                    target_bbox[1] * src_h - target_bbox[3] * src_h / 2,
                    target_bbox[0] * src_w + target_bbox[2] * src_w / 2,
                    target_bbox[1] * src_h + target_bbox[3] * src_h / 2,
                ]
                giou = box_ops.generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
                for k in self.k:
                    if max(giou[:k]) >= self.thresh_iou:
                        score_dict[k] += 1.0
                        dataset2score[data_src][k] += 1.0
                count += 1.0
                dataset2count[data_src] += 1.0
            else:
                assert "raw" in prediction and "pred" in prediction and "target" in prediction
                data_src = anno_name.split("|")[-1].split("_")[0]
                prediction_raw = prediction["raw"]
                prediction_pred = prediction["pred"]
                prediction_target = prediction["target"]

                sorted_scores_boxes_raw = sorted(
                    zip(prediction_raw["scores"].tolist(), prediction_raw["boxes"].tolist()), reverse=True)
                sorted_scores_raw, sorted_boxes_raw = zip(*sorted_scores_boxes_raw)
                sorted_boxes_raw = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes_raw])
                target_bbox_raw = prediction["boxes_target"][0].tolist()
                src_h_raw, src_w_raw = prediction["orig_size"].tolist()
                converted_bbox_raw = [
                    target_bbox_raw[0] * src_w_raw - target_bbox_raw[2] * src_w_raw / 2,
                    target_bbox_raw[1] * src_h_raw - target_bbox_raw[3] * src_h_raw / 2,
                    target_bbox_raw[0] * src_w_raw + target_bbox_raw[2] * src_w_raw / 2,
                    target_bbox_raw[1] * src_h_raw + target_bbox_raw[3] * src_h_raw / 2,
                ]
                giou_raw = box_ops.generalized_box_iou(sorted_boxes_raw,
                                                       torch.as_tensor(converted_bbox_raw).view(-1, 4))

                sorted_scores_boxes_pred = sorted(
                    zip(prediction_pred["scores"].tolist(), prediction_pred["boxes"].tolist()), reverse=True)
                sorted_scores_pred, sorted_boxes_pred = zip(*sorted_scores_boxes_pred)
                sorted_boxes_pred = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes_pred])
                target_bbox_pred = prediction["boxes_target"][0].tolist()
                src_h_pred, src_w_pred = prediction["orig_size"].tolist()
                converted_bbox_pred = [
                    target_bbox_pred[0] * src_w_pred - target_bbox_pred[2] * src_w_pred / 2,
                    target_bbox_pred[1] * src_h_pred - target_bbox_pred[3] * src_h_pred / 2,
                    target_bbox_pred[0] * src_w_pred + target_bbox_pred[2] * src_w_pred / 2,
                    target_bbox_pred[1] * src_h_pred + target_bbox_pred[3] * src_h_pred / 2,
                ]
                giou_pred = box_ops.generalized_box_iou(sorted_boxes_pred,
                                                        torch.as_tensor(converted_bbox_pred).view(-1, 4))

                sorted_scores_boxes_target = sorted(
                    zip(prediction_target["scores"].tolist(), prediction_target["boxes"].tolist()), reverse=True)
                sorted_scores_target, sorted_boxes_target = zip(*sorted_scores_boxes_target)
                sorted_boxes_target = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes_target])
                target_bbox_target = prediction["boxes_target"][0].tolist()
                src_h_target, src_w_target = prediction["orig_size"].tolist()
                converted_bbox_target = [
                    target_bbox_target[0] * src_w_target - target_bbox_target[2] * src_w_target / 2,
                    target_bbox_target[1] * src_h_target - target_bbox_target[3] * src_h_target / 2,
                    target_bbox_target[0] * src_w_target + target_bbox_target[2] * src_w_target / 2,
                    target_bbox_target[1] * src_h_target + target_bbox_target[3] * src_h_target / 2,
                ]
                giou_target = box_ops.generalized_box_iou(sorted_boxes_target,
                                                          torch.as_tensor(converted_bbox_target).view(-1, 4))

                for k in self.k:
                    if max(giou_pred[:k]) >= self.thresh_iou:
                        score_dict[k] += 1.0
                        dataset2score[data_src][k] += 1.0
                count += 1.0
                dataset2count[data_src] += 1.0
                info_cnt[anno_name] = {
                    "raw_sent": prediction["raw_sent"],
                    "pred_sent": prediction["pred_sent"],
                    "cor_sent": prediction["cor_sent"],
                    "raw_rcl1": max(giou_raw[:1]) >= self.thresh_iou,
                    "pred_rcl1": max(giou_pred[:1]) >= self.thresh_iou,
                    "target_rcl1": max(giou_target[:1]) >= self.thresh_iou,
                }

        for k in self.k:
            score_dict[k] /= count
        for key, value in dataset2score.items():
            for k in self.k:
                value[k] /= dataset2count[key]

        results = {}
        for key, value in dataset2score.items():
            results[key] = sorted([v for k, v in value.items()])
            print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

        results_all = sorted(list(score_dict.values()))
        print(f" Precision @ 1, 5, 10: {results_all} \n")

        results.update({"refexp": results_all})
        return results
