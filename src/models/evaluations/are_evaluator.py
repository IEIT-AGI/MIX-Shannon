from typing import Dict, List, Callable
from omegaconf import DictConfig
import torch
from torch.nn import functional as F
from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src.models.builder import ANSWER_MASK, build_answer_mask


class Retrieval:

    def __init__(self, cfg: DictConfig):
        self.rank = cfg.get("rank", (10, 30, 50))
        self.rank = eval(self.rank) if isinstance(self.rank, str) else self.rank
        self.rank = sorted(self.rank)

    def __call__(self, predicts: torch.Tensor, target: torch.Tensor) -> Dict:
        if len(target.shape) == 3:
            target = target[0]
        _, ok = predicts.topk(50, dim=1)
        agreeing_all = torch.zeros([predicts.shape[0], 1], dtype=torch.float, device=predicts.device)
        output = {f"HIT@{k}": 0 for k in self.rank}
        for i in range(50):
            tmp = ok[:, i].reshape(-1, 1)
            agreeing_all += target.gather(dim=1, index=tmp)
            if i + 1 in self.rank:
                output[f"HIT@{i + 1}"] = (agreeing_all * 0.3).clamp(max=1)

        output = {k: 100.0 * v.sum().item() / len(predicts) for k, v in output.items()}
        return output


@ANSWER_MASK.register_module()
class SoftMask:

    def __init__(self, value: float = 0.25):
        self.soft_score = value

    def __call__(self, mask_answer):
        return -1 * self.soft_score * mask_answer


@ANSWER_MASK.register_module()
class HardMask:

    def __init__(self, value: float = 100.0):
        self.hard_score = value

    def __call__(self, mask_answer):
        return -1 * self.hard_score * mask_answer


class ARE_QA:

    def __init__(self, cfg: DictConfig):
        self.event_to_answer_file = cfg.event_to_answer
        self.event_to_answer = self.load_json_file(cfg.event_to_answer)
        self.qa_eval = Retrieval(cfg.evaluator)
        self.max_rank = max(self.qa_eval.rank)
        self.answer_mask: Callable = build_answer_mask(cfg.answer_mask)
        self.wHIT_metric = self.load_weighted_hit_metric(cfg.weighted_hit_score)

    def load_weighted_hit_metric(self, weighted_hit_file: str) -> Dict:
        data: Dict = self.load_json_file(weighted_hit_file)
        label_2_idx = {label: idx for idx, label in enumerate(data.keys())}
        for _, value in data.items():
            label_idx = [label_2_idx.get(idx) for idx in value[0]]
            value[1] = [float(v) for v in value[1]]
            value.append(label_idx)
        return data

    @staticmethod
    def load_json_file(file: str) -> Dict:
        import json
        with open(file) as f:
            return json.load(f)

    def mask_on_answer(self, predict_event: torch.Tensor, predict_answer: torch.Tensor) -> torch.Tensor:  # todo
        mask_answer = torch.ones(predict_event.shape[0], predict_answer.shape[1], device=predict_event.device)
        _, top_event_idx = predict_event.topk(self.max_rank, dim=1)
        for i in range(len(predict_event)):
            for j in range(self.max_rank):
                key = str(top_event_idx[i][j].item())
                if key in self.event_to_answer.keys():
                    for k in self.event_to_answer[key]:
                        mask_answer[i][k] = 0

        return self.answer_mask(mask_answer)

    def __call__(self, predict_event: torch.Tensor, predict_answer: torch.Tensor, answers: torch.Tensor,
                 answers_label: List) -> Dict:

        wHIT: Dict = self._calculate_weighted_hit_metric(predict_answer, answers, answers_label)

        mask_answer = self.mask_on_answer(predict_event, predict_answer)
        predict_answer += mask_answer
        qa_eval_rst: Dict = self.qa_eval(predict_answer, answers)

        # output
        output = wHIT
        output.update(qa_eval_rst)
        return output

    def _wHIT_indicator(self,
                        answer: torch.Tensor,
                        answer_label: str,
                        indices_1: torch.Tensor,
                        indices_other: torch.Tensor,
                        top_k: int = 10) -> float:
        idx_intersection: Callable = lambda src_idx, dst_idx: set(src_idx).intersection(set(dst_idx))
        is_exist: Callable = lambda src_idx, dst_idx: len(idx_intersection(src_idx, dst_idx)) != 0

        answer_label_idx: List = self.wHIT_metric.get(answer_label)[2][:top_k]
        answer_label_similarity_score: List = self.wHIT_metric.get(answer_label)[1][:top_k]
        _indices_other = indices_other.cpu().tolist()

        if indices_1 == answer.argmax():
            return 1
        elif is_exist(_indices_other, answer_label_idx):
            similar_set = idx_intersection(_indices_other, answer_label_idx)
            similar_idx = [answer_label_idx.index(_idx) for _idx in similar_set]
            similar_answer_score = [answer_label_similarity_score[_idx] for _idx in similar_idx]
            return sum(similar_answer_score) / len(similar_idx)
        else:
            return 0

    def _calculate_weighted_hit_metric(self, predict: torch.Tensor, answers: torch.Tensor, answers_label: List):
        _, indices_1 = predict.topk(1, dim=1)
        _, indices_5 = predict.topk(5, dim=1)
        _, indices_10 = predict.topk(10, dim=1)

        hit_score5 = 0
        hit_score10 = 0
        for idx, answer, label in zip(range(len(answers_label)), answers, answers_label):
            hit_score5 += self._wHIT_indicator(answer, label, indices_1[idx], indices_5[idx], 5)
            hit_score10 += self._wHIT_indicator(answer, label, indices_1[idx], indices_10[idx], 10)

        return {"wHIT@5": hit_score5 * 100 / len(answers), "wHIT@10": hit_score10 * 100 / len(answers)}


class ARE_QA1:

    def __init__(self, cfg: DictConfig):
        self.event_to_answer_file = cfg.event_to_answer
        self.event_to_answer = self.load_json_file(cfg.event_to_answer)

        self.relation_to_answer_file = cfg.relation_to_answer
        self.relation_to_answer = self.load_json_file(cfg.relation_to_answer)

        self.qa_eval = Retrieval(cfg.evaluator)
        self.max_rank = max(self.qa_eval.rank)
        self.relation_top_k = cfg.get("relation_top_k", 3)
        # self.soft_score = cfg.get("soft_score", 0.25)
        self.answer_mask: Callable = build_answer_mask(cfg.answer_mask)
        self.wHIT_metric = self.load_weighted_hit_metric(cfg.weighted_hit_score)

    def load_weighted_hit_metric(self, weighted_hit_file: str) -> Dict:
        data: Dict = self.load_json_file(weighted_hit_file)
        label_2_idx = {label: idx for idx, label in enumerate(data.keys())}
        for key, value in data.items():
            label_idx = [label_2_idx.get(idx) for idx in value[0]]
            value[1] = [float(v) for v in value[1]]
            value.append(label_idx)
        return data

    @staticmethod
    def load_json_file(file: str) -> Dict:
        import json
        with open(file) as f:
            return json.load(f)

    def mask_on_answer(self, predict_event: torch.Tensor, predict_answer: torch.Tensor) -> torch.Tensor:  # todo
        mask_answer = torch.ones(predict_event.shape[0], predict_answer.shape[1], device=predict_event.device)
        _, top_event_idx = predict_event.topk(self.max_rank, dim=1)
        for i in range(len(predict_event)):
            for j in range(self.max_rank):
                key = str(top_event_idx[i][j].item())
                if key in self.event_to_answer.keys():
                    for k in self.event_to_answer[key]:
                        mask_answer[i][k] = 0

        # mask_answer = -1 * self.soft_score * mask_answer
        return self.answer_mask(mask_answer)

    def mask_on_answer_for_relation(self, predict_relation: torch.Tensor, predict_answer: torch.Tensor) -> torch.Tensor:
        mask_answer = torch.ones(predict_relation.shape[0], predict_answer.shape[1], device=predict_relation.device)
        _, top_event_idx = predict_relation.topk(self.relation_top_k, dim=1)
        for i in range(len(predict_relation)):
            for j in range(self.relation_top_k):
                key = str(top_event_idx[i][j].item())
                if key in self.relation_to_answer.keys():
                    for k in self.relation_to_answer[key]:
                        mask_answer[i][k] = 0

        # mask_answer = -1 * self.soft_score * mask_answer
        return mask_answer

    def mask_on_answer_for_event(self, predict_event: torch.Tensor,
                                 predict_answer: torch.Tensor) -> torch.Tensor:  # todo
        mask_answer = torch.ones(predict_event.shape[0], predict_answer.shape[1], device=predict_event.device)
        _, top_event_idx = predict_event.topk(self.max_rank, dim=1)
        for i in range(len(predict_event)):
            for j in range(self.max_rank):
                key = str(top_event_idx[i][j].item())
                if key in self.event_to_answer.keys():
                    for k in self.event_to_answer[key]:
                        mask_answer[i][k] = 0

        # mask_answer = -1 * self.soft_score * mask_answer
        return mask_answer

    def __call__(self, predict_event: torch.Tensor, predict_answer: torch.Tensor, predict_relation: torch.Tensor,
                 answers: torch.Tensor, answers_label: List) -> Dict:

        # wHIT: Dict = self._calculate_weighted_hit_metric(predict_answer, answers, answers_label)  # wHIT

        if predict_relation is None:  # only event
            mask_answer = self.mask_on_answer(predict_event, predict_answer)
        else:  # evnet + realation

            def inter_op(a, b):
                # a: 1 0 0 1
                # b: 1 0 1 0
                # result: 1 0 1 1

                # simple method
                # c = a+b
                # g = c.to(bool).to(float)

                c = ~(a.to(bool) ^ b.to(bool))
                d = a.to(bool) & b.to(bool)
                e = c.to(int) - d.to(int)
                f = ~e.to(bool)
                g = f.to(float)
                return g

            mask_answer_event = self.mask_on_answer_for_event(predict_event, predict_answer)
            mask_answer_relation = self.mask_on_answer_for_relation(predict_relation, predict_answer)
            mask_answer = inter_op(mask_answer_event, mask_answer_relation)

            mask_answer = self.answer_mask(mask_answer)

        predict_answer += mask_answer

        wHIT: Dict = self._calculate_weighted_hit_metric(predict_answer, answers, answers_label)  # todo
        qa_eval_rst: Dict = self.qa_eval(predict_answer, answers)

        # output
        output = wHIT
        output.update(qa_eval_rst)
        return output

    def _wHIT_indicator(self,
                        answer: torch.Tensor,
                        answer_label: str,
                        indices_1: torch.Tensor,
                        indices_other: torch.Tensor,
                        top_k: int = 10):
        idx_intersection: Callable = lambda src_idx, dst_idx: set(src_idx).intersection(set(dst_idx))
        is_exist: Callable = lambda src_idx, dst_idx: len(idx_intersection(src_idx, dst_idx)) != 0

        answer_label_idx: List = self.wHIT_metric.get(answer_label)[2][:top_k]
        answer_label_similarity_score: List = self.wHIT_metric.get(answer_label)[1][:top_k]
        _indices_other = indices_other.cpu().tolist()

        if indices_1 == answer.argmax():
            return 1
        elif is_exist(_indices_other, answer_label_idx):
            similar_set = idx_intersection(_indices_other, answer_label_idx)
            similar_idx = [answer_label_idx.index(_idx) for _idx in similar_set]
            similar_answer_score = [answer_label_similarity_score[_idx] for _idx in similar_idx]
            return sum(similar_answer_score) / len(similar_idx)
        else:
            return 0

    def _calculate_weighted_hit_metric(self, predict: torch.Tensor, answers: torch.Tensor, answers_label: List):
        _, indices_1 = predict.topk(1, dim=1)
        _, indices_5 = predict.topk(5, dim=1)
        _, indices_10 = predict.topk(10, dim=1)

        hit_score5 = 0
        hit_score10 = 0
        for idx, answer, label in zip(range(len(answers_label)), answers, answers_label):
            hit_score5 += self._wHIT_indicator(answer, label, indices_1[idx], indices_5[idx], 5)
            hit_score10 += self._wHIT_indicator(answer, label, indices_1[idx], indices_10[idx], 10)

        return {"wHIT@5": hit_score5 * 100 / len(answers), "wHIT@10": hit_score10 * 100 / len(answers)}


class AREGrounding:

    def __init__(self, cfg: DictConfig):
        self.rank = cfg.get("rank", (1, 5, 10))
        self.rank = eval(self.rank) if isinstance(self.rank, str) else self.rank
        self.rank = sorted(self.rank)
        self.iou_thresh = cfg.get("iou_thresh", 0.5)

    @staticmethod
    def convert(predict: Dict, target_size: torch.Tensor) -> List:
        pred_logic, pred_bbox = predict["pred_logics"], predict["pred_boxes"]
        prob = F.softmax(pred_logic, -1)
        scores, labels = prob[..., :-1].max(-1)
        labels = torch.ones_like(labels)
        scores = 1 - prob[:, :, -1]

        boxes = box_cxcywh_to_xyxy(pred_bbox)
        img_h, img_w = target_size.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        assert len(scores) == len(labels) == len(boxes)

        # output
        return [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

    def __call__(self, predict: Dict, target: Dict) -> Dict:
        predict = self.convert(predict, target["orig_size"])
        dataset2score = {"ai-vqa": {k: 0.0 for k in self.rank}}
        dataset2count = {"ai-vqa": 0.0}
        score_dict = {k: 0.0 for k in self.rank}
        count = 0.0
        data_src = "ai-vqa"
        grd_cache = {}
        split_cache = {}
        for pred, img_id, orig_size, gt_box in zip(predict, target["image_id"], target["orig_size"], target["boxes"]):
            sorted_scores_boxes = sorted(zip(pred["scores"].tolist(), pred["boxes"].tolist()), reverse=True)
            sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
            sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])

            target_bbox = [gt_box.tolist()]  # todo what?
            src_h, src_w = orig_size.tolist()
            convert_gt_bbox = [[
                bbox[0] * src_w - bbox[2] * src_w / 2, bbox[1] * src_h - bbox[3] * src_h / 2,
                bbox[0] * src_w + bbox[2] * src_w / 2, bbox[1] * src_h + bbox[3] * src_h / 2
            ] for bbox in target_bbox]
            gious = generalized_box_iou(sorted_boxes, torch.as_tensor(convert_gt_bbox))

            for k in self.rank:
                if max(gious[:k]) >= self.iou_thresh:
                    score_dict[k] += 1.0
                    dataset2score[data_src][k] += 1.0
            count += 1.0
            dataset2count[data_src] += 1.0

            grd_cache[img_id] = gious
            split_cache[img_id] = data_src

        name1 = list(grd_cache)[0]
        grd_cache[name1] = grd_cache[name1].repeat(1, 2)

        for k in self.rank:
            score_dict[k] /= count
        for key, value in dataset2score.items():
            for k in self.rank:
                value[k] /= dataset2count[key]

        results = {}
        for key, value in dataset2score.items():
            results[key] = sorted([v for k, v in value.items()])
            # print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

        results_all = sorted(list(score_dict.values()))

        # print(f" Precision @ 1, 5, 10: {results_all} \n")

        results.update({"grounding_result:": results_all})

        # return results
        output = {}
        data_precision = {f"{data_src}-precision-Recall@{rank}": p for rank, p in zip([1, 5, 10], results[data_src])}
        output.update(data_precision)
        grounding_result = {f"{data_src}-grounding_result-Recall@{rank}": p for rank, p in zip([1, 5, 10], results_all)}
        output.update(grounding_result)

        return output


class AREEvaluator:

    def __init__(self, cfg: DictConfig):
        self.event_eval = Retrieval(cfg.retrieval_support_event)
        self.are_qa_eval = ARE_QA(cfg.answer_question)
        self.grd_key_object_eval = AREGrounding(cfg.grounding_key_object)

    def __call__(self, predict_event: torch.Tensor, predict_answer: torch.Tensor, predict_grd: Dict,
                 target: Dict) -> Dict:
        from src.datamodules import AIVQA
        event_eval_rst = self.event_eval(predict_event, target["fact_label"])
        qa_eval_rst = self.are_qa_eval(predict_event, predict_answer, target["answer_label"], target['answer'])

        target_grd = {k: v for k, v in target.items() if k in AIVQA.grounding_field}
        grd_object_eval_rst = self.grd_key_object_eval(predict_grd, target_grd)

        output = {
            "retrieval-support-event": event_eval_rst,
            "answer-question:": qa_eval_rst,
            "ground-key-object": grd_object_eval_rst
        }
        return output


class AREEvaluator1:

    def __init__(self, cfg: DictConfig):
        self.event_eval = Retrieval(cfg.retrieval_support_event)
        self.are_qa_eval = ARE_QA1(cfg.answer_question)
        self.grd_key_object_eval = AREGrounding(cfg.grounding_key_object)

    def __call__(self,
                 predict_event: torch.Tensor,
                 predict_answer: torch.Tensor,
                 predict_grd: Dict,
                 target: Dict,
                 predict_relation: torch.Tensor = None) -> Dict:
        from src.datamodules import AIVQA
        event_eval_rst = self.event_eval(predict_event, target["fact_label"])
        qa_eval_rst = self.are_qa_eval(predict_event, predict_answer, predict_relation, target["answer_label"],
                                       target['answer'])

        target_grd = {k: v for k, v in target.items() if k in AIVQA.grounding_field}
        grd_object_eval_rst = self.grd_key_object_eval(predict_grd, target_grd)

        output = {
            "retrieval-support-event": event_eval_rst,
            "answer-question:": qa_eval_rst,
            "ground-key-object": grd_object_eval_rst
        }
        return output