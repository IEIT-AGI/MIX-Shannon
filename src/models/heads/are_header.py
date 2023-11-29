from torch import nn
from omegaconf import DictConfig
import torch
from typing import Dict, Callable, List, Optional
from src.models.builder import HEADS, build_head, build_matcher, build_loss


@HEADS.register_module()
class EventHeader(nn.Module):

    def __init__(self, tau: float = 0.01, loss_weight: float = 1):
        super(EventHeader, self).__init__()
        self.tau = tau  # temperature parameter
        self.loss_weight = loss_weight

    def forward(self, event_feat: Dict, target: Optional[List] = None) -> Dict:
        if self.training:
            return self.forward_train(event_feat=event_feat, target=target)
        else:
            return self.forward_test(event_feat=event_feat)

    def forward_train(self, event_feat: Dict, target: List) -> Dict:
        predict_rst: Dict = self.forward_test(event_feat=event_feat)
        tgt = torch.stack(target, dim=0)
        rst = -predict_rst["predict"].log_softmax(-1)
        loss = (rst * tgt / tgt.sum(1, keepdim=True)).sum(dim=1).mean()
        loss *= self.loss_weight

        # output
        output = predict_rst
        output.update({"loss": loss})
        return output

    def forward_test(self, event_feat: Dict) -> Dict:

        event_fusion_token_embedding = event_feat["feature"]
        event_embedding = event_feat["embedding"]

        predicts = event_fusion_token_embedding.mm(event_embedding.t()) / self.tau

        return {"predict": predicts}


@HEADS.register_module()
class AnswerHeader(nn.Module):

    def __init__(self, tau: float = 0.01, loss_weight: float = 1):
        super(AnswerHeader, self).__init__()
        self.tau = tau  # temperature parameter
        self.loss_weight = loss_weight

    def forward(self, answer_feat: Dict, target: Optional[List] = None) -> Dict:
        if self.training:
            return self.forward_train(answer_feat, target)
        else:
            return self.forward_test(answer_feat)

    def forward_test(self, answer_feat: Dict) -> Dict:  # todo VS EventHeader
        answer_fusion_token_embedding = answer_feat["feature"]
        answer_embedding = answer_feat["embedding"]
        predicts = answer_fusion_token_embedding.mm(answer_embedding.t()) / self.tau

        return {"predict": predicts}

    def forward_train(self, answer_feat: Dict, target: List) -> Dict:
        predict_rst: Dict = self.forward_test(answer_feat=answer_feat)
        tgt = torch.stack(target, dim=0)

        rst = -predict_rst["predict"].log_softmax(-1)
        loss = (rst * tgt / tgt.sum(1, keepdim=True)).sum(dim=1).mean()
        loss *= self.loss_weight

        # output
        output = predict_rst
        output.update({"loss": loss})
        return output


@HEADS.register_module()
class RelationHeader1(nn.Module):

    def __init__(self, tau: float = 0.01, loss_weight: float = 1):
        super(RelationHeader1, self).__init__()
        self.tau = tau  # temperature parameter
        self.loss_weight = loss_weight

    def forward(self, event_feat: Dict, target: Optional[List] = None) -> Dict:
        if self.training:
            return self.forward_train(event_feat=event_feat, target=target)
        else:
            return self.forward_test(event_feat=event_feat)

    def forward_train(self, event_feat: Dict, target: List) -> Dict:
        predict_rst: Dict = self.forward_test(event_feat=event_feat)
        tgt = torch.stack(target, dim=0)
        rst = -predict_rst["predict"].log_softmax(-1)
        loss = (rst * tgt / tgt.sum(1, keepdim=True)).sum(dim=1).mean()
        loss *= self.loss_weight

        # output
        output = predict_rst
        output.update({"loss": loss})
        return output

    def forward_test(self, event_feat: Dict) -> Dict:

        event_fusion_token_embedding = event_feat["feature"]
        event_embedding = event_feat["embedding"]

        predicts = event_fusion_token_embedding.mm(event_embedding.t()) / self.tau

        return {"predict": predicts}


@HEADS.register_module()
class ReasoningHeader1(nn.Module):

    def __init__(self, event_header: DictConfig, answer_header: DictConfig, relation_header: DictConfig):
        super(ReasoningHeader1, self).__init__()
        self.event_header: Callable = build_head(event_header)
        self.answer_header: Callable = build_head(answer_header)
        self.relation_header: Callable = build_head(relation_header)
        self.transE_loss: Callable = nn.L1Loss()  # reduction='sum'

    def forward(self, batch: Dict, event_info: Dict, answer_info: Dict, relation_info: Dict):
        event_rst: Dict = self.event_header(event_info, batch["fact_label"])
        answer_rst: Dict = self.answer_header(answer_info, batch["answer_label"])
        relation_rst: Dict = self.relation_header(relation_info, batch["relation_label"])

        # output: modify the keys of event and answer
        output = {f"{k}_event": v for k, v in event_rst.items()}
        output.update({f"{k}_answer": v for k, v in answer_rst.items()})
        output.update({f"{k}_relation": v for k, v in relation_rst.items()})
        return output

    def calculate_transE_loss(self, event, relation, answer):
        return self.transE_loss(event + relation, answer)


@HEADS.register_module()
class ObjectDetectionHeader1(nn.Module):

    def __init__(self, matcher: DictConfig, label_loss: DictConfig, bbox_loss: DictConfig):
        super(ObjectDetectionHeader1, self).__init__()
        self.matcher: Callable = build_matcher(matcher)
        self.label_header: Callable = build_loss(label_loss)
        self.bbox_header: Callable = build_loss(bbox_loss)

    def forward_train(self, predict: Dict, target: Dict) -> Dict:
        indices: List = self.matcher(predict, target)
        label_rst = self.label_header(predict["pred_logics"], target["boxes"], indices, target["positive_map"])
        bbox_rst = self.bbox_header(predict["pred_boxes"], target["boxes"], indices)

        # output
        output = label_rst
        output.update(bbox_rst)
        for k, v in output.items():
            if k.startswith("loss"):
                output[k] = v / len(target["boxes"])

        output.update({"match_indices": indices})
        return output

    def forward_test(self, predict: Dict, target: Dict) -> Dict:
        indices: List = self.matcher(predict, target)
        return {"match_indices": indices}

    def forward(self, predict: Dict, target: Dict) -> Dict:
        if self.training:
            return self.forward_train(predict, target)
        else:
            return self.forward_test(predict, target)


@HEADS.register_module()
class ReasoningHeader(nn.Module):

    def __init__(self, event_header: DictConfig, answer_header: DictConfig):
        super(ReasoningHeader, self).__init__()
        self.event_header: Callable = build_head(event_header)
        self.answer_header: Callable = build_head(answer_header)

    def forward(self, batch: Dict, event_info: Dict, answer_info: Dict):
        event_rst: Dict = self.event_header(event_info, batch["fact_label"])
        answer_rst: Dict = self.answer_header(answer_info, batch["answer_label"])

        # output: modify the keys of event and answer
        output = {f"{k}_event": v for k, v in event_rst.items()}
        output.update({f"{k}_answer": v for k, v in answer_rst.items()})

        return output


@HEADS.register_module()
class ObjectDetectionHeader(nn.Module):

    def __init__(self, matcher: DictConfig, label_loss: DictConfig, bbox_loss: DictConfig):
        super(ObjectDetectionHeader, self).__init__()
        self.matcher: Callable = build_matcher(matcher)
        self.label_header: Callable = build_loss(label_loss)
        self.bbox_header: Callable = build_loss(bbox_loss)

    def forward(self, predict: Dict, target: Dict) -> Dict:
        indices: List = self.matcher(predict, target)
        label_rst = self.label_header(predict["pred_logics"], target["boxes"], indices, target["positive_map"])
        bbox_rst = self.bbox_header(predict["pred_boxes"], target["boxes"], indices)

        # output
        output = label_rst
        output.update(bbox_rst)
        for k, v in output.items():
            if k.startswith("loss"):
                output[k] = v / len(target["boxes"])

        return output