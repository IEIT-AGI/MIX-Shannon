import torch
from torch import nn
from src.models.builder import MATCHES
from typing import Dict, List
from scipy.optimize import linear_sum_assignment
from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


@MATCHES.register_module()
class HungarianMatcher(nn.Module):  # todo
    def __init__(self,
                 cost_class: float = 1.0,
                 cost_bbox: float = 1.0,
                 cost_giou: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Softmax(-1)
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self,
                object_detect_rst: Dict,
                targets: Dict) -> List:
        pred_boxes = object_detect_rst["pred_boxes"]
        pred_logics = object_detect_rst["pred_logics"]

        batch_size, query_nums = pred_boxes.shape[:2]

        out_prob = self.norm(pred_logics.flatten(0, 1))
        out_bbox = pred_boxes.flatten(0, 1)

        tgt_bbox = torch.cat(targets["boxes"])
        positive_map = targets.get("positive_map", None)

        assert len(tgt_bbox) == len(positive_map)

        # Compute the soft-cross entropy between the predicted token alignment and the GT one for each box
        cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        assert cost_class.shape == cost_bbox.shape

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(batch_size, query_nums, -1).cpu()  # todo

        bbox_sizes = [len(bbox) for bbox in targets["boxes"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(bbox_sizes, -1))]
        # return [(int(i), int(j)) for i, j in indices] #todo
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
