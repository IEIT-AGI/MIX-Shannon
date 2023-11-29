# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

# from src.models.builder import MATCHES

# from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from src.utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy


# @MATCHES.register_module()
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Softmax(-1)
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, positive_map=None):
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        out_prob = self.norm(outputs["pred_logits"].flatten(0, 1))  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # tgt_bbox = torch.cat([v["boxes"][:1] for v in targets])
        # TODO not condider more than one boxes
        # print(len(tgt_bbox), len(positive_map))
        assert len(tgt_bbox) == len(positive_map)

        # Compute the soft-cross entropy between the predicted token alignment and the GT one for each box
        cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        assert cost_class.shape == cost_bbox.shape

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # sizes = [len(v["boxes"][:1]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    if args.set_loss == "hungarian":
        matcher = HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
        )
    else:
        raise ValueError(f"Only hungarian accepted, got {args.set_loss}")
    return matcher
