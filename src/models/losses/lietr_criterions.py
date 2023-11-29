import torch
from torch import nn

# import torch.nn.functional as F
# from utils.misc import NestedTensor, interpolate
# from utils import box_ops, dist
from src.utils import dist
from .mdetr_criterions import SetCriterion as MDETRSetCriterion


class SetCriterion(MDETRSetCriterion):
    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        super().__init__(num_classes, matcher, eos_coef, losses, temperature)
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=1)
        self.mse = nn.MSELoss()
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets, positive_map=None):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets, positive_map)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.as_tensor([num_boxes],dtype=positive_map.dtype,device=positive_map.device)
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, positive_map)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "isfinal": self.loss_isfinal,
            "contrastive_align": self.loss_contrastive_align,
            "caption_ce": self.loss_caption_ce,
            "caption_mse": self.loss_caption_mse
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"  # TODO how funny!
        if loss == "caption_ce":
            if "caption_cls_info" not in outputs:
                return {}
            return loss_map[loss](outputs["caption_cls_info"])
        elif loss == "caption_mse":
            if "caption_dist_info" not in outputs:
                return {}
            return loss_map[loss](outputs["caption_dist_info"])
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def loss_caption_ce(self, caption_cls_infos):
        loss = None
        for caption_cls_info in caption_cls_infos:
            prob = None
            target = None
            focus = None
            for k, v in caption_cls_info.items():
                if "prob" in k:
                    prob = v[:-1]
                elif "tokenized" in k:
                    target = v.data["input_ids"].t()[1:]
                elif "labels" in k:
                    target = v[1:]
                elif "focus" in k:
                    focus = v
            assert prob is not None and target is not None
            loss_single = self.cross_entropy(prob.view(-1, prob.shape[-1]).contiguous(), target.reshape(-1))
            if focus is not None:
                prob_focus = []
                target_focus = []
                for kidx, fo in enumerate(focus):
                    if len(fo) == 0:
                        continue
                    prob_focus.append(prob[fo - 1, kidx])
                    target_focus.append(target[fo - 1, kidx])
                if len(target_focus) > 0:
                    prob_focus = torch.cat(prob_focus)
                    target_focus = torch.cat(target_focus)
                    loss_focus = self.cross_entropy(prob_focus.contiguous(), target_focus)
                    loss_single += 2 * loss_focus
            if loss is None:
                loss = loss_single
            else:
                loss += loss_single
        return {"loss_caption_ce": loss}

    def loss_caption_mse(self, caption_dist_info):
        feature_tarinfo = caption_dist_info["feature_tarinfo"].transpose(0, 1).sum(1)
        feature_rawinfo = caption_dist_info["feature_rawinfo"].transpose(0, 1).sum(1)
        feature_corinfo = caption_dist_info["feature_corinfo"].transpose(0, 1).sum(1)

        # gap = feature_tarinfo - feature_rawinfo - feature_corinfo
        gap = feature_tarinfo - feature_rawinfo
        loss = self.mse(feature_corinfo, gap)

        return {"loss_caption_mse": loss}

