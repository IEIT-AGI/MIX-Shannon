# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
from typing import Any, Dict, List, Optional

import torch
import torchvision
from torch import Tensor


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(do_round, batch):
    raw_images = []
    for d in batch:
        raw_images.append(d.pop("image"))
    images = NestedTensor.from_tensor_list(raw_images, do_round)

    final_batch = {}
    final_batch["images"] = images

    if "positive_map_raw" in batch[0]:
        max_len = max([b["positive_map_raw"].shape[1] for b in batch])
        nb_boxes = sum([b["positive_map_raw"].shape[0] for b in batch])
        batched_pos_map_raw = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for b in batch:
            cur_pos = b["positive_map_raw"]
            batched_pos_map_raw[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)
        assert cur_count == len(batched_pos_map_raw)
        final_batch["positive_map_raw"] = batched_pos_map_raw.float()

    if "positive_map_cor" in batch[0]:
        max_len = max([b["positive_map_cor"].shape[1] for b in batch])
        nb_boxes = sum([b["positive_map_cor"].shape[0] for b in batch])
        batched_pos_map_cor = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for b in batch:
            cur_pos = b["positive_map_cor"]
            batched_pos_map_cor[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)
        assert cur_count == len(batched_pos_map_cor)
        final_batch["positive_map_cor"] = batched_pos_map_cor.float()

    if "positive_map_cor_first" in batch[0]:
        max_len = max([b["positive_map_cor_first"].shape[1] for b in batch])
        nb_boxes = sum([b["positive_map_cor_first"].shape[0] for b in batch])
        batched_pos_map_cor_first = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for b in batch:
            cur_pos = b["positive_map_cor_first"]
            batched_pos_map_cor_first[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)
        assert cur_count == len(batched_pos_map_cor_first)
        final_batch["positive_map_cor_first"] = batched_pos_map_cor_first.float()

    final_batch["infos"] = tuple(batch)
    return final_batch


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list, do_round=False):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return cls(tensor, mask)

    def __repr__(self):
        return repr(self.tensors)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    assert input.shape[0] != 0 or input.shape[1] != 0, "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(input.transpose(0, 1), size, scale_factor, mode, align_corners).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)



def targets_to(targets: List[Dict[str, Any]], device):
    """Moves the target dicts to the given device."""
    excluded_keys = [
        "questionId",
        "tokens_positive",
        "tokens",
        "dataset_name",
        "sentence_id",
        "original_img_id",
        "nb_eval",
        "task_id",
        "original_id",
    ]
    return [{k: v.to(device) if k not in excluded_keys else v for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]
