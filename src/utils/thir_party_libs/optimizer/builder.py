# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import Dict, List, Optional
import torch
from src.utils.thir_party_libs.registry import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizer')
SCHEDULERS = Registry('scheduler')


# OPTIMIZER_BUILDERS = Registry('optimizer builder')


def register_torch_optimizers() -> List:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer(cfg: Dict):
    return OPTIMIZERS.build(cfg)


def build_scheduler(cfg: Dict):
    return SCHEDULERS.build(cfg)
