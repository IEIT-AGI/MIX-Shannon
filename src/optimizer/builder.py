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


# def build_optimizer_constructor(cfg: DictConfig):
#     return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


# def build_optimizer(cfg: DictConfig):
#     optimizer_cfg = copy.deepcopy(cfg)
#     constructor_type = optimizer_cfg.pop('constructor',
#                                          'DefaultOptimizerConstructor')
#     paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
#     optim_constructor = build_optimizer_constructor(
#         dict(
#             type=constructor_type,
#             optimizer_cfg=optimizer_cfg,
#             paramwise_cfg=paramwise_cfg))
#     optimizer = optim_constructor(model)
#     return optimizer

def build_optimizer(cfg: Dict):
    return OPTIMIZERS.build(cfg)


def build_scheduler(cfg: Dict):
    return SCHEDULERS.build(cfg)
