import torch
from typing import List
import inspect
from src.models.builder import EMBEDDINGS


def register_torch_embedding() -> List:
    _torch_embeddings = []
    for module_name in dir(torch.nn.modules.sparse):
        if module_name.startswith('__'):
            continue
        _embed = getattr(torch.nn.modules.sparse, module_name)
        if inspect.isclass(_embed) and module_name.find("Embedding") != -1:
            EMBEDDINGS.register_module()(_embed)
            _torch_embeddings.append(module_name)
    return _torch_embeddings


# torch_embeddings = register_torch_embedding()

if len(EMBEDDINGS) < 4:  # todo
    register_torch_embedding()
    from loguru import logger

    logger.info("register_torch_embedding")
