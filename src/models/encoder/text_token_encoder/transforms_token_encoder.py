from typing import Callable, List, Tuple
from loguru import logger
import inspect
from src.models.builder import TEXT_ENCODERS, TOKENIZERS


def register_transformers_tokenizer_and_text_encoder() -> Tuple[List, List]:
    import transformers
    _torch_tokenizers = []
    _torch_text_encoders = []

    is_model: Callable = lambda key: key.endswith("Model")
    is_tokenizer: Callable = lambda key: key.find("Tokenizer") != -1
    is_exist: Callable = lambda key: is_model(key) or is_tokenizer(key)
    is_registered: Callable = lambda k: k not in [TEXT_ENCODERS, TOKENIZERS]
    for module_name in dir(transformers):
        logger.info(module_name)
        if module_name.startswith("__") or not is_exist(module_name):
            continue

        try:
            _cls = getattr(transformers, module_name)
            if inspect.isclass(_cls) and is_registered(_cls):  # todo
                if is_model(module_name):
                    TEXT_ENCODERS.register_module()(_cls)
                    _torch_text_encoders.append(_cls)
                else:
                    TOKENIZERS.register_module()(_cls)
                    _torch_tokenizers.append(_cls)
        except AttributeError as attr_err:
            logger.warning(f"load {module_name} exception: {attr_err}")
            continue
        except ImportError as import_err:
            logger.warning(f"load {module_name} exception: {import_err}")
            continue
        except RuntimeError as runtime_err:
            logger.warning(f"load {module_name} exception: {runtime_err}")
            continue

    return _torch_tokenizers, _torch_text_encoders


if len(TOKENIZERS) == 0:
    register_transformers_tokenizer_and_text_encoder()
    from loguru import logger

    logger.info("register_transformers_tokenizer_and_text_encoder")
