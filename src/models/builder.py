from omegaconf import DictConfig
from src.utils.thir_party_libs.registry import Registry
from typing import Dict, Union

MODELS = Registry('models')
HEADS = Registry('heads')
ENCODERS = Registry('encoder')
# TEXT_ENCODERS = Registry("text_encoder")
TOKENIZERS = Registry('tokenizer')
# EMBEDDINGS = Registry('embedding')
POSITION_EMBEDDING = Registry('position_embedding')  # todo
DECODERS = Registry('decoder')
MATCHES = Registry('matcher')
LOSSES = Registry("loss")
BACKBONES = Registry('backbones')
ANSWER_MASK = Registry('answer_mask')


def build_head(cfg: Union[DictConfig, Dict]):
    """Build head."""
    return HEADS.build(cfg)


def build_encoder(cfg: Union[DictConfig, Dict]):
    """Build head."""
    return ENCODERS.build(cfg)


def build_decoder(cfg: Union[DictConfig, Dict]):
    """Build head."""
    return DECODERS.build(cfg)


def build_matcher(cfg: Union[DictConfig, Dict]):
    """Build head."""
    return MATCHES.build(cfg)


def build_loss(cfg: Union[DictConfig, Dict]):
    return LOSSES.build(cfg)


def build_models(cfg: Union[DictConfig, Dict]):
    return MODELS.build(cfg)


def build_tokenizer(cfg: Union[DictConfig, Dict]):
    return TOKENIZERS.build(cfg)


# def build_text_encoder(cfg: Union[DictConfig, Dict]):
#     return TEXT_ENCODERS.build(cfg)
#
#
# def build_embedding(cfg: Union[DictConfig, Dict]):
#     return EMBEDDINGS.build(cfg)


def build_backbone(cfg: Union[DictConfig, Dict]):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_position_embedding(cfg: Union[DictConfig, Dict]):
    return POSITION_EMBEDDING.build(cfg)


def build_answer_mask(cfg: Union[DictConfig, Dict]):
    return ANSWER_MASK.build(cfg)
