from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AIVQA:
    annotation_field: Tuple[str] = ('head', 'relation', 'tail',
                                    'object_key', 'answer', 'question',
                                    'image_id', 'reason_path')
    grounding_field: Tuple[str] = ("orig_size", "image_id", "boxes")
    validation_field: Tuple[str] = ("answer", "answer_label", "fact_label",
                                    "orig_size", "image_id", "boxes")
    dataset_field: Tuple[str] = ("image", "question", "answer_label", "relation_label",
                                 "fact_label", "boxes", "image_id", "positive_map", "answer", "gate_label")


@dataclass(frozen=True)
class FRECField:
    align_field: Tuple[str] = ('positive_map_raw', 'positive_map_cor', 'positive_map_cor_first')
    validation_field: Tuple[str] = ('name', 'boxes', 'orig_size', 'raw_sent', 'cor_sent_list', 'rationale_list')
    special_field: Tuple[str] = ('cor_sent_list', 'rationale_list')
