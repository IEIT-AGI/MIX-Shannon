import os
from typing import Optional, Dict, Any, Tuple, Callable, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from transformers import PreTrainedTokenizerBase
from easydict import EasyDict as eDict
from itertools import accumulate
from functools import partial

from .transforms.image_transforms import Compose
from .data_bases import create_positive_map
from src.utils.misc import NestedTensor
from src.utils.running_state import which_one_running_state, RunningStage
from src.datamodules.dataset_field import AIVQA


def ai_vqa_collate_fn(batch: List, is_do_round: bool) -> Dict:

    def get_positive_map():
        positive_map = [bt.pop("positive_map") for bt in batch]
        max_len = max([p.shape[1] for p in positive_map])
        nb_boxes = sum([p.shape[0] for p in positive_map])
        _bt_positive_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)

        idx = list(accumulate([0] + [len(p) for p in positive_map]))
        for pm, start_idx, end_idx in zip(positive_map, idx[:-1], idx[1:]):
            _bt_positive_map[start_idx:end_idx, :pm.shape[1]] = pm

        return _bt_positive_map.float()

    bt_image = [bt.pop("image") for bt in batch]
    bt_image = NestedTensor.from_tensor_list(bt_image, is_do_round)
    bt_positive_map = get_positive_map()

    # output
    bt_output = {}
    for bt in batch:
        for key, value in bt.items():
            if key in bt_output:
                bt_output[key].append(value)
            else:
                bt_output[key] = [value]

    bt_output.update({"image": bt_image})
    bt_output.update({"positive_map": bt_positive_map})

    return bt_output


class AIVQADataset(Dataset):
    # REQUIRED_FIELD = ['head', 'relation', 'tail', 'object_key', 'answer', 'question', 'image_id',
    #                   'reason_path']  # TODO all field

    def __init__(
            self,
            img_path: str,
            annotation_file: str,
            answer_file: str,
            event_file: str,
            img_transforms: partial,  # TODO
            tokenizer: PreTrainedTokenizerBase,
            running_stage: str = 'train'):

        self.img_path: str = img_path
        self.annotation_file: str = annotation_file
        self.answer_file: str = answer_file
        self.event_file: str = event_file
        self.running_stage: str = running_stage

        self.img_transforms: Compose = img_transforms(running_stage)
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.annotations: List[Dict] = self.load_annotation(file=self.annotation_file)
        self.answers: Dict = self.load_json(file=self.answer_file)["answer"]
        self.event: Dict = self.load_json(file=self.event_file)["answer"]

    @staticmethod
    def load_json(file: str) -> Any:
        import json
        with open(file) as f:
            return json.load(f)

    @staticmethod
    def load_annotation(file: str) -> List[Dict]:
        annotations: Dict = AIVQADataset.load_json(file)
        return [anno for img_id, anno in annotations.items()]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        fn = 'get_' + self.running_stage
        if hasattr(self, fn):
            return getattr(self, fn)(index)
        else:
            raise RuntimeError(f"{self} has not {self.running_stage}() function")

    def __len__(self):
        return len(self.annotations)
        # return 40  # TODO debug

    def get_image_info(self, annotation: Dict) -> Dict:
        img_file = os.path.join(self.img_path, annotation["image_id"] + ".jpg")
        image = Image.open(img_file).convert("RGB")
        bbox_info: Dict = self.prepare_box(image, annotation["bbox"])  # TODO
        image, target = self.img_transforms(image, bbox_info)  # TODO

        output = target
        output.update({"image": image})
        return output

    def _encode_multi_hot_labels(self, answer: str, relation: str, fact: str, max_len: int = 10) -> Dict:
        """ Turn an answer into a vector """

        max_fact_index = len(self.event)
        fact_vec = torch.zeros(max_fact_index)

        max_ans_index = len(self.answers)
        ans_vec = torch.zeros(max_ans_index)

        relation_vec = torch.tensor(-1)  # TODO remove ?

        for i in range(max_len):  # TODO bug ?
            fact_index = self.event.get(fact)
            if fact_index is not None:
                if fact_index < max_fact_index:
                    fact_vec[fact_index] += 1

            ans_index = self.answers.get(answer)
            if ans_index is not None:
                if ans_index < max_ans_index:
                    ans_vec[ans_index] += 1

        output = {"answer_label": ans_vec, "relation_label": relation_vec, "fact_label": fact_vec}  # TODO label or ID
        return output

    def get_text_info(self, annotation: Dict) -> eDict:

        def get_kg_gallery() -> str:
            _kg_gallery = annotation['kg_gallery']
            _kg_gallery = [x.split(' ')[0] + ' ' + x.split(' ')[1] + ' ' + x.split(' ')[-1] for x in _kg_gallery]
            _kg_gallery = " <sep> ".join(_kg_gallery)
            return _kg_gallery

        question, answer = annotation['question'], annotation['answer']
        head, tail = annotation['head'], annotation['tail']
        relation = annotation['relation']

        multi_hot_labels: Dict = self._encode_multi_hot_labels(answer, relation, head)
        kg_gallery: str = get_kg_gallery()

        tokens_positive = [[[0, len(question)]]]
        positive_map: torch.Tensor = create_positive_map(self.tokenizer(question, return_tensors="pt"), tokens_positive)

        # output
        output = eDict({k: annotation.get(k) for k in AIVQA.annotation_field if k in annotation})
        output.update(multi_hot_labels)
        output.kg_gallery = kg_gallery
        output.positive_map = positive_map
        output.tokens_positive = tokens_positive

        return output

    def get_train(self, index: int) -> eDict:
        annotation: Dict = self.annotations[index]
        text_info: eDict = self.get_text_info(annotation)
        img_info: Dict = self.get_image_info(annotation)

        # ouput
        sample = text_info
        sample.update(img_info)
        return sample

    @staticmethod
    def prepare_box(image: Image, bbox: List) -> Dict:
        w, h = image.size
        boxes = torch.tensor(bbox).unsqueeze(0)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        return {
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "boxes": boxes
        }  # TODO strange key

    get_validate = get_train
    get_test = get_train  # TODO


class AIVQADataset1(AIVQADataset):

    def __init__(
            self,
            img_path: str,
            annotation_file: str,
            answer_file: str,
            event_file: str,
            relation_file: str,
            img_transforms: partial,  # TODO
            tokenizer: PreTrainedTokenizerBase,
            running_stage: str = 'train'):
        super().__init__(img_path, annotation_file, answer_file, event_file, img_transforms, tokenizer, running_stage)

        self.relation_file: str = relation_file
        self.relation: Dict = self.load_json(file=self.relation_file)["answer"]

    def _encode_multi_hot_labels(self, answer: str, relation: str, fact: str, max_len: int = 10) -> Dict:
        """ Turn an answer into a vector """

        max_fact_index = len(self.event)
        fact_vec = torch.zeros(max_fact_index)

        max_ans_index = len(self.answers)
        ans_vec = torch.zeros(max_ans_index)

        # relation_vec = torch.tensor(-1)  # TODO remove ?
        max_relation_index = len(self.relation)
        relation_vec = torch.zeros(max_relation_index)

        for i in range(max_len):  
            fact_index = self.event.get(fact)
            if fact_index is not None:
                if fact_index < max_fact_index:
                    fact_vec[fact_index] += 1

            ans_index = self.answers.get(answer)
            if ans_index is not None:
                if ans_index < max_ans_index:
                    ans_vec[ans_index] += 1

            relation_index = self.relation.get(relation)
            if relation_index is not None:
                if relation_index < max_relation_index:
                    relation_vec[relation_index] += 1

        output = {"answer_label": ans_vec, "relation_label": relation_vec, "fact_label": fact_vec}  # TODO label or ID
        return output

    @staticmethod
    def add_gate_label(sample):
        # 4 class
        # is_head_tail:0  is_head:1 is_tail:2 is_no_head_tail:3
        gata_label = torch.zeros(4)
        tail = sample.get("tail")
        head = sample.get("head")
        object_key = sample.get("object_key")

        is_head = head.find(object_key) >= 0
        is_tail = tail.find(object_key) >= 0

        if is_head and is_tail:
            gata_label[0] = 1
        elif is_head:
            gata_label[1] = 1
        elif is_tail:
            gata_label[2] = 1
        else:
            gata_label[3] = 1

        sample["gate_label"] = gata_label

    def get_train(self, index: int) -> eDict:
        sample = super().get_train(index)
        self.add_gate_label(sample)  # add gate label

        return sample


class AIVQADataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self, *args, **kwargs):  # TODO add debug.yaml
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        is_stage: Callable = lambda stage_, running_stage: which_one_running_state(stage_) is running_stage.value
        dataset_cls = eval(self.hparams.dataset.dataset_cls)

        # TODO merge three stage
        if not self.data_train and self.hparams.dataset.train_cfg and is_stage(stage, RunningStage.TRAINING):
            kwargs = self.hparams.dataset.train_cfg
            kwargs.running_stage = RunningStage.TRAINING.value
            self.data_train = dataset_cls(**kwargs)

        if not self.data_val and self.hparams.dataset.val_cfg and stage in [
                RunningStage.VALIDATING, RunningStage.FITTING
        ]:
            kwargs = self.hparams.dataset.val_cfg
            kwargs.running_stage = RunningStage.VALIDATING.value
            self.data_val = dataset_cls(**kwargs)

        if not self.data_test and self.hparams.dataset.test_cfg and is_stage(stage, RunningStage.TESTING):
            kwargs = self.hparams.dataset.test_cfg
            kwargs.running_stage = RunningStage.TESTING.value
            self.data_test = AIVQADataset(**kwargs)

    def train_dataloader(self) -> DataLoader:
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_train
        kwargs.shuffle = kwargs.get("shuffle", True)
        return DataLoader(**kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_val
        kwargs.shuffle = kwargs.get("shuffle", False)
        return DataLoader(**kwargs)

    def test_dataloader(self) -> DataLoader:
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_test
        kwargs.shuffle = kwargs.get("shuffle", False)
        return DataLoader(**kwargs)
