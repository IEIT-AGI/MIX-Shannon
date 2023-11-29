from typing import Optional, Dict, Any, Tuple, List, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import os
import random
from .data_bases import create_positive_map
from PIL import Image
import torch
from src.utils.running_state import which_one_running_state, RunningStage
from transformers import PreTrainedTokenizerBase
from functools import partial
from .transforms.image_transforms import Compose
from src.datamodules.dataset_field import FRECField


def field_align_in_FREC(batch: List, field: str) -> torch.Tensor:
    max_len = max([b[field].shape[1] for b in batch])
    nb_boxes = sum([b[field].shape[0] for b in batch])
    data = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
    cur_count = 0
    for bt in batch:
        cur_pos = bt.pop(field)
        data[cur_count:cur_count + len(cur_pos), :cur_pos.shape[1]] = cur_pos
        cur_count += len(cur_pos)
    assert cur_count == len(data)
    return data.float()


def frec_collate_fn(batch: List, is_do_round: bool) -> Dict:

    def get_batch_img():
        from src.utils.misc import NestedTensor
        raw_images: List = []
        for dt in batch:
            raw_images.append(dt.pop("image"))
        return NestedTensor.from_tensor_list(raw_images, is_do_round)

    bt_output: Dict = {}
    bt_output["images"] = get_batch_img()

    for field in FRECField.align_field:
        if field in batch[0]:
            bt_output[field] = field_align_in_FREC(batch, field)

    for field in batch[0].keys():
        if field is not 'area':  # todo
            bt_output[field] = [bt[field] for bt in batch]

    # todo rewrite
    # if "positive_map_raw" in batch[0]:
    #     max_len = max([b["positive_map_raw"].shape[1] for b in batch])
    #     nb_boxes = sum([b["positive_map_raw"].shape[0] for b in batch])
    #     batched_pos_map_raw = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
    #     cur_count = 0
    #     for b in batch:
    #         cur_pos = b["positive_map_raw"]
    #         batched_pos_map_raw[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
    #         cur_count += len(cur_pos)
    #     assert cur_count == len(batched_pos_map_raw)
    #     bt_output["positive_map_raw"] = batched_pos_map_raw.float()
    #
    # if "positive_map_cor" in batch[0]:
    #     max_len = max([b["positive_map_cor"].shape[1] for b in batch])
    #     nb_boxes = sum([b["positive_map_cor"].shape[0] for b in batch])
    #     batched_pos_map_cor = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
    #     cur_count = 0
    #     for b in batch:
    #         cur_pos = b["positive_map_cor"]
    #         batched_pos_map_cor[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
    #         cur_count += len(cur_pos)
    #     assert cur_count == len(batched_pos_map_cor)
    #     bt_output["positive_map_cor"] = batched_pos_map_cor.float()
    #
    # if "positive_map_cor_first" in batch[0]:
    #     max_len = max([b["positive_map_cor_first"].shape[1] for b in batch])
    #     nb_boxes = sum([b["positive_map_cor_first"].shape[0] for b in batch])
    #     batched_pos_map_cor_first = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
    #     cur_count = 0
    #     for b in batch:
    #         cur_pos = b["positive_map_cor_first"]
    #         batched_pos_map_cor_first[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
    #         cur_count += len(cur_pos)
    #     assert cur_count == len(batched_pos_map_cor_first)
    #     bt_output["positive_map_cor_first"] = batched_pos_map_cor_first.float()

    # bt_output["infos"] = tuple(batch)
    return bt_output


class FREC(Dataset):

    def __init__(self,
                 image_dir: str,
                 annotation_dir: str,
                 tokenizer: PreTrainedTokenizerBase,
                 data_file: str,
                 image_transforms: partial,
                 running_stage: str = 'train'):

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.running_stage = running_stage
        self.data_file_name = data_file

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.transforms: Compose = image_transforms(running_stage)
        self.annotations: List[Dict] = self.load_annotation(file=data_file)

    def load_annotation(self, file: str, limits: int = 2) -> List[Dict]:
        annotations = []

        with open(file, "r") as f:
            imgs_name = [fn.strip() for fn in f.readlines()]

        # if "train_names" in file:
        #     imgs_name = imgs_name[:limits]  # todo debug
        # else:
        #     imgs_name = imgs_name[:limits]  # todo debug

        import json
        for img_name in imgs_name:
            anno_file = os.path.join(self.annotation_dir, img_name + ".json")
            info = json.load(open(anno_file, "r"))
            box_info, name, annos = info["box_info"], info["name"], info["annos"]

            for anno in annos:
                corr_infos = []
                for correction_info in anno["correction_infos"]:
                    box_idx_str = str(correction_info["box_idx"])
                    if len(correction_info["sent"]) and box_idx_str in box_info:
                        corr_info = {"sent": correction_info["sent"], "bbox": box_info[box_idx_str]}
                        corr_infos.append(corr_info)
                anno["correction_infos"] = corr_infos
                anno["rationales"] = [r for r in anno["rationales"] if len(r) > 0]
                anno.update({"name": name, "anno_name": name + "_" + anno["anno_name"]})
                if len(corr_infos) > 0:
                    annotations.append(anno)

        return annotations

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        fn = 'get_' + self.running_stage
        if hasattr(self, fn):
            return getattr(self, fn)(index)
        else:
            raise RuntimeError(f"{self} has not {self.running_stage}() function")

    def __len__(self):
        return len(self.annotations)  #todo debug
        # return 10

    def get_train(self, index: int):
        annotation = self.annotations[index]
        raw_sent = annotation["raw_sent"]
        rationale = random.choice(annotation["rationales"])
        correction_info = random.choice(annotation["correction_infos"])

        tokens_positive_raw = [[[0, len(raw_sent)]]]
        positive_map_raw = create_positive_map(self.tokenizer(raw_sent, return_tensors="pt"), tokens_positive_raw)
        tokens_positive_cor = [[[0, len(correction_info["sent"])]]]
        positive_map_cor = create_positive_map(self.tokenizer(correction_info["sent"], return_tensors="pt"),
                                               tokens_positive_cor)

        image = Image.open(os.path.join(self.image_dir, annotation["name"] + ".jpg")).convert("RGB")
        target = self.prepare_box(image, correction_info)
        image, target = self.transforms(image, target)

        return_infos = {
            "name": annotation["anno_name"],
            "image": image,
            "raw_sent": raw_sent,
            "cor_sent": correction_info["sent"],
            'rationale': rationale,
            "tokens_positive_raw": tokens_positive_raw,
            "tokens_positive_cor": tokens_positive_cor,
            "positive_map_raw": positive_map_raw,
            "positive_map_cor": positive_map_cor
        }
        return_infos.update(target)
        return return_infos

    def get_validate(self, index: int):
        annotation = self.annotations[index]
        raw_sent = annotation["raw_sent"]
        rationale_list = annotation["rationales"]
        correction_infos = annotation["correction_infos"]
        cor_sent_list = [correction_info["sent"] for correction_info in correction_infos]

        tokens_positive_raw = [[[0, len(raw_sent)]]]
        positive_map_raw = create_positive_map(self.tokenizer(raw_sent, return_tensors="pt"), tokens_positive_raw)

        tokens_positive_cors = []
        positive_map_cors = []
        for idxc, correction_info in enumerate(correction_infos):
            tokens_positive_cor = [[[0, len(correction_info["sent"])]]]
            positive_map_cor = create_positive_map(self.tokenizer(correction_info["sent"], return_tensors="pt"),
                                                   tokens_positive_cor)
            tokens_positive_cors.append(tokens_positive_cor)
            positive_map_cors.append(positive_map_cor)
        tokens_positive_cor_first = tokens_positive_cors[0]
        positive_map_cor_first = positive_map_cors[0]

        image = Image.open(os.path.join(self.image_dir, annotation["name"] + ".jpg")).convert("RGB")
        target_list = [self.prepare_box(image, correction_info) for correction_info in correction_infos]
        boxes = [target["boxes"] for target in target_list]
        boxes = torch.cat(boxes)
        target = target_list[0]
        target["boxes"] = boxes
        image, target = self.transforms(image, target)

        return_infos = {
            "image_name": annotation["name"],
            "name": annotation["anno_name"],
            "image": image,
            "raw_sent": raw_sent,
            "cor_sent_list": cor_sent_list,
            'rationale_list': rationale_list,
            "tokens_positive_raw": tokens_positive_raw,
            "tokens_positive_cors": tokens_positive_cors,
            "positive_map_raw": positive_map_raw,
            "positive_map_cors": positive_map_cors,
            "tokens_positive_cor_first": tokens_positive_cor_first,
            "positive_map_cor_first": positive_map_cor_first,
        }
        return_infos.update(target)
        return return_infos

    def prepare_box(self, image, correction_info):
        w, h = image.size
        boxes = torch.tensor(correction_info["bbox"]).unsqueeze(0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        return {
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "boxes": boxes
        }

    get_test = get_validate


class FRECDataModule(LightningDataModule):
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

    def __init__(self, *args, **kwargs):
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
        if not self.data_train and self.hparams.dataset.train_cfg and is_stage(stage, RunningStage.TRAINING):
            kwargs = self.hparams.dataset.train_cfg
            kwargs.running_stage = RunningStage.TRAINING.value
            self.data_train = FREC(**kwargs)

        if not self.data_val and self.hparams.dataset.val_cfg and stage in [
                RunningStage.VALIDATING, RunningStage.FITTING
        ]:
            kwargs = self.hparams.dataset.val_cfg
            kwargs.running_stage = RunningStage.VALIDATING.value
            self.data_val = FREC(**kwargs)

        if not self.data_test and self.hparams.dataset.get("test_cfg", False) and is_stage(stage, RunningStage.TESTING):
            kwargs = self.hparams.dataset.test_cfg
            kwargs.running_stage = RunningStage.TESTING.value
            self.data_test = FREC(**kwargs)

    def train_dataloader(self):
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_train
        kwargs.shuffle = kwargs.get("shuffle", True)
        kwargs.drop_last = kwargs.get("drop_last", True)

        return DataLoader(**kwargs)

    def val_dataloader(self):
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_val
        kwargs.shuffle = kwargs.get("shuffle", False)

        return DataLoader(**kwargs)

    def test_dataloader(self):
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_test
        kwargs.shuffle = kwargs.get("shuffle", False)

        return DataLoader(**kwargs)
