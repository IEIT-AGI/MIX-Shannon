import os
from typing import Optional, Dict, Any, Tuple, Callable, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import torch
from src.utils.running_state import which_one_running_state, RunningStage
from src.datamodules.dataset_field import AIVQA
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
from einops import rearrange
import numpy as np
import math
from torch.utils.data._utils.collate import default_collate


def vqai_collate_fn(batch: List) -> Dict:

    # output
    bt_output = {}
    require_key = ["input_image", "gt_image"]
    for key in require_key:
        for bt in batch:
            if key not in bt_output:
                bt_output[key] = [bt.pop(key)]
            else:
                bt_output[key].append(bt.pop(key))

    _bt = default_collate(batch)
    for key, value in _bt.items():
        bt_output[key] = value

    return bt_output


class VQAIDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 sample_file: str,
                 img_random_crop: "torchvision.transforms" = None,
                 img_flip: "torchvision.transforms" = None,
                 max_resize_resolution: int = 512,
                 min_resize_resolution: int = 512,
                 blip2_data_processor: "DictConfig" = None,
                 causal_feature: str = None,
                 num_causal_samples=8,
                 running_stage: str = 'train'):

        self.image_dir: str = image_dir
        self.max_resize_resolution: int = max_resize_resolution
        self.min_resize_resolution: int = min_resize_resolution
        self.running_stage: str = running_stage

        self.img_random_crop: RandomCrop = img_random_crop
        self.img_flip: RandomHorizontalFlip = img_flip

        self.samples_info = self.load_json(sample_file)
        self.num_causal_samples = num_causal_samples
        self.causal_feature = self.load_causal_file(causal_feature)
        self.blip2_vis_processor, self.blip2_text_processor = self.instantiate_blip2_data_processor(
            blip2_data_processor)

    @staticmethod
    def instantiate_blip2_data_processor(cfg: "DictConfig") -> Tuple[Callable, Callable]:
        from stable_diffusion.ldm.util import instantiate_from_config
        text_processor: Callable = instantiate_from_config(cfg.text_processor)
        vis_processor: Callable = instantiate_from_config(cfg.vis_processor)
        return vis_processor, text_processor

    @staticmethod
    def load_json(file: str) -> Any:
        import json
        with open(file) as f:
            return json.load(f)

    @staticmethod
    def load_causal_file(causal_feat_file: str) -> Dict:
        if causal_feat_file is None:
            return None
        else:
            causal_data = torch.load(causal_feat_file)
            return {d["sample_name"]: d for d in causal_data}

    def load_blip_data(self, path: str) -> List[Dict]:

        def get_all_files(dir_path, file_format="*.json"):
            import glob
            files = glob.glob(os.path.join(dir_path, file_format))
            return [os.path.basename(file) for file in files]

        sample_files = get_all_files(path)
        blip_info = []
        replace_name_fn: Callable = lambda sample_name, new_str: sample_name.replace("+", "").replace(".json", new_str)
        for sample_name in sample_files:
            single_info = {
                "sample_name": replace_name_fn(sample_name, ""),
                "image1_name": replace_name_fn(sample_name, "a.jpg"),
                "image2_name": replace_name_fn(sample_name, "b.jpg"),
                "sent": sample_name
            }
            blip_info.append(single_info)

        return blip_info

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        fn = 'get_' + self.running_stage
        if hasattr(self, fn):
            return getattr(self, fn)(index)
        else:
            raise RuntimeError(f"{self} has not {self.running_stage}() function")

    def __len__(self):
        return len(self.samples_info)

    def get_image_info(self, info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        img0_path = os.path.join(self.image_dir, info.get("image1_name"))
        img1_path = os.path.join(self.image_dir, info.get("image2_name"))

        img0_data = Image.open(img0_path).convert("RGB")
        img1_data = Image.open(img1_path).convert("RGB")

        # image resize
        random_resize = torch.randint(self.min_resize_resolution, self.max_resize_resolution + 1, ()).item()
        img0_data = img0_data.resize((random_resize, random_resize), Image.Resampling.LANCZOS)
        img1_data = img1_data.resize((random_resize, random_resize), Image.Resampling.LANCZOS)

        img0_data = rearrange(2 * torch.tensor(np.array(img0_data)).float() / 255 - 1, "h w c -> c h w")
        img1_data = rearrange(2 * torch.tensor(np.array(img1_data)).float() / 255 - 1, "h w c -> c h w")

        img_concat = torch.cat((img0_data, img1_data))
        img0_data, img1_data = self.img_flip(self.img_random_crop(img_concat)).chunk(2)

        return img0_data, img1_data

    def get_blip2_train_data(self, info: Dict) -> Dict:
        img0_path = os.path.join(self.image_dir, info["image1_name"])
        img0 = Image.open(img0_path).convert("RGB")

        image = self.blip2_vis_processor(img0)
        question = self.blip2_text_processor(info["question"])

        answer_weight = {}
        loss_weight = []  # process answer is empty
        info_list: List = [info["sent"]]
        for answer in info_list:
            if answer == "":
                loss_weight.append(0)
                answer = "END"
            else:
                loss_weight.append(1)

            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(info_list)
            else:
                answer_weight[answer] = 1 / len(info_list)

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
            "blip2_loss_weight": loss_weight
        }

    def get_blip2_val_data(self, info: Dict, prompt: str = " let's do the causal reasoning step by step.") -> Dict:
        img0_path = os.path.join(self.image_dir, info["image1_name"])
        img0 = Image.open(img0_path).convert("RGB")

        image = self.blip2_vis_processor(img0)
        q_txt = info["question"] + prompt
        question = self.blip2_text_processor(q_txt)

        answer_weight = {}
        loss_weight = []  # process answer is empty
        info_list: List = [""]
        for answer in info_list:
            if answer == "":
                loss_weight.append(0)
                answer = "END"
            else:
                loss_weight.append(1)

            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(info_list)
            else:
                answer_weight[answer] = 1 / len(info_list)

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
            "blip2_loss_weight": loss_weight
        }

    def get_blip2_data(self, info: Dict) -> Dict:
        if self.running_stage == "train":
            return self.get_blip2_train_data(info)
        else:
            return self.get_blip2_val_data(info)

    def get_stable_diffusion_train_data(self, info: Dict) -> Dict:
        img0_data, img1_data = self.get_image_info(info=info)
        prompt = info["sent"]  # prompt

        #output
        return dict(edited=img1_data, edit=dict(c_concat=img0_data, c_crossattn=prompt))

    def get_causal_data(self, info: Dict) -> Dict:

        def get_negative_sample(sample_name: str) -> torch.Tensor:
            import random
            random_sample_names = random.sample(list(self.causal_feature.keys()), self.num_causal_samples + 1)
            _negative_sample = []
            for rnd_name in random_sample_names:
                if rnd_name != sample_name:
                    _negative_sample.append(self.causal_feature[rnd_name]["causal_eos_token_feature"])

            return torch.vstack(_negative_sample)[:self.num_causal_samples]

        sample_name = info["sample_name"]
        if sample_name in self.causal_feature:
            positive_sample = self.causal_feature[sample_name]["causal_eos_token_feature"]
            negative_sample = get_negative_sample(sample_name)
            num_pos = min(len(positive_sample), self.num_causal_samples // 2)
            positive_sample = positive_sample[:num_pos]
            samples = torch.vstack([positive_sample, negative_sample])[:self.num_causal_samples]
        else:
            num_pos = 0
            negative_sample = get_negative_sample(sample_name)
            samples = negative_sample

        return {"num_pos": num_pos, "samples": samples}

    def get_train(self, index: int):
        sample_info = self.samples_info[index]
        stable_diffusion_data = self.get_stable_diffusion_train_data(
            sample_info)  # data input to stable-diffusion model

        # input to blip2 model
        blip2_data = self.get_blip2_data(sample_info)
        causal_data = self.get_causal_data(sample_info)

        # output
        output = stable_diffusion_data
        output["blip2_data"] = blip2_data
        output["causal_data"] = causal_data
        return output

    def get_validate(self, index: int):
        sample_info = self.samples_info[index]
        stable_diffusion_data = self.get_stable_diffusion_validate_data(sample_info)
        blip2_data = self.get_blip2_data(sample_info)

        # output
        output = stable_diffusion_data
        output["blip2_data"] = blip2_data
        return output

    def get_stable_diffusion_validate_data(self, sample_info) -> Dict:
        input_img_path = os.path.join(self.image_dir, sample_info.get("image1_name"))
        gt_img_path = os.path.join(self.image_dir, sample_info.get("image2_name"))

        input_image = Image.open(input_img_path).convert("RGB")
        gt_image = Image.open(gt_img_path).convert("RGB")

        width, height = input_image.size
        factor = self.max_resize_resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64

        # image resize
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
        input_image = input_image.resize((self.max_resize_resolution, self.max_resize_resolution))

        gt_image = ImageOps.fit(gt_image, (width, height), method=Image.Resampling.LANCZOS)
        gt_image = gt_image.resize((self.max_resize_resolution, self.max_resize_resolution)).resize((width, height))

        img_a = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        img_a = rearrange(img_a, "h w c -> c h w")

        input_img_idx = sample_info.get("image1_name")[:-5]

        return {"input_image": input_image, "gt_image": gt_image, "input_img_idx": input_img_idx, "img_a": img_a}

    get_test = get_validate  # TODO


class VQAIDataModule(LightningDataModule):
    """
    Example of LightningDataModule for VQAIDataModule dataset.

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

        # TODO merge three stage
        if not self.data_train and self.hparams.dataset.train_cfg and is_stage(stage, RunningStage.TRAINING):
            kwargs = self.hparams.dataset.train_cfg
            kwargs.running_stage = RunningStage.TRAINING.value
            self.data_train = VQAIDataset(**kwargs)

        if not self.data_val and getattr(self.hparams.dataset, "val_cfg",
                                         False) and stage in [RunningStage.VALIDATING, RunningStage.FITTING]:
            kwargs = self.hparams.dataset.val_cfg
            kwargs.running_stage = RunningStage.VALIDATING.value
            self.data_val = VQAIDataset(**kwargs)

        if not self.data_test and getattr(self.hparams.dataset, "test_cfg", False) and is_stage(
                stage, RunningStage.TESTING):
            kwargs = self.hparams.dataset.test_cfg
            kwargs.running_stage = RunningStage.TESTING.value
            self.data_test = VQAIDataset(**kwargs)

    def train_dataloader(self) -> DataLoader:
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_train
        kwargs.shuffle = kwargs.get("shuffle", True)
        return DataLoader(**kwargs)

    def val_dataloader(self) -> DataLoader:
        self.hparams.dataloader.batch_size = 1 if getattr(self.hparams.dataloader, "batch_size") else 1
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_val
        kwargs.shuffle = kwargs.get("shuffle", False)
        kwargs.collate_fn = vqai_collate_fn
        return DataLoader(**kwargs)

    def test_dataloader(self) -> DataLoader:
        self.hparams.dataloader.batch_size = 1 if getattr(self.hparams.dataloader, "batch_size") else 1
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_test
        kwargs.shuffle = kwargs.get("shuffle", False)
        return DataLoader(**kwargs)
