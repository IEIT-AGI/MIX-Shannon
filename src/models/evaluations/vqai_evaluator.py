from typing import Dict, List, Callable, Tuple, Any
from omegaconf import DictConfig
import torch
from torch.nn import functional as F
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip
from loguru import logger
from tqdm import tqdm


class DummyDataset(Dataset):

    def __init__(self, img_dir: str, assert_num: int, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.assert_num = assert_num  # len(seeds)*len(method)+2
        self.samples = self.get_all_samples_info()

    def __getitem__(self, item):
        sample = self.samples[item]

        output = {}
        for name, paths in sample.items():
            output["name"] = name
            output["files"] = paths
            output["data"], output["data_info"] = self._img_process(paths)

        return output

    def _img_process(self, files: str):
        imgs_data = []
        imgs_info = {}
        for idx, file in enumerate(files):
            img = Image.open(file)
            img = self.transform(img) if self.transform else img
            imgs_data.append(img)

            file_name = file.split("/")[-1]
            if file_name.find("input_img") != -1:
                imgs_info["input_img_idx"] = idx
            elif file_name.find("target_img") != -1:
                imgs_info["target_img_idx"] = idx
            else:
                if "generate_img_idx" in imgs_info:
                    imgs_info["generate_img_idx"].append(idx)
                else:
                    imgs_info["generate_img_idx"] = [idx]

        return imgs_data, imgs_info

    def __len__(self):
        return len(self.samples)

    def get_all_samples_info(self):
        sample_names = [fn for fn in os.listdir(self.img_dir) if not fn.startswith(".")]
        samples = []
        for sample_name in sample_names:
            sample_path = os.path.join(self.img_dir, sample_name)
            img_files = [fn for fn in os.listdir(sample_path) if fn.endswith(".jpg")]
            if len(img_files) != self.assert_num:
                continue
            img_files = [os.path.join(sample_path, n) for n in img_files]
            samples.append({sample_name: img_files})
        return samples


class VQAIEvaluator:

    def __init__(self,
                 seeds: Tuple[int],
                 sample_file: str,
                 generate_sample_dir: str,
                 clip_model: str = "ViT-B/32",
                 clip_model_device: str = "cpu"):
        # self.cfg = cfg
        self.seeds = seeds
        self.sample_file = sample_file
        self.generate_sample_dir = generate_sample_dir
        self.clip_model_device = clip_model_device

        self.model, self.preprocess = clip.load(clip_model, device=clip_model_device)
        self.dataloader = self.build_dataloader()
        self.category = self.read_category_file(sample_file)

    def read_category_file(self, sample_file: str) -> Dict:
        # this file includes scenery variation(SV), more entities(ME), fewer entities(FE) , entities variation(EV),and emotion variation(EMV)
        import json

        with open(sample_file) as f:
            return json.load(f)

    def build_dataloader(self):
        single_folder_imgs = len(self.seeds) + 2
        dataset = DummyDataset(img_dir=self.generate_sample_dir,
                               assert_num=single_folder_imgs,
                               transform=self.preprocess)
        # return DataLoader(dataset=dataset, batch_size=single_folder_imgs, num_workers=single_folder_imgs // 2)
        return DataLoader(dataset=dataset)

    def compute_all_category_similarity(self, scores):
        category_dict = {
            "total sample": "total sample",
            "scenery variation(SV)": "scenery variation",
            "more entities(ME)": "more entities",
            "fewer entities(FE)": "fewer entities",
            "entities variation(EV)": "entities variation",
            "emotion variation(EMV)": "emotion variation"
        }
        for category, name in category_dict.items():
            sample = self.category.get(name)

            sample_score = {}
            for name in sample:
                if name in scores:
                    sample_score[name] = scores.get(name)

            logger.info(f"{category} have {len(sample_score)} samples:")
            if len(sample_score) == 0:
                continue
            else:
                self.compute_similarity(sample_score)

    def compute_similarity(self, scores):

        def get_target_vs_generate():
            pred_scores = []
            for k, v in scores.items():
                pred_score = v["LGD"]["target_vs_generate"]
                pred_scores.append(pred_score)
            return pred_scores

        def comput_auc_score(data: "numpy.ndarray", method: str = "mean") -> float:
            auc_mean: Callable = lambda d, p: (d > p).sum() / d.size
            auc_max: Callable = lambda d, p: (d.max(1) > p).sum() / len(d)

            compute_auc: Callable = auc_mean if method == "mean" else auc_max

            auc = []
            for p in [num for num in list(range(100, 0, -1))]:
                single_auc = compute_auc(pred_scores, p)
                auc.append(single_auc)
            score = sum(auc) / 100
            return score

        pred_scores = torch.vstack(get_target_vs_generate()).cpu().detach().numpy()

        sim_avgs = pred_scores.mean() / 100
        sim_bests = pred_scores.max(1).mean() / 100
        auc_avgs = comput_auc_score(pred_scores, method="mean")
        auc_bests = comput_auc_score(pred_scores, method="max")

        logger.info(f"Sim-Avg:{sim_avgs}")
        logger.info(f"Sim-Best:{sim_bests}")
        logger.info(f"AUC-Avg:{auc_avgs}")
        logger.info(f"AUC-Best:{auc_bests}")

    @torch.no_grad()
    def __call__(self) -> Dict:
        logit_scale = self.model.logit_scale.exp()
        scores = {}
        cal_score: Callable = lambda a, b: (logit_scale * (a * b).sum(1))

        def get_idx(data_info: Dict) -> Tuple:
            input_img_idx = data_info.get("input_img_idx")
            target_img_idx = data_info.get("target_img_idx")
            generate_img_idx = torch.hstack(data_info.get("generate_img_idx"))

            return input_img_idx, target_img_idx, generate_img_idx

        for batch in tqdm(self.dataloader, desc="VQAIEvaluator Processing"):
            imgs = torch.vstack(batch["data"])
            img_features = self.model.encode_image(imgs.to(self.clip_model_device))
            img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)

            input_idx, gt_idx, gen_idx = get_idx(batch["data_info"])
            src_img_feat = img_features[input_idx]
            gt_img_feat = img_features[gt_idx]
            lgd_img_feat = img_features[gen_idx]

            sample_name = batch["name"][0]
            single_score = {}
            single_score["input_vs_target"] = cal_score(src_img_feat, gt_img_feat)
            single_score["LGD"] = {
                "input_vs_generate": cal_score(src_img_feat, lgd_img_feat),
                "target_vs_generate": cal_score(gt_img_feat, lgd_img_feat)
            }
            scores[sample_name] = single_score

        self.compute_all_category_similarity(scores)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import ast
    cfg_file = os.path.join(os.path.expanduser('~'), "imix2.0/configs/model/LGD.yaml")
    cfg = OmegaConf.load(cfg_file)
    eval_cfg = cfg.evaluate1
    eval_cfg.seeds = ast.literal_eval(eval_cfg.seeds)

    VQAIEval_obj = VQAIEvaluator(seeds=eval_cfg.seeds,
                                 sample_file=eval_cfg.sample_file,
                                 clip_model=eval_cfg.clip_model,
                                 generate_sample_dir=eval_cfg.sample_dir,
                                 clip_model_device="cpu")
    VQAIEval_obj()
