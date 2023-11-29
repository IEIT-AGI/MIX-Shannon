from typing import Optional, Dict, Any, Tuple, List, Union, Callable
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import dgl
import copy
from src.utils.running_state import which_one_running_state, RunningStage
from transformers import PreTrainedTokenizerBase


def recipe_collate_fn(batch: List) -> Dict:  # TODO
    img_feature = [bt["img_feature"] for bt in batch]
    ingredients_feature = [bt["ingredients_feature"] for bt in batch]
    instruction_feature = [bt["instruction_feature"] for bt in batch]
    title_feature = [bt["title_feature"] for bt in batch]
    graph = [bt["graph"] for bt in batch]
    title = [bt["title"][0] for bt in batch]

    bt_graph = dgl.batch(graph)
    bt_img_feature = torch.cat(img_feature)
    bt_ingredients_feature = torch.cat(ingredients_feature)
    bt_instruction_feature = torch.cat(instruction_feature)
    bt_title_feature = torch.stack(title_feature)
    bt_title = torch.hstack([torch.tensor(t) for t in title])
    bt_single_recipe_img_nums = [len(img_feat) for img_feat in img_feature]
    bt_labels = torch.tensor(list(range(len(bt_single_recipe_img_nums))) * 2)

    return {"img_feature": bt_img_feature,
            "instruction_feature": bt_instruction_feature,
            "ingredients_feature": bt_ingredients_feature,
            "title_feature": bt_title_feature,
            "graph": bt_graph,
            "title": bt_title,
            "single_recipe_img_nums": bt_single_recipe_img_nums,
            "labels": bt_labels}


class RecipeDataset(Dataset):
    REMOVE_TITLE_CHARS = ["?", "\\'", ".", ","]
    REMOVE_INSTRUCTION_CHARS = ["?", "\'", ".", ","]

    def __init__(self,
                 recipe_text_feature: str,
                 image_feature: str,
                 vocab: PreTrainedTokenizerBase,  # Vocab type
                 max_sentence_len: int,
                 running_stage: str = 'train'):
        self.recipe_text_feature = recipe_text_feature
        self.image_feature = image_feature
        self.max_sentence_len = max_sentence_len
        self.running_stage = running_stage
        self.recipe_samples = self.merge_recipe_text_and_img()
        self.vocab: PreTrainedTokenizerBase = vocab

    def merge_recipe_text_and_img(self) -> List:
        def load_pkl(file: str) -> List:
            import pickle
            with open(file, "rb") as f:
                return pickle.load(f)

        recipe_samples = []
        recipe_text = load_pkl(self.recipe_text_feature)
        recipe_img = load_pkl(self.image_feature)

        for text_info, img_info in zip(recipe_text, recipe_img):
            assert text_info["title"] == img_info["title"]
            img_info.update(text_info)
            recipe_samples.append(img_info)

        return recipe_samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        fn = 'get_' + self.running_stage
        if hasattr(self, fn):
            return getattr(self, fn)(index)
        else:
            raise RuntimeError(f"{self} has not {self.running_stage}() function")

    def __len__(self):
        return len(self.recipe_samples)
        # return 64  # TODO debug

    def generate_graph(self,
                       instruction: List,
                       main_ingredients: List,
                       ins2ing_node_connection: np.ndarray = None
                       ) -> dgl.DGLGraph:
        def token_and_pads(data: List, replace_chars: List) -> List:
            data_token_pads = []
            for dt in data:
                dt = self.replace_char(input_str=dt, old_replace_chars=replace_chars)
                dt = dt.split(" ")
                dt_tokens = [self.get_token(word) for word in dt if len(word)]
                dt_tokens_add_pads = self.get_pad(dt_tokens, self.max_sentence_len)
                data_token_pads.append(dt_tokens_add_pads)
            return data_token_pads

        main_ingredients_token = token_and_pads(main_ingredients, self.REMOVE_TITLE_CHARS)
        instruction_token = token_and_pads(instruction, self.REMOVE_INSTRUCTION_CHARS)
        main_ingredients_len = len(main_ingredients_token)
        instruction_len = len(instruction_token)
        if ins2ing_node_connection is None:
            # w2s_w = {i: {j: 1 for j in range(main_ingredients_len)} for i in range(instruction_len)}
            ins2ing_node_connection = np.ones((instruction_len, main_ingredients_len), dtype=int)

        dgl_graph_obj = dgl.DGLGraph()

        dgl_graph_obj.add_nodes(main_ingredients_len)
        dgl_graph_obj.set_n_initializer(dgl.init.zero_initializer)
        dgl_graph_obj.ndata["unit"] = torch.zeros(main_ingredients_len)
        dgl_graph_obj.ndata["dtype"] = torch.zeros(main_ingredients_len)

        dgl_graph_obj.add_nodes(instruction_len)
        dgl_graph_obj.set_e_initializer(dgl.init.zero_initializer)
        dgl_graph_obj.ndata["unit"][main_ingredients_len:] = torch.ones(instruction_len)
        dgl_graph_obj.ndata["dtype"][main_ingredients_len:] = torch.ones(instruction_len)

        m_2_nid = [i for i in range(main_ingredients_len)]
        i_2_nid = [i + main_ingredients_len for i in range(instruction_len)]
        for i in range(instruction_len):
            for j in range(main_ingredients_len):
                tfidf = ins2ing_node_connection[i][j] if (i < instruction_len and j < main_ingredients_len) else 0
                tfidf_box = np.round(tfidf)
                edge_feature_data = {"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])}
                dgl_graph_obj.add_edges(m_2_nid[j], i_2_nid[i], data=copy.deepcopy(edge_feature_data))
                dgl_graph_obj.add_edges(i_2_nid[i], m_2_nid[j], data=copy.deepcopy(edge_feature_data))
            dgl_graph_obj.add_edges(i_2_nid[i], i_2_nid, data={"dtype": torch.ones(instruction_len)})
            dgl_graph_obj.add_edges(i_2_nid, i_2_nid[i], data={"dtype": torch.ones(instruction_len)})

        dgl_graph_obj.nodes[m_2_nid].data["words"] = torch.LongTensor(main_ingredients_token)  # [N, seq_len]
        dgl_graph_obj.nodes[m_2_nid].data["position"] = torch.arange(1, main_ingredients_len + 1).view(-1, 1).long()
        dgl_graph_obj.nodes[i_2_nid].data["words"] = torch.LongTensor(instruction_token)  # [N, seq_len]
        dgl_graph_obj.nodes[i_2_nid].data["position"] = torch.arange(1, instruction_len + 1).view(-1, 1).long()

        return dgl_graph_obj

    @staticmethod
    def replace_char(input_str: str, old_replace_chars: List[str]) -> str:
        for old_char in old_replace_chars:
            input_str = input_str.replace(old_char, " ")
        return input_str

    def get_train(self, index: int):  # TODO
        sample = self.recipe_samples[index]
        title = sample["title"]
        main_ingredients = sample["main_ingredients"] if "main_ingredients" in sample else sample["ingredients"]
        instruction = sample["instruction"] if "instruction" in sample else sample["instructions"]
        ingredients_feature = sample["embedding_ingredients"]
        instruction_feature = sample["embedding_instruction"]
        img_feature = sample["resnet50"]  # TODO other visual feature
        title_feature = sample["embedding_title"]
        ins2ing_node_connection = sample["arr"]  # TODO what?

        graph = self.generate_graph(instruction, main_ingredients, ins2ing_node_connection)

        title_token = [self.get_token(word) for word in self.replace_char(title, self.REMOVE_TITLE_CHARS).split(" ") if
                       len(word)]
        title_token_ids = [self.get_pad(title_token, -1)]

        return {"img_feature": torch.tensor(img_feature),
                "ingredients_feature": torch.tensor(ingredients_feature),
                "instruction_feature": torch.tensor(instruction_feature),
                "title_feature": torch.tensor(title_feature),
                "graph": graph,
                "title": title_token_ids
                }

    def get_token(self, word: str):
        return self.vocab._word_to_id.get(word.lower()) if word.lower() in self.vocab._word_to_id else 1

    @staticmethod
    def get_pad(tokens: List, length: int) -> List:
        return tokens if length < 0 else (tokens + [0] * length)[:length]

    get_validate = get_train


class RecipeDataModule(LightningDataModule):
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

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        is_stage: Callable = lambda stage_, running_stage: which_one_running_state(stage_) is running_stage.value
        if not self.data_train and self.hparams.dataset.train_cfg and is_stage(stage, RunningStage.TRAINING):
            kwargs = self.hparams.dataset.train_cfg
            kwargs.running_stage = RunningStage.TRAINING.value
            self.data_train = RecipeDataset(**kwargs)

        if not self.data_val and self.hparams.dataset.val_cfg and stage in [RunningStage.VALIDATING,
                                                                            RunningStage.FITTING]:
            kwargs = self.hparams.dataset.val_cfg
            kwargs.running_stage = RunningStage.VALIDATING.value
            self.data_val = RecipeDataset(**kwargs)

        if not self.data_test and self.hparams.dataset.get("test_cfg", False) and is_stage(stage, RunningStage.TESTING):
            kwargs = self.hparams.dataset.test_cfg
            kwargs.running_stage = RunningStage.TESTING.value
            self.data_test = RecipeDataset(**kwargs)

    def train_dataloader(self):
        kwargs = self.hparams.dataloader
        kwargs.dataset = self.data_train
        kwargs.shuffle = kwargs.get("shuffle", True)

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
