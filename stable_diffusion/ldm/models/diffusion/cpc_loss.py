from typing import List, Dict, Callable

import torch
from torch import nn
import pytorch_lightning as pl


class CPCLoss(pl.LightningModule):
    def __init__(self, method: str = "mean"):
        super(CPCLoss, self).__init__()
        self.similarity_fn: Callable = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.method = method

    def forward(self,
                predict: torch.Tensor,
                positive_sample: torch.Tensor,
                negative_sample: torch.Tensor) -> torch.Tensor:
        positive_dist = 1 - self.similarity_fn(predict, positive_sample)
        negative_dist = 1 - self.similarity_fn(predict, negative_sample)
        return self.calculate_loss(positive_dist, negative_dist)

    def calculate_loss(self, positive_dist, negative_dist) -> torch.Tensor:
        loss_zero = torch.tensor(0,dtype=positive_dist.dtype,device=positive_dist.device)
        if self.method == "mean":
            positive_mean = positive_dist.mean()
            negative_mean = negative_dist.mean()

            return max(positive_mean - negative_mean, loss_zero)
        elif self.method == "margin":
            positive_max = positive_dist.max()
            negative_min = negative_dist.min()

            return max(positive_max - negative_min, loss_zero)



if __name__ == '__main__':
    cpc_loss_obj: Callable = CPCLoss()
    causal_file = "/home/datasets/VQAI/VQAI_v6/VQAI-causal-feature-xxl.pt"
    data = torch.load(causal_file)

    predict = data[0]["causal_eos_token_feature"]
    positive_sample = data[1]["causal_eos_token_feature"]
    negative_sample = data[97]["causal_eos_token_feature"]

    rst = cpc_loss_obj(predict, positive_sample, negative_sample)
    a = 1
