import torch
import torch.nn as nn


class CharbonnierL1Loss(nn.Module):
    def __init__(self):
        super(CharbonnierL1Loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
