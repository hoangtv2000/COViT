import numpy as np
import os
import torch
from torch.utils import model_zoo
from .pvt_v2 import PyramidVisionTransformerV2
from .vit import ViT
from torch import nn

class COVID_PVTv2(PyramidVisionTransformerV2):
    def __init__(self):
        super().__init__()
        self.pvt = PyramidVisionTransformerV2()
        self.classifier = nn.Sequential(nn.Linear(1000, 16), nn.GELU(), nn.Dropout(0.2) , nn.Linear(16, 3))

        self.pvt.freeze_patch_emb()

        for p in self.pvt.block1.parameters():
            p.requires_grad == False

        for p in self.pvt.block2.parameters():
            p.requires_grad == False

    def forward(self, x):
        out = self.pvt(x)
        return self.classifier(out)


class COVID_ViT(ViT):
    def __init__(self):
        super().__init__()
        self.vit = ViT()
        self.classifier = nn.Sequential(nn.LayerNorm(21843), nn.Linear(21843, 16), nn.GELU(), nn.Dropout(0.2), nn.Linear(16, 3))

    def forward(self, x):
        out = self.vit(x)
        return self.classifier(out)



def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string .pth) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint_loader = torch.load(checkpoint, map_location='cpu')
    print('Checkpoint dist contains: ',checkpoint_loader.keys())
    model.load_state_dict(checkpoint_loader['state_dict'], strict=False)

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    if optimizer != None:
        optimizer.load_state_dict(checkpoint_loader['optimizer'])

    if scheduler != None:
        scheduler.load_state_dict(checkpoint_loader['scheduler'])

    epoch = checkpoint_loader['epoch']

    return model, optimizer, scheduler, epoch
