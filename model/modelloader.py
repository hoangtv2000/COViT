import numpy as np
import os
import torch
from .pvt_v2 import PyramidVisionTransformerV2
from torch import nn
from functools import partial
from torchvision import models


class COVID_PVTv2(nn.Module):

    def __init__(self, config):
        super(COVID_PVTv2, self).__init__()
        self.config = config
        self.pvt = PyramidVisionTransformerV2()
        model_path = self.config.pretrained_dir + self.config.model_PVT_V2.pretrained_cpkt
        if os.path.exists(model_path):
            if(self.config.model_PVT_V2.pretrained_cpkt == '/pvt_v2_b2_li.pth'):
                self.pvt = PyramidVisionTransformerV2()
                self.embed_dim = 512
            elif(self.config.model_PVT_V2.pretrained_cpkt == '/pvt_v2_b0.pth'):
                self.pvt = PyramidVisionTransformerV2(embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
                self.embed_dim = 256
            else:
                raise ValueError('Pretrained model checkpoint: {} does not exist.'.format(self.config.model_PVT_V2.pretrained_cpkt))

        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, 64), nn.GELU(), nn.BatchNorm1d(64),\
                            nn.Dropout(0.5),\
                            nn.Linear(64, 16),nn.GELU(), nn.BatchNorm1d(16),\
                            nn.Dropout(0.3),\
                            nn.Linear(16, self.config.dataset.num_classes))

        if self.config.load_for_training == False and self.config.ImgNet_pretrained == True:
            model_checkpoint = torch.load(model_path)
            model_statedict = self.pvt.state_dict()

            for k, v in model_checkpoint.items():
                if (k not in model_statedict.copy().keys())\
                        or model_statedict[k].shape != model_checkpoint[k].shape:
                    print(f'! Failed to load {k}')
                    continue

                if [(f'block{i+1}' in k) or (f'patch_embed{i+1}' in k) or (f'norm{i+1}' in k)\
                        for i in range(self.config.block_collect)]:
                    model_statedict.update({k: v})
                else:
                    continue

            self.pvt.load_state_dict(model_statedict)


    def freeze_layers(self):
        self.pvt.freeze_patch_emb()
        print(self.config.freeze_block)
        if self.config.freeze_block == 1:
            for p in self.pvt.block1.parameters():
                p.requires_grad = False
        elif self.config.freeze_block == 2:
            for p in self.pvt.block1.parameters():
                p.requires_grad = False
            for p in self.pvt.block2.parameters():
                p.requires_grad = False
        elif self.config.freeze_block == 3:
            for p in self.pvt.block1.parameters():
                p.requires_grad = False
            for p in self.pvt.block2.parameters():
                p.requires_grad = False
            for p in self.pvt.block3.parameters():
                p.requires_grad = False
        else:
            raise ValueError('You should config "freeze_block" as 0, 1 or 2!')

    def forward(self, x):
        x = self.pvt(x)
        x = self.classifier(x)

        return x




class COVID_ViT(nn.Module):

    def __init__(self, config):
        super(COVID_ViT, self).__init__()
        self.embed_dim = 1000
        self.config = config

        self.vit = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=False)

        embed_dim = getattr(self.vit, 'embed_dim')

        self.classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.BatchNorm1d(64),\
                            nn.Dropout(0.5),\
                            nn.Linear(64, 16),nn.GELU(), nn.BatchNorm1d(16),\
                            nn.Dropout(0.3),\
                            nn.Linear(16, self.config.dataset.num_classes))

        if self.config.load_for_training == False and self.config.ImgNet_pretrained == True:
            model = torch.hub.load('facebookresearch/deit:main',
                               'deit_tiny_patch16_224', pretrained=True)

            model_checkpoint = model.state_dict()
            my_statedict = self.vit.state_dict()

            for k, v in model_checkpoint.items():
                if('patch_embed'in k) or ('norm' in k):
                    my_statedict.update({k: v})
                if [f'.{i}.' in k for i in range(self.config.block_collect)]:
                    my_statedict.update({k: v})
                else:
                    continue
            self.vit.load_state_dict(my_statedict)

    def freeze_layers(self):
        self.vit.freeze_patch_emb()
        for i in range(self.config.freeze_block):
            for p in self.vit.blocks[i].parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.vit(x)
        return self.classifier(out)





def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    """
    checkpoint_loader = torch.load(checkpoint, map_location='cpu')
    print('Checkpoint dist contains: ',checkpoint_loader.keys())

    my_state_dict = model.state_dict()
    for k, v in checkpoint_loader['state_dict'].items():
        k = k.replace('module.', '')
        my_state_dict.update({k: v})

    model.load_state_dict(my_state_dict, strict=True)

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    if optimizer != None:
        optimizer.load_state_dict(checkpoint_loader['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = 0.0001

    if scheduler != None:
        scheduler.load_state_dict(checkpoint_loader['scheduler'])

    epoch = checkpoint_loader['epoch']

    return model, optimizer, scheduler, epoch
