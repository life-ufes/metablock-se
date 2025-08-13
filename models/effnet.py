# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com
"""

import torch
from torch import nn
from models.metablock import MetaBlock
from models.cross_attention import CrossAttention
import warnings


class MyEffnet (nn.Module):

    def __init__(self, efnet, num_class, neurons_reducer_block=256, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=1792):

        super(MyEffnet, self).__init__()

        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if isinstance(comb_config, int):
                    self.comb_feat_maps = 32
                    self.comb = MetaBlock(self.comb_feat_maps, comb_config)
                elif isinstance(comb_config, list):
                    self.comb_feat_maps = comb_config[0]
                    self.comb = MetaBlock(self.comb_feat_maps, comb_config[1])
                else:
                    raise Exception(
                        "comb_config must be a list or int to define the number of feat maps and the metadata")
            elif comb_method == 'cross-attention':
                self.comb = CrossAttention(comb_config[1], n_feat_conv)
            else:
                raise Exception("There is no comb_method called " + comb_method + ". Please, check this out.")
        else:
            self.comb = None

        self.efnet = efnet
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.LazyLinear(num_class)


    def forward(self, img, meta_data=None):

        # Checking if when passing the metadata, the combination method is set
        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        x = self.avg_pooling(self.efnet.extract_features(img))
        if self.comb == None:
            x = x.view(x.size(0), -1)  # flatting
        elif isinstance(self.comb, CrossAttention):
            x = x.view(x.size(0), -1)
            x = self.comb(x, meta_data.float())
        elif isinstance(self.comb, MetaBlock):
            x = x.view(x.size(0), self.comb_feat_maps, 32, -1).squeeze(-1)  # getting the feature maps
            x = self.comb(x, meta_data.float())  # applying metablock
            x = x.view(x.size(0), -1)  # flatting

        return self.classifier(x)
