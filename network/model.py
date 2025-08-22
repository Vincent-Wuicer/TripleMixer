# Copyright 2024 - xiongwei zhao @ grandzhaoxw@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from .mixer import Mixer
from .geometry import GeoMixer


class Triplemixer(nn.Module):
    def __init__(
        self,
        input_channels,
        feat_channels,
        nb_class,
        depth,      
        grid_shape,
        drop_path_prob=0, 
        layer_norm=False,      
    ):
        super().__init__()
        # GeoMixer layer
        self.embed = GeoMixer(input_channels, feat_channels)
        # backbone
        self.wavelet = Mixer(feat_channels, depth, grid_shape, drop_path_prob, layer_norm)
        # Classification layer
        self.classif = nn.Conv1d(feat_channels, nb_class, 1)

    def forward(self, feats, cell_ind, occupied_cell, neighbors):

        tokens = self.embed(feats, neighbors)
        tokens = self.wavelet(tokens, cell_ind, occupied_cell)
    
        return self.classif(tokens)


