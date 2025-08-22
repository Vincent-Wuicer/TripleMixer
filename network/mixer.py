# Copyright 2024 - xiongwei zhao @ grandzhaoxw@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import autocast
from .wavelet import LiftingScheme2D


class myLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


def build_proj_matrix(
    indices_non_zeros, occupied_cell, batch_size, num_2d_cells, inflate_ind, channels
):
    num_points = indices_non_zeros.shape[1] // batch_size
    
    matrix_shape = (batch_size, num_2d_cells, num_points)

    inflate = torch.sparse_coo_tensor(
        indices_non_zeros, occupied_cell.reshape(-1), matrix_shape
    ).transpose(1, 2)
    
    inflate_ind = inflate_ind.unsqueeze(1).expand(-1, channels, -1)
    
    with autocast("cuda", enabled=False):
        num_points_per_cells = torch.bmm(
            inflate, torch.bmm(inflate.transpose(1, 2), occupied_cell.unsqueeze(-1))
        )
        
    weight_per_point = 1.0 / (num_points_per_cells.reshape(-1) + 1e-6)
    weight_per_point *= occupied_cell.reshape(-1)
    flatten = torch.sparse_coo_tensor(indices_non_zeros, weight_per_point, matrix_shape)

    return {"flatten": flatten, "inflate": inflate_ind}


class DropPath(nn.Module):
    """
    Stochastic Depth

    Original code of this module is at:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def extra_repr(self):
        return f"prob={self.drop_prob}"

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output



#Channel
class ChaMixer(nn.Module):
    def __init__(self, channels, drop_path_prob, layer_norm=False):
        super().__init__()
        self.compressed = False
        self.layer_norm = layer_norm
        if layer_norm:
           self.norm = myLayerNorm(channels)
        else:
           self.norm = nn.BatchNorm1d(channels)
        
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  
        
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, tokens):
        """tokens <- tokens + LayerScale( MLP( BN(tokens) ) )"""
        if self.compressed:
            assert not self.training
            return tokens + self.drop_path(self.mlp(tokens))
        else:
            return tokens + self.drop_path(self.scale(self.mlp(self.norm(tokens))))


## Frequency
class FreMixer(nn.Module):
    def __init__(self, channels, grid_shape, drop_path_prob, layer_norm=False):
        super().__init__()
        self.compressed = False
        self.H, self.W = grid_shape
        self.num_levels = 2
        self.mult = 2
        self.dropout = 0.5
        if layer_norm:
            self.norm = myLayerNorm(channels)
        else:
            self.norm = nn.BatchNorm1d(channels)
            
        self.reduction = nn.Conv2d(channels, int(channels/4), 1)
        
        self.wavelet1 = LiftingScheme2D(in_planes=int(channels/4), share_weights=True)
        
        self.wavelet2 = LiftingScheme2D(in_planes=int(channels/4), share_weights=True)
        
        self.feedforward2 = nn.Sequential(
                nn.Conv2d(channels, channels,1),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Conv2d(channels, channels, 1),
                nn.ConvTranspose2d(channels, int(channels/2), 4, stride=2, padding=1),
                nn.BatchNorm2d(int(channels/2))
            )
        
        self.feedforward1 = nn.Sequential(
                nn.Conv2d(channels + int(channels/2), channels,1),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Conv2d(channels, channels, 1),
                nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels)
            )   
        
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )   
        
        self.grid_shape = grid_shape
        self.drop_path = DropPath(drop_path_prob)

    def extra_repr(self):
        return f"(grid): [{self.grid_shape[0]}, {self.grid_shape[1]}]"


    def forward(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        with autocast("cuda", enabled=False):
            residual = torch.bmm(
                sp_mat["flatten"], residual.transpose(1, 2).float()
            ).transpose(1, 2)
        residual = residual.reshape(B, C, self.H, self.W)
        
        residual_re = self.reduction(residual)
        _, _, LL1, LH1, HL1, HH1 = self.wavelet1(residual_re)
        _, _, LL2, LH2, HL2, HH2 = self.wavelet2(LL1)
        
        x2_wavel=torch.cat([LL2,LH2,HL2,HH2],1)
        residual_wavel2 = self.feedforward2(x2_wavel)
        
        x1_wavel=torch.cat([LL1,LH1,HL1,HH1],1) 
        x1_wavel = torch.cat((x1_wavel,residual_wavel2), 1)
        residual_wavel1 = self.feedforward1(x1_wavel)
        
        residual = residual + residual_wavel1
        
        # FFN
        residual = self.ffn(residual)
        # LayerScale
        residual = residual.reshape(B, C, self.H * self.W)
        residual = self.scale(residual)
        # Inflate
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)



class Mixer(nn.Module):
    def __init__(self, channels, depth, grids_shape, drop_path_prob, layer_norm=False):
        super().__init__()
        self.depth = depth
        self.grids_shape = grids_shape
        self.channel_mix = nn.ModuleList(
            [ChaMixer(channels, drop_path_prob, layer_norm) for _ in range(depth)]
        )
        self.spatial_mix = nn.ModuleList(
            [
                FreMixer(channels, grids_shape[d % len(grids_shape)], drop_path_prob, layer_norm)
                for d in range(depth)
            ]
        )

    def forward(self, tokens, cell_ind, occupied_cell):
        batch_size, num_points = tokens.shape[0], tokens.shape[-1]
        
        point_ind = (
            torch.arange(num_points, device=tokens.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(1, -1)
        )
        
        batch_ind = (
            torch.arange(batch_size, device=tokens.device)
            .unsqueeze(1)
            .expand(-1, num_points)
            .reshape(1, -1)
        )
        
        non_zeros_ind = []
        for i in range(cell_ind.shape[1]):
            non_zeros_ind.append(
                torch.cat((batch_ind, cell_ind[:, i].reshape(1, -1), point_ind), axis=0)
            )
            
        sp_mat = [
            build_proj_matrix(
                id,
                occupied_cell,
                batch_size,
                np.prod(sh),
                cell_ind[:, i],
                tokens.shape[1],
            )
            for i, (id, sh) in enumerate(zip(non_zeros_ind, self.grids_shape))
        ]
        
        for d, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens = smix(tokens, sp_mat[d % len(sp_mat)])
            tokens = cmix(tokens)
        return tokens
