import torch
from torch import nn
import torch.nn.functional as F

class LinearLayers(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, dropout_rate=0.25):
        super().__init__()
        self.linear_layers = nn.Sequential()
        for idx in range(num_layers):
            self.linear_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=True))
            self.linear_layers.append(nn.BatchNorm1d(num_features=out_dim))
            self.linear_layers.append(nn.ReLU(inplace=False))
            self.linear_layers.append(nn.Dropout(p=dropout_rate, inplace=False))
            in_dim = out_dim
        self.apply(self._init_weights)

    def forward(self, x):
        output = self.linear_layers(x)
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)



class ResLinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, dropout_rate):
        super().__init__()
        self.linear_layers = LinearLayers(in_dim, out_dim, num_layers, dropout_rate)

    def forward(self, x):
        output = self.linear_layers(x) + x
        return output

class PointEmbedder(nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.embedder = embedding_layer

    def forward(self, x):
        return self.embedder(x)


class SimpleModel(nn.Module):
    def __init__(self, in_dim, out_dim, embedder="point", hidden_dim=1024, num_residual_linear_blocks=2, num_layers_per_resblock=2, dropout_rate=0.25):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.blocks.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        self.blocks.add_module('fc0', LinearLayers(in_dim, hidden_dim, 1, dropout_rate))
        
        for idx in range(num_residual_linear_blocks):
            self.blocks.add_module(f'res{idx:01}',ResLinearBlock(hidden_dim, hidden_dim, num_layers_per_resblock, dropout_rate))

        if embedder == "point":
            self.blocks.add_module('point',PointEmbedder(nn.Linear(hidden_dim, out_dim, bias=True)))

    def forward(self, x):
        fc0_result = None
        for block_name, block in self.blocks.items():
            x = block(x)
            if(block_name=='fc0'):
                fc0_result = x
        return x, fc0_result




class TemporalSimpleModel(nn.Module):
    def __init__(self, in_dim, out_dim=256, embedder="point", hidden_dim=1024, clip_len=10, num_residual_linear_blocks=2, num_layers_per_resblock=2, dropout_rate=0.5):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.blocks.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
        self.blocks.add_module('fc0', LinearLayers(in_dim, hidden_dim, 1, dropout_rate))
        
        for idx in range(num_residual_linear_blocks):
            self.blocks.add_module(f'res{idx:01}',ResLinearBlock(hidden_dim, hidden_dim, num_layers_per_resblock, dropout_rate))

        self.blocks.add_module('LateFuseProject', LinearLayers(clip_len*hidden_dim, hidden_dim, 1, dropout_rate))
        self.blocks.add_module('LateFuseResBlock',ResLinearBlock(hidden_dim, hidden_dim, num_layers_per_resblock, dropout_rate))
        
        if embedder == "point":
            self.blocks.add_module('point',PointEmbedder(nn.Linear(hidden_dim, out_dim, bias=True)))
    

    def forward(self, x):
        N, T, K, C = x.shape
        x =  x.reshape(N*T, K, C)
        for block_name, block in self.blocks.items():
            if(block_name=='LateFuseProject'):
                x = x.reshape(N,-1)
            x = block(x)
        x = F.normalize(x, dim=1)
        return x


