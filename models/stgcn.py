
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import Graph
from .gconv import ConvTemporalGraphical
import math

def zero(x):
    return 0


def iden(x):
    return x


class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 clip_len=10,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        # Add a buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter. 
        # For example, BatchNorm’s running_mean is not a parameter, but is part of the module’s state. 
        # Buffers, by default, are persistent and will be saved alongside parameters.
        self.register_buffer('A', A)
        # I will be saved in the model's state_dict, but will not be updated
        # use self.register_buffer('A', A) in a class inheriting from nn.Module, you can access the registered buffer A via self.A
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0) # the number of A
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # in_channels*num_nodes
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        
        # kwargs excluding dropout
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}


        self.st_gcn_networks = nn.ModuleList((
            # The 0th layer has no dropout or residual
            # While the subsequent layers do
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            # Layer 1
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            # Layer 2
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            # Layer 3
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            # Layer 4, TCN stride is 2
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            # Layer 5
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            # Layer 6
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            # Layer 7, TCN stride is 2
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            # Layer 8
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            # Layer 9
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            # Each layer has an edge_importance_weighting
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        # Assuming we maintain (final_batch_size, apn, clip_len, kpts_size, channel_size)
        # Here, we treat M as apn (although originally, M should refer to the number of people)
        N, M, T, V, C = x.size()
        # T must be at least 10
        if T < 10:
            # Calculate the number of repetitions needed
            repeat_count = (10 + T - 1) // T  # Round up
            # Repeat the tensor
            x = x.repeat(1, 1, repeat_count, 1, 1)
            # Trim to the first 10 T dimensions
            x = x[:, :, :10, :, :]
        
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, V * C, T)
        # For data_bn, the batch_size is final_batch_size*apn
        # Data features are kpts_size*channel_size, clip_len is sequence length
        # nn.BatchNorm1d normalizes along the num_features dimension. Specifically, it normalizes each feature to have a mean of 0 and a standard deviation of 1 across the batch.
        # In other words, for a single feature, the mean of that feature across the entire batch is 0, and the standard deviation is 1
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # Last two dimensions: time and space axis form 2D information
        x = x.view(N * M, C, T, V)

        # forward
        # pair corresponding elements from two lists into a tuple
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        # Perform avg_pool2d on T and V
        x = F.avg_pool2d(x, x.size()[2:])
        
        # This normalization is added by me, so the norm of 256 dimensions will be 1
        x = F.normalize(x, dim=1)

        # here M is apn, so it should be split later
        x = x.view(N, M, -1, 1, 1)

        return x
        


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        # (time, space)
        assert len(kernel_size) == 2
        # the kernel for the time axis must be odd
        assert kernel_size[0] % 2 == 1
        # If you want the size to remain unchanged after convolution, padding should be (kernel_size*dilation - dilation) // 2
        # the dilation for the time axis set to 1 by default (?
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Try leakyReLU
            #nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        # Try leakyReLU
        #self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A