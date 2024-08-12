import os
import torch
import torch.optim as optim
from functools import partial
import random
import numpy as np
import pdb

from configs.test_config import Training_Config as training_cfg
from data.keypoints_dataset import Keypoints_Paths_Dataset
from data.data_setup_for_test import create_dataloader, dataset_collate_fn
from models.stgcn import ST_GCN_18
from losses.triplet_loss import Triplet_Loss
import engine_for_test as engine

from models.cv_mim import TemporalSimpleModel
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def set_seeds(seed: int=100):
    random.seed(seed)
    np.random.seed(seed)
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

if training_cfg.kpts_type_for_train == "motionbert":
    keypoint_size = 17
    if training_cfg.use_2d_kpts:
        input_channel_size = len(training_cfg.data_type_views)*2 + 1 if "angle" not in training_cfg.data_type_views else (len(training_cfg.data_type_views)-1)*2 + 2
    else:
        input_channel_size = len(training_cfg.data_type_views)*3 + 1 if "angle" not in training_cfg.data_type_views else (len(training_cfg.data_type_views)-1)*3 + 2
elif training_cfg.kpts_type_for_train == "vitpose":
    keypoint_size = 35
    iinput_channel_size = len(training_cfg.data_type_views)*2 + 1 if "angle" not in training_cfg.data_type_views else (len(training_cfg.data_type_views)-1)*2 + 2
elif training_cfg.kpts_type_for_train == "mediapipe":
    keypoint_size = 29
    input_channel_size = len(training_cfg.data_type_views)*3 + 1 if "angle" not in training_cfg.data_type_views else (len(training_cfg.data_type_views)-1)*3 + 2



graph_cfg = {'layout': training_cfg.kpts_type_for_train, 'strategy': 'spatial', 'max_hop': 1}
stgcn = ST_GCN_18(in_channels=input_channel_size, num_class=24, graph_cfg=graph_cfg, edge_importance_weighting=True, dropout=0.5, clip_len=training_cfg.clip_len)
#stgcn = TemporalSimpleModel(in_dim=keypoint_size*input_channel_size, clip_len=training_cfg.clip_len)

set_seeds()

tensor = (torch.rand(2, 1, 10, keypoint_size, input_channel_size),)
flops = FlopCountAnalysis(stgcn, tensor)
print("FLOPs: ", flops.total())
print(parameter_count_table(stgcn))