import os
import torch
import torch.optim as optim
from functools import partial
import random
import numpy as np
import pdb

from configs.config import Training_Config as training_cfg
from data.keypoints_dataset import Keypoints_Paths_Dataset
from data.data_setup import create_dataloader, dataset_collate_fn
from models.stgcn import ST_GCN_18
from losses.triplet_loss import Triplet_Loss
import engine

from models.linear import TemporalSimpleModel

def set_seeds(seed: int=100):
    random.seed(seed)
    np.random.seed(seed)
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def main():

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.name == 'posix':
        root = os.path.abspath(__file__).split("embs_model/encoder")[0]
    elif os.name == 'nt':
        root = os.path.abspath(__file__).split("embs_model\\encoder")[0]

    if training_cfg.kpts_type_for_train == "mediapipe":
        train_dataset = Keypoints_Paths_Dataset(os.path.join(root,"datasets",f"mediapipe_train_file_paths_3d_world.json"))
        test_dataset = Keypoints_Paths_Dataset(os.path.join(root,"datasets",f"mediapipe_test_file_paths_3d_world.json"))
    else:
        train_dataset = Keypoints_Paths_Dataset(os.path.join(root,"datasets",f"motionbert_train_file_paths_3d.json"))
        test_dataset = Keypoints_Paths_Dataset(os.path.join(root,"datasets",f"motionbert_test_file_paths_3d.json")) 

    # cnt = 0
    # for i in range(24):
    #     idx = f"{i:02}"
    #     print(idx, len(train_dataset.form_3d_file_paths_dict[idx])+len(test_dataset.form_3d_file_paths_dict[idx]))
    #     cnt += len(train_dataset.form_3d_file_paths_dict[idx])
    #     cnt += len(test_dataset.form_3d_file_paths_dict[idx])
    # print(cnt)
    # breakpoint()

    train_collate_fn = partial(dataset_collate_fn, form_file_paths_dict=train_dataset.form_3d_file_paths_dict, cfg=training_cfg, dataset_type="train")
    test_collate_fn = partial(dataset_collate_fn, form_file_paths_dict=test_dataset.form_3d_file_paths_dict, cfg=training_cfg, dataset_type="test")

    train_dataloader = create_dataloader(train_dataset, training_cfg.batch_size, train_collate_fn, num_workers=training_cfg.num_workers, is_training=True)
    test_dataloader = create_dataloader(test_dataset, training_cfg.batch_size, test_collate_fn, num_workers=training_cfg.num_workers, is_training=False)


    if training_cfg.only_for_sampling_demo:
        loader_iter = iter(train_dataloader)
        while True:
            # torch.Size([final_batch_size, apn, clip_len, kpts_size, channel_size])
            _ = next(loader_iter)
            breakpoint()

    if training_cfg.kpts_type_for_train == "motionbert":
        keypoint_size = 17
        input_channel_size = len(training_cfg.data_type_views)*3 + 1 if "angle" not in training_cfg.data_type_views else (len(training_cfg.data_type_views)-1)*3 + 2
    elif training_cfg.kpts_type_for_train == "vitpose":
        keypoint_size = 35
        iinput_channel_size = len(training_cfg.data_type_views)*2 + 1 if "angle" not in training_cfg.data_type_views else (len(training_cfg.data_type_views)-1)*2 + 2
    elif training_cfg.kpts_type_for_train == "mediapipe":
        keypoint_size = 29
        input_channel_size = len(training_cfg.data_type_views)*3 + 1 if "angle" not in training_cfg.data_type_views else (len(training_cfg.data_type_views)-1)*3 + 2

    graph_cfg = {'layout': training_cfg.kpts_type_for_train, 'strategy': 'spatial', 'max_hop': 1}
    stgcn = ST_GCN_18(in_channels=input_channel_size, num_class=24, graph_cfg=graph_cfg, edge_importance_weighting=True, dropout=0.5, clip_len=training_cfg.clip_len).to(device)
    # stgcn = TemporalSimpleModel(in_dim=keypoint_size*input_channel_size, clip_len=training_cfg.clip_len).to(device)

    # # Freeze some layers
    # for param in stgcn.data_bn.parameters():
    #     param.requires_grad = False

    # for layer_id in range(8):
    #     for param in stgcn.st_gcn_networks[layer_id].parameters():
    #         param.requires_grad = False
    #     for param in stgcn.edge_importance[layer_id].parameters():
    #         param.requires_grad = False

    stgcn_trainable_parameters = list(filter(lambda p: p.requires_grad, stgcn.parameters()))
    optimizer = optim.SGD(stgcn_trainable_parameters, lr=training_cfg.learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    loss_fn = Triplet_Loss(training_cfg).to(device)

    set_seeds()
    engine.train(stgcn, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, training_cfg, device)


# The implementation of multiprocessing is different on Windows, which uses spawn instead of fork. 
# So we have to wrap the code with an if-clause to protect the code from executing multiple times. 
# https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
if __name__=='__main__':
    main()
    

