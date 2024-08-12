import torch
import numpy as np
import os
import json
import math
from torch.utils.data import Dataset, DataLoader
from motionbert.utils.utils_data import crop_scale

def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y

def cocowhole2h36m(x):
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    # 0: hip
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    # 1: right hip
    y[:,1,:] = x[:,12,:]
    # 2: right knee
    y[:,2,:] = x[:,14,:]
    # 3: right foot
    y[:,3,:] = x[:,16,:]
    # 4: left hip
    y[:,4,:] = x[:,11,:]
    # 5: left knee
    y[:,5,:] = x[:,13,:]
    # 6: left foot
    y[:,6,:] = x[:,15,:]
    # 8: thorax
    y[:,8,:] = (x[:,5,:] + x[:,6,:]) * 0.5
    # 10: head
    y[:,10,:] = (x[:,3,:] + x[:,4,:]) * 0.5
    # 9: neck
    y[:,9,:] = (y[:,8,:] + y[:,10,:]) * 0.5
    # 7: spine
    y[:,7,:] = (y[:,8,:] + y[:,0,:]) * 0.5
    # 11: left shoulder
    y[:,11,:] = x[:,5,:]
    # 12: left elbow
    y[:,12,:] = x[:,7,:]
    # 13: left wrist
    y[:,13,:] = x[:,9,:]
    # 14: right shoulder
    y[:,14,:] = x[:,6,:]
    # 15: right elbow
    y[:,15,:] = x[:,8,:]
    # 16: right wrist
    y[:,16,:] = x[:,10,:]
    return y


def mediapipe2h36m(x):
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    # 0: hip
    y[:,0,:] = (x[:,23,:] + x[:,24,:]) * 0.5
    # 1: right hip
    y[:,1,:] = x[:,24,:]
    # 2: right knee
    y[:,2,:] = x[:,26,:]
    # 3: right foot
    y[:,3,:] = x[:,28,:]
    # 4: left hip
    y[:,4,:] = x[:,23,:]
    # 5: left knee
    y[:,5,:] = x[:,25,:]
    # 6: left foot
    y[:,6,:] = x[:,27,:]
    # 8: thorax
    y[:,8,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    # 10: head
    y[:,10,:] = (x[:,7,:] + x[:,8,:]) * 0.5
    # 9: neck
    y[:,9,:] = (x[:,8,:] + y[:,10,:]) * 0.5
    # 7: spine
    y[:,7,:] = (y[:,8,:] + y[:,0,:]) * 0.5
    # 11: left shoulder
    y[:,11,:] = x[:,11,:]
    # 12: left elbow
    y[:,12,:] = x[:,13,:]
    # 13: left wrist
    y[:,13,:] = x[:,15,:]
    # 14: right shoulder
    y[:,14,:] = x[:,12,:]
    # 15: right elbow
    y[:,15,:] = x[:,14,:]
    # 16: right wrist
    y[:,16,:] = x[:,16,:]
    return y


# 我加的
def read_keypoints(kpts_all, vid_size, scale_range, focus, data_type="vitpose"):
    kpts_all = np.array(kpts_all)
    if data_type == "vitpose":
        kpts_all = cocowhole2h36m(kpts_all)
    elif data_type == "mediapipe":
        kpts_all = mediapipe2h36m(kpts_all)

    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all

    if scale_range:
        motion = crop_scale(kpts_all, scale_range)

    return motion.astype(np.float32)


def read_input(json_path, vid_size, scale_range, focus):

    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []

    for item in results:
        kpts = np.array(item['keypoints'])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)
    kpts_all = cocowhole2h36m(kpts_all)


    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, keypoints, clip_len=243, vid_size=None, scale_range=None, focus=None, data_type="vitpose"):
        self.clip_len = clip_len
        self.vid_all = read_keypoints(keypoints, vid_size, scale_range, focus, data_type)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]