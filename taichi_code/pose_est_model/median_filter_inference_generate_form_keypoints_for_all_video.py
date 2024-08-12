import argparse
import os
import time

import cv2
import numpy as np
import torch

from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader
from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.visualization import joints_dict

import pdb
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# motionbert 
import torch.nn as nn
from torch.utils.data import DataLoader
from motionbert.utils.tools import *
from motionbert.utils.learning import *
from motionbert.data.dataset_wild import WildDetDataset
from motionbert.utils.utils_data import flip_data
from motionbert.utils.vismo import my_render_fn
from tqdm import tqdm
import matplotlib.pyplot as plt


class OneEuroFilter:

    def __init__(self, t0, input0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.input_prev = input0
        self.dinput_prev = np.zeros_like(input0)
        self.t_prev = float(t0)
        self.threshold = 0.5

    def __call__(self, t, input):
        
        """Compute the filtered signal."""
        # (period: how many seconds per sample)
        Te = t - self.t_prev + 1e-7
        
        # alpha_d is a smoothing parameter computed based on Te and the cutoff derivative
        alpha_d = self.smoothing_factor(Te, self.d_cutoff)
        # The rate of change of the input compared to the previous value (i.e., the derivative)
        dinput = (input - self.input_prev) / Te

        # exponential smoothing is a low-pass filter
        # The filtered derivative of the signal.
        # The hat symbol indicates it has been filtered
        dinput_hat = self.exponential_smoothing(alpha_d, dinput, self.dinput_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dinput_hat) + 1e-7
        alpha = self.smoothing_factor(Te, cutoff)
        # The filtered signal.
        input_hat = self.exponential_smoothing(alpha, input, self.input_prev)

        # Memorize the previous values.
        self.input_prev = input_hat
        
        self.dinput_prev = dinput_hat

        self.t_prev = t

        return input_hat
    

    def smoothing_factor(self, Te, cutoff):
        tau = 1. / (2*np.pi*cutoff)
        alpha = 1. / (1+tau/Te)
        return alpha
    
    def exponential_smoothing(self, alpha, input, input_prev):
        return alpha * input + (1 - alpha) * input_prev

    def landmark2ndarray(self, landmarks):
        pose_landmarks_ndarray = np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility,landmark.presence] for landmark in landmarks.pose_landmarks[0]])
        pose_world_landmarks_ndarray = np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility,landmark.presence] for landmark in landmarks.pose_world_landmarks[0]])
        # shape (2,33,3)
        return np.stack((pose_landmarks_ndarray,pose_world_landmarks_ndarray))

    def ndarray2landmark(self, filtered_ndarray):
        pose_landmarker_results_filterd = []
        for f in range(filtered_ndarray.shape[0]):
            tmp_pose_landmarks_list = []
            tmp_pose_world_landmarks_list = []
            for idx in range(filtered_ndarray.shape[2]):
                tmp_pose_landmarks_list.append(NormalizedLandmark(x=filtered_ndarray[f,0,idx,0],y=filtered_ndarray[f,0,idx,1],z=filtered_ndarray[f,0,idx,2],visibility=filtered_ndarray[f,0,idx,3], presence=filtered_ndarray[f,0,idx,4]))
                tmp_pose_world_landmarks_list.append(Landmark(x=filtered_ndarray[f,1,idx,0],y=filtered_ndarray[f,1,idx,1],z=filtered_ndarray[f,1,idx,2],visibility=filtered_ndarray[f,1,idx,3], presence=filtered_ndarray[f,1,idx,4]))
            pose_landmarker_results_filterd.append(PoseLandmarkerResult([tmp_pose_landmarks_list], [tmp_pose_world_landmarks_list]))

        return pose_landmarker_results_filterd


def median_or_mean_filter(data, window_size, filter="median"):
    half_window = window_size // 2
    filtered_data = np.copy(data)
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        if filter == "median":
            filtered_data[i] = np.median(data[start:end], axis=0)
        elif filter == "mean":
            filtered_data[i] = np.mean(data[start:end], axis=0)

    return filtered_data

def bidirectional_filter(data, window_size, filter="median"):
    # Apply forward median filtering
    forward_filtered = median_or_mean_filter(data, window_size, filter)
    
    # Reverse the result of the forward filtering
    reversed_data = forward_filtered[::-1]
    
    # Apply median filtering again on the reversed data
    reverse_filtered = median_or_mean_filter(reversed_data, window_size, filter)
    
    # Reverse the result of the backward filtering
    bidirectional_filtered = reverse_filtered[::-1]

    return bidirectional_filtered


def kalman_filter(poses):
    n_timesteps, n_joints, n_dims = poses.shape

    # Initial state is the first observation
    initial_state = poses[0]

    # State transition matrix
    A = np.eye(n_dims)

    # Observation matrix
    H = np.eye(n_dims)

    # Process noise covariance matrix
    Q = np.eye(n_dims) * 1e-3

    # Observation noise covariance matrix
    R = np.eye(n_dims) * 1e-2

    # Initial state covariance
    P = np.eye(n_dims)

    # Store the filtered results
    smoothed_poses = np.zeros((n_timesteps, n_joints, n_dims))

    for t in range(n_timesteps):
        if t == 0:
            state_estimate = initial_state
        else:
            # state prediction
            state_predict = A @ state_estimate.T
            # state_predict shape: (channels_size, kpts_size)
            P_predict = A @ P @ A.T + Q

            # kalman gain
            K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)

            # state update
            state_estimate = state_predict + K @ (poses[t].T - H @ state_predict)
            # state_estimate shape: (channels_size, kpts_size)
            # P = (np.eye(n_dims) - K @ H) @ P_predict
            P = (np.eye(n_dims) - K @ H) @ P_predict @ (np.eye(n_dims) - K @ H).T + (K @ R @ K.T)
            state_estimate = state_estimate.T

        smoothed_poses[t] = state_estimate

    return smoothed_poses


def process_video(input_file=None, output_path=None, vitpose_model=None, motionbert_model=None, args=None, fps_in=10):

    input_path = input_file
    ext = input_path[input_path.rfind('.'):]    

    if output_path:
        if os.path.isdir(output_path):
            output_path_video = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}{ext}"))
            output_path_video_2d = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_2d{ext}"))
            output_path_video_3d = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_3d{ext}"))
            output_path_2d_npz = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_2d.npz"))
            output_path_3d_npz = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_3d.npz"))
        else:
            output_path_video = output_path + f"_fps{fps_in}{ext}"
            output_path_video_2d = output_path + f"_fps{fps_in}_2d{ext}"
            output_path_video_3d = output_path + f"_fps{fps_in}_3d{ext}"
            output_path_2d_npz = output_path + f"_fps{fps_in}_2d.npz"
            output_path_3d_npz = output_path + f"_fps{fps_in}_3d.npz"

    if(os.path.isfile(output_path_2d_npz)):
        print(f'>>> {output_path_2d_npz} existed')
        return

    assert os.path.isfile(input_path), 'The input file does not exist'

    is_video = input_path[input_path.rfind('.')+1 : ].lower() in ['mp4','webm']
    if is_video:
        cap = cv2.VideoCapture(input_path)
        assert cap.isOpened()     
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_size = (frame_width,frame_height)
        
        if args.save_video:
            figsize = (4, 4)
            dpi = 100 
            fig = plt.figure(0, figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=12., azim=80)
            out_writer = cv2.VideoWriter(output_path_video_2d,
                                        #cv2.VideoWriter_fourcc(*'vp09'),
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        fps_in, (frame_width, frame_height))  # type: ignore

            out_writer_3d = cv2.VideoWriter(output_path_video_3d,
                                            #cv2.VideoWriter_fourcc(*'vp09'),
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps_in, (400,400))  # type: ignore
                                            
        keypoints = []
        keypoints_3d = []
        origin_imgs = []
        frame_ith = []
        ith = 0

        print(f'>>> Running inference on {input_path}')

        while cap.grab():
            success, frame = cap.retrieve()

            if not success:
                print("Error !!")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            # /home/charl0tte/easy_ViTPose/easy_ViTPose/inference.py
            frame_keypoints, bbox_empty = vitpose_model.inference(frame_rgb, True)
            if bbox_empty:
                continue 
            
            keypoints.append(frame_keypoints[0])
            frame_ith.append(ith)
            origin_imgs.append(frame_rgb)
            ith += 1
        
        # keypoints shape: (frames_size, kpts_size, channels_size)
        keypoints = np.array(keypoints, dtype=np.float32)
        keypoints[..., :-1] = bidirectional_filter(data=keypoints[...,:-1], window_size=5, filter='median')
        
        if args.save_video:
            for f in range(len(keypoints)):
                keypoints_img = vitpose_model.draw_with_img({0:keypoints[f]}, origin_imgs[f], False, False, 0.0)[..., ::-1]
                out_writer.write(keypoints_img)
        
        if args.save_video:
            print('>>> Saved output 2d video')
            out_writer.release()

        # The original coordinates of keypoints from easy_vitpose are (y, x, confidence)
        # so I changed it to (x, y, confidence)
        np.savez(output_path_2d_npz, frame_ith=np.array(frame_ith, dtype=np.int32), keypoints=keypoints)
        print('>>> Saved keypoints 2d npz')
        
        wild_dataset = WildDetDataset(keypoints, clip_len=243, vid_size=vid_size, scale_range=None,focus=None)

        test_loader = DataLoader(wild_dataset)
        keypoints_3d_with_score = []

        with torch.no_grad():
            for batch_input in test_loader:

                N, T = batch_input.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                
                predicted_3d_pos = motionbert_model(batch_input)

                tmp_kpts_3d_with_score = torch.cat((predicted_3d_pos, batch_input[...,-1:]), dim=-1).cpu().numpy()

                keypoints_3d_with_score.append(tmp_kpts_3d_with_score)
                


        # shape: (batch_size, frame_size, 17, 3)
        keypoints_3d_with_score = np.hstack(keypoints_3d_with_score)
        # shape: (batch_size*frame_size, 17, 3)
        keypoints_3d_with_score = np.concatenate(keypoints_3d_with_score)
        keypoints_3d_with_score[..., :-1] = bidirectional_filter(data=keypoints_3d_with_score[...,:-1], window_size=10, filter='median')
        
        if args.save_video:
            figsize = (4, 4)
            dpi = 100 
            fig = plt.figure(0, figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=12., azim=80)
            output_imgs_3d = my_render_fn(keypoints_3d_with_score[..., :-1], fig, ax)
            for img_3d in output_imgs_3d:
                out_writer_3d.write(img_3d)
            out_writer_3d.release()
            print('>>> Saved output 3d video')

        # Convert to pixel coordinates
        keypoints_3d_with_score[..., :-1] = keypoints_3d_with_score[..., :-1] * (min(vid_size) / 2.0)
        keypoints_3d_with_score[..., :-2] = keypoints_3d_with_score[..., :-2] + np.array(vid_size) / 2.0

        np.savez(output_path_3d_npz, frame_ith=np.array(frame_ith, dtype=np.int32), keypoints=np.array(keypoints_3d_with_score, dtype=np.float32))
        print('>>> Saved keypoints 3d npz')
            
def main(args):
    vitpose_model = VitInference(args.model, args.yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=True,
                         single_pose=args.single_pose,
                         yolo_step=args.yolo_step)  # type: ignore
    
    print(f">>> Vitpose Model loaded: {args.model}")  

    # === motionbert model ===
    motionbert_args = get_config("motionbert/configs/pose3d/MB_ft_h36m_global_lite.yaml")
    motionbert_model_backbone = load_backbone(motionbert_args)
    if torch.cuda.is_available():
        # Added for compatibility with multi-GPU machines
        # The original checkpoint also used this, so we use it here as well
        motionbert_model_backbone = nn.DataParallel(motionbert_model_backbone)
        motionbert_model_backbone = motionbert_model_backbone.cuda()
    
    motionbert_checkpoint = torch.load("models/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin", map_location=lambda storage, loc: storage.cuda())
    motionbert_model_backbone.load_state_dict(motionbert_checkpoint['model_pos'], strict=True)
    motionbert_model_pos = motionbert_model_backbone
    motionbert_model_pos.eval()
    print(f">>> Motionbert Model loaded", "models/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin")  
    # === motionbert model ends ===

    root = os.path.abspath(__file__).split("pose_est_model")[0]
    forms_video_folder = os.path.join(root,f"fps{args.fps}","Taichi_Clip")
    forms_keypoint_folder = os.path.join(root,f"fps{args.fps}","Taichi_Clip","forms_keypoints")
    form_ids = [f"{i:02}" for i in range(24)]
    file_count = 0
    
    for form_id in form_ids:
        for form_root, form_dirs, form_files in sorted(os.walk(os.path.join(forms_video_folder, form_id))):
            for form_file in sorted(form_files):
                if form_file.endswith(".mp4"):   
                    _, camera_view, human_id, file_name_with_ext = form_file.split("_")
                    output_folder = os.path.join(forms_keypoint_folder, form_id)
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f"f{form_id}_v{camera_view}_h{human_id}_{file_count:05}")
                    input_video_path = os.path.join(form_root, form_file)
                    process_video(input_video_path, output_path, vitpose_model, motionbert_model_pos, args, args.fps)
                    file_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='./models/vitpose/vitpose-b-wholebody.pth',
                        help='checkpoint path of the model')

    parser.add_argument('--yolo', type=str, default='./models/yolov8s.pt',
                        help='checkpoint path of the yolo model')

    parser.add_argument('--dataset', type=str, default="wholebody",
                        help='Name of the dataset. If None it"s extracted from the file name. \
                              ["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')

    parser.add_argument('--det-class', type=str, default="human",
                        help='["human"]')

    parser.add_argument('--model-name', type=str, default="b", choices=['s', 'b', 'l', 'h'],
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')

    parser.add_argument('--yolo-size', type=int, default=320,
                        help='YOLOv8 image size during inference')

    # # not used
    # parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0,
    #                     help='Rotate the image of [90, 180, 270] degress counterclockwise')

    parser.add_argument('--yolo-step', type=int, default=1,
                        help='The tracker can be used to predict the bboxes instead of yolo for performance, '
                             'this flag specifies how often yolo is applied (e.g. 1 applies yolo every frame). '
                             'This does not have any effect when is_video is False')

    parser.add_argument('--single-pose', default=True, action='store_true',
                        help='Do not use SORT tracker because single pose is expected in the video')
    
    parser.add_argument('--save-video', default=False, action='store_true',
                        help='save keypoints video results')

    parser.add_argument('--fps', type=int, default=10,
                        help='fps')

    args = parser.parse_args()

    main(args)