import mediapipe as mp

from mediapipe.tasks.python.components.containers.landmark import Landmark, NormalizedLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode as VisionRunningMode
from mediapipe.tasks.python import BaseOptions 

import cv2
import numpy as np
import pdb
import time 
import argparse
import os
import subprocess

import dataclasses
from typing import List, Mapping, Optional, Tuple, Union
import enum

import matplotlib.pyplot as plt

# motionbert 
import torch.nn as nn
from torch.utils.data import DataLoader
from motionbert.utils.tools import *
from motionbert.utils.learning import *
from motionbert.data.dataset_wild import WildDetDataset
from motionbert.utils.utils_data import flip_data
from motionbert.utils.vismo import my_render_fn
from tqdm import tqdm


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

@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = (224, 224, 224)
    thickness: int = 2
    circle_radius: int = 2


class PoseLandmarkID(enum.IntEnum):
    """The 33 pose landmarks."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return value >= 0 and value <= 1

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
            return None
            
    x_px = min(int(np.floor(normalized_x * image_width)), image_width - 1)
    y_px = min(int(np.floor(normalized_y * image_height)), image_height - 1)

    return x_px, y_px

def draw_landmarks(image,landmark_list):
    _BGR_CHANNELS = 3
    _VISIBILITY_THRESHOLD = 0.0
    _PRESENCE_THRESHOLD = 0.0

    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')

    image_height, image_width, _ = image.shape
    # image = np.ones((image_width,image_height,3),dtype=np.uint8)*255
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list):
        if (landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.presence < _PRESENCE_THRESHOLD):
            continue

        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)
    
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    # POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    #                               (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    #                               (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    #                               (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    #                               (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    #                               (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    #                               (29, 31), (30, 32), (27, 31), (28, 32)])


    POSE_CONNECTIONS = frozenset([(0, 2), (2, 7), (0, 5), (5, 8), (9, 10), (11, 12), (11, 13),
                                  (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                                  (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                                  (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                                  (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                                  (29, 31), (30, 32), (27, 31), (28, 32)])

    POSE_LANDMARKS_LEFT = frozenset([
        PoseLandmarkID.LEFT_EYE_INNER, PoseLandmarkID.LEFT_EYE,
        PoseLandmarkID.LEFT_EYE_OUTER, PoseLandmarkID.LEFT_EAR, PoseLandmarkID.MOUTH_LEFT,
        PoseLandmarkID.LEFT_SHOULDER, PoseLandmarkID.LEFT_ELBOW,
        PoseLandmarkID.LEFT_WRIST, PoseLandmarkID.LEFT_PINKY, PoseLandmarkID.LEFT_INDEX,
        PoseLandmarkID.LEFT_THUMB, PoseLandmarkID.LEFT_HIP, PoseLandmarkID.LEFT_KNEE,
        PoseLandmarkID.LEFT_ANKLE, PoseLandmarkID.LEFT_HEEL,
        PoseLandmarkID.LEFT_FOOT_INDEX
    ])

    POSE_LANDMARKS_RIGHT = frozenset([
        PoseLandmarkID.RIGHT_EYE_INNER, PoseLandmarkID.RIGHT_EYE,
        PoseLandmarkID.RIGHT_EYE_OUTER, PoseLandmarkID.RIGHT_EAR,
        PoseLandmarkID.MOUTH_RIGHT, PoseLandmarkID.RIGHT_SHOULDER,
        PoseLandmarkID.RIGHT_ELBOW, PoseLandmarkID.RIGHT_WRIST,
        PoseLandmarkID.RIGHT_PINKY, PoseLandmarkID.RIGHT_INDEX,
        PoseLandmarkID.RIGHT_THUMB, PoseLandmarkID.RIGHT_HIP, PoseLandmarkID.RIGHT_KNEE,
        PoseLandmarkID.RIGHT_ANKLE, PoseLandmarkID.RIGHT_HEEL,
        PoseLandmarkID.RIGHT_FOOT_INDEX
    ])

    # left_spec = DrawingSpec(color=(0, 138, 255), thickness=2)
    # right_spec = DrawingSpec(color=(231, 217, 0), thickness=2)
    left_spec = DrawingSpec(color=(255, 138, 0), thickness=2)
    right_spec = DrawingSpec(color=(0, 217, 213), thickness=2)

    landmark_drawing_spec = {}

    for landmark in POSE_LANDMARKS_LEFT:
        landmark_drawing_spec[landmark] = left_spec
    for landmark in POSE_LANDMARKS_RIGHT:
        landmark_drawing_spec[landmark] = right_spec

    landmark_drawing_spec[PoseLandmarkID.NOSE] = DrawingSpec(color=(224,224,224), thickness=2)

    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    line_drawing_spec = DrawingSpec(color=(224,224,224), thickness=2)
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection from landmark #{start_idx} to landmark #{end_idx}.')
                         
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(image, idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], line_drawing_spec.color,
                line_drawing_spec.thickness)
                
    
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for idx, landmark_px in idx_to_coordinates.items():
        drawing_spec = landmark_drawing_spec[idx]
        
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
        cv2.circle(image, landmark_px, circle_border_radius, (224,224,224), drawing_spec.thickness)
        
        # Fill color into the circle
        cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)
    
    # if(image_width>image_height):
    #     image = cv2.resize(image, (256, int(image_height*(256/image_width))), interpolation=cv2.INTER_LINEAR)
    # else:
    #     image = cv2.resize(image, (192, int(image_height*(192/image_width))), interpolation=cv2.INTER_LINEAR)

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr_image


def plot_landmarks(landmark_list, fig, ax):
    _VISIBILITY_THRESHOLD = 0.0
    _PRESENCE_THRESHOLD = 0.0

    if not landmark_list:
        return

    # ax.set_box_aspect([1,1,1])
    # ax.xaxis.set_major_locator(MultipleLocator(0.1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    # ax.zaxis.set_major_locator(MultipleLocator(0.1))
    plotted_landmarks = {}

    for idx, landmark in enumerate(landmark_list):
        if (landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.presence < _PRESENCE_THRESHOLD):
            continue
        
        plotted_landmarks[idx] = (landmark.x*100, landmark.y*100, landmark.z*100)

    plotted_landmarks_ndarray = np.array([[landmark[0],landmark[1],landmark[2]] for landmark in plotted_landmarks.values()])
    all_x,all_y,all_z = plotted_landmarks_ndarray[:,0],plotted_landmarks_ndarray[:,1],plotted_landmarks_ndarray[:,2]
    
    ax.cla()

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.   
    # plot_radius = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    # mid_x = (all_x.max()+all_x.min())/2.0
    # mid_y = (all_y.max()+all_y.min())/2.0
    # mid_z = (all_z.max()+all_z.min())/2.0
    # ax.set_xlim(mid_x - plot_radius, mid_x + plot_radius)
    # ax.set_ylim(mid_y - plot_radius, mid_y + plot_radius)
    # ax.set_zlim(mid_z - plot_radius, mid_z + plot_radius)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)

    POSE_CONNECTIONS = frozenset([(0, 2), (2, 7), (0, 5), (5, 8), (9, 10), (11, 12), (11, 13),
                                  (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                                  (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                                  (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                                  (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                                  (29, 31), (30, 32), (27, 31), (28, 32)])

    POSE_LANDMARKS_LEFT = frozenset([
        PoseLandmarkID.LEFT_EYE_INNER, PoseLandmarkID.LEFT_EYE,
        PoseLandmarkID.LEFT_EYE_OUTER, PoseLandmarkID.LEFT_EAR, PoseLandmarkID.MOUTH_LEFT,
        PoseLandmarkID.LEFT_SHOULDER, PoseLandmarkID.LEFT_ELBOW,
        PoseLandmarkID.LEFT_WRIST, PoseLandmarkID.LEFT_PINKY, PoseLandmarkID.LEFT_INDEX,
        PoseLandmarkID.LEFT_THUMB, PoseLandmarkID.LEFT_HIP, PoseLandmarkID.LEFT_KNEE,
        PoseLandmarkID.LEFT_ANKLE, PoseLandmarkID.LEFT_HEEL,
        PoseLandmarkID.LEFT_FOOT_INDEX
    ])

    POSE_LANDMARKS_RIGHT = frozenset([
        PoseLandmarkID.RIGHT_EYE_INNER, PoseLandmarkID.RIGHT_EYE,
        PoseLandmarkID.RIGHT_EYE_OUTER, PoseLandmarkID.RIGHT_EAR,
        PoseLandmarkID.MOUTH_RIGHT, PoseLandmarkID.RIGHT_SHOULDER,
        PoseLandmarkID.RIGHT_ELBOW, PoseLandmarkID.RIGHT_WRIST,
        PoseLandmarkID.RIGHT_PINKY, PoseLandmarkID.RIGHT_INDEX,
        PoseLandmarkID.RIGHT_THUMB, PoseLandmarkID.RIGHT_HIP, PoseLandmarkID.RIGHT_KNEE,
        PoseLandmarkID.RIGHT_ANKLE, PoseLandmarkID.RIGHT_HEEL,
        PoseLandmarkID.RIGHT_FOOT_INDEX
    ])

    num_landmarks = len(landmark_list)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in POSE_CONNECTIONS:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')

      if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
        if(start_idx in POSE_LANDMARKS_LEFT):
            kpt_color = "#FF8A00"
        elif(start_idx in POSE_LANDMARKS_RIGHT):
            kpt_color = "#00D9E7"
        else:
            kpt_color = "#23577E"
        landmark_pair = [
            plotted_landmarks[start_idx], plotted_landmarks[end_idx]
        ]
        
        ax.plot(
            xs = [-landmark_pair[0][0], -landmark_pair[1][0]],
            ys = [-landmark_pair[0][2], -landmark_pair[1][2]],
            zs = [-landmark_pair[0][1], -landmark_pair[1][1]],
            color = kpt_color)
    
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def draw_landmarks_on_video(rgb_image, detection_result, fig, ax):
    annotated_image_2d = np.copy(rgb_image)
    annotated_image_2d = draw_landmarks(annotated_image_2d,detection_result.pose_landmarks[0])
    annotated_image_3d = plot_landmarks(detection_result.pose_world_landmarks[0], fig, ax)
    return annotated_image_2d, annotated_image_3d

def process_video(input_file=None, output_path=None, motionbert_model=None, args=None, fps_in=10):

    input_path = input_file
    ext = input_path[input_path.rfind('.'):]    

    if output_path:
        if os.path.isdir(output_path):
            output_path_video = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}{ext}"))
            output_path_video_3d_pose = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_3d_pose{ext}"))
            output_path_video_3d_world = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_3d_world{ext}"))
            output_path_3d_pose_npz = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_3d_pose.npz"))
            output_path_3d_world_npz = os.path.join(output_path, os.path.basename(input_path).replace(ext, f"_fps{fps_in}_3d_world.npz"))
        else:
            output_path_video = output_path + f"_fps{fps_in}{ext}"
            output_path_video_3d_pose = output_path + f"_fps{fps_in}_3d_pose{ext}"
            output_path_video_3d_world = output_path + f"_fps{fps_in}_3d_world{ext}"
            output_path_3d_pose_npz = output_path + f"_fps{fps_in}_3d_pose.npz"
            output_path_3d_world_npz = output_path + f"_fps{fps_in}_3d_world.npz"

    if(os.path.isfile(output_path_3d_world_npz)):
        print(f'>>> {output_path_3d_world_npz} existed')
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
            out_writer = cv2.VideoWriter(output_path_video_3d_pose,
                                        #cv2.VideoWriter_fourcc(*'vp09'),
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        fps_in, (frame_width, frame_height))  # type: ignore

            out_writer_3d = cv2.VideoWriter(output_path_video_3d_world,
                                            #cv2.VideoWriter_fourcc(*'vp09'),
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps_in, (400,400))  # type: ignore

        # Using GPU in WSL on Windows doesn’t work; only CPU can be used.
        # On Ubuntu, GPU can be used.
        if os.name == 'posix' and 'WSL_DISTRO_NAME' not in os.environ:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=args.model,delegate=BaseOptions.Delegate.GPU),
                running_mode=VisionRunningMode.VIDEO)
        elif os.name == 'nt' or 'WSL_DISTRO_NAME' in os.environ:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=args.model,delegate=BaseOptions.Delegate.CPU),
                running_mode=VisionRunningMode.VIDEO)

        landmarker = PoseLandmarker.create_from_options(options)

        keypoints = []
        keypoints_3d = []
        frame_ith = []
        ith = 0
        t0_timestamp = time.time()
        t0_start = False
        timestamp_list = []
        timestamp_list.append(t0_timestamp)
        print(f'>>> Running inference on {input_path}')

        while cap.grab():
            success, frame = cap.retrieve()

            if not success:
                print("Error !!")
                break

            frame_timestamp = time.time()
            timestamp_list.append(frame_timestamp)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_timestamp*1000))
            
            if(len(pose_landmarker_result.pose_landmarks)==0):
                continue
            
            result_ndarray = OneEuroFilter.landmark2ndarray(pose_landmarker_result)

            if not t0_start:
                euro_filter = OneEuroFilter(t0=t0_timestamp, input0=result_ndarray, min_cutoff=0.3, beta=0.007, d_cutoff=fps_in)
                t0_start = True

            ndarray_filtered = euro_filter(t=frame_timestamp, input=result_ndarray)
            pose_landmarker_result_filterd = OneEuroFilter.ndarray2landmark(ndarray_filtered,pose_landmarker_result)
            pose_landmarker_result = pose_landmarker_result_filterd
            if args.save_video:
                img_3d, img_3d_world = draw_landmarks_on_video(mp_image.numpy_view(), pose_landmarker_result, fig, ax)
                out_writer.write(img_3d)
                # out_writer_3d.write(img_3d_world)

            image_height, image_width, _ = mp_image.numpy_view().shape
            pose_landmarks = pose_landmarker_result.pose_landmarks[0]
            pose_world_landmarks = pose_landmarker_result.pose_world_landmarks[0]
            tmp_pose_data = np.array([[lm.x, lm.y, lm.visibility] for lm in pose_landmarks])
            tmp_pose_data[:,0] = np.clip(tmp_pose_data[:,0]*image_width, a_min=0, a_max=image_width - 1)
            tmp_pose_data[:,1] = np.clip(tmp_pose_data[:,1]*image_height, a_min=0, a_max=image_height - 1)
            tmp_pose_world_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_world_landmarks])
            keypoints.append(tmp_pose_data)
            keypoints_3d.append(tmp_pose_world_data)
            frame_ith.append(ith)
            ith += 1

        landmarker.close()

        np.savez(output_path_3d_pose_npz, frame_ith=np.array(frame_ith, dtype=np.int32), keypoints=np.array(keypoints, dtype=np.float32))
        print('>>> Saved keypoints 3d_pose npz')

        wild_dataset = WildDetDataset(keypoints, clip_len=1, vid_size=vid_size, scale_range=None, focus=None, data_type="mediapipe")
        
        test_loader = DataLoader(wild_dataset)
        t0_start = False
        ith = 0
        keypoints_3d_with_score = []

        with torch.no_grad():
            for batch_input in test_loader:

                N, T = batch_input.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                
                predicted_3d_pos = motionbert_model(batch_input)

                tmp_kpts_3d_with_score = torch.cat((predicted_3d_pos, batch_input[...,-1:]), dim=-1).cpu().numpy()

                if not t0_start:
                    euro_filter = OneEuroFilter(t0=timestamp_list[ith], input0=tmp_kpts_3d_with_score[...,:-1], min_cutoff=1.0, beta=0.007, d_cutoff=fps_in)
                    t0_start = True
                    ith += 1
    
                tmp_kpts_3d_with_score[...,:-1] = euro_filter(t=timestamp_list[ith], input=tmp_kpts_3d_with_score[...,:-1])

                keypoints_3d_with_score.append(tmp_kpts_3d_with_score)
                ith += 1


        # shape: (batch_size, frame_size, 17, 3)
        keypoints_3d_with_score = np.hstack(keypoints_3d_with_score)
        # shape: (batch_size*frame_size, 17, 3)
        keypoints_3d_with_score = np.concatenate(keypoints_3d_with_score)

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

        np.savez(output_path_3d_world_npz, frame_ith=np.array(frame_ith, dtype=np.int32), keypoints=np.array(keypoints_3d_with_score, dtype=np.float32))
        print('>>> Saved keypoints 3d npz')

        if args.save_video:
            print('>>> Saved output 3d_pose & 3d_world video')
            out_writer.release()

        if(len(keypoints)==0):
            if args.save_video:
                os.remove(output_path_video_3d_pose)
                os.remove(output_path_video_3d_world)
            return

        np.savez(output_path_3d_world_npz, frame_ith=np.array(frame_ith, dtype=np.int32), keypoints=np.array(keypoints_3d, dtype=np.float32))
        print('>>> Saved keypoints 3d_world npz')


def main(args):

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
    forms_keypoint_folder = os.path.join(root,f"fps{args.fps}","Taichi_Clip","forms_keypoints_mediapipe")
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
                    process_video(input_video_path, output_path, motionbert_model_pos, args, args.fps)
                    file_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='./models/mediapipe/pose_landmarker_heavy.task',
                        help='checkpoint path of the model')
 
    parser.add_argument('--save-video', default=False, action='store_true',
                        help='save keypoints video results')

    parser.add_argument('--fps', type=int, default=10,
                        help='fps')

    args = parser.parse_args()

    main(args)


'''
references:

https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/vision/pose_landmarker.py
https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/components/containers/landmark.py
https://jaantollander.com/post/noise-filtering-using-one-euro-filter/
https://drive.google.com/file/d/10WlcTvrQnR_R2TdTmKw0nkyRLqrwNkWU/preview
https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md#pose_landmarks
https://github.com/google/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py#L34-L121
'''


