import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import Landmark, NormalizedLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode as VisionRunningMode
from mediapipe.tasks.python import BaseOptions 

import numpy as np
import torch
import time
import enum
import dataclasses
import cv2
from typing import List, Mapping, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os

# motionbert 
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from motionbert.utils.tools import *
# from motionbert.utils.learning import *
# from motionbert.data.dataset_wild import WildDetDataset
# from motionbert.data.dataset_wild import read_keypoints
# from motionbert.utils.vismo import my_render_fn
# from tqdm import tqdm


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


class Pose_Estimation:

    def mediapipe2h36m(self, x):
        V, C = x.shape
        y = np.zeros([17,C])
        # 0: hip
        y[0,:] = (x[23,:] + x[24,:]) * 0.5
        # 1: right hip
        y[1,:] = x[24,:]
        # 2: right knee
        y[2,:] = x[26,:]
        # 3: right foot
        y[3,:] = x[28,:]
        # 4: left hip
        y[4,:] = x[23,:]
        # 5: left knee
        y[5,:] = x[25,:]
        # 6: left foot
        y[6,:] = x[27,:]
        # 8: thorax
        y[8,:] = (x[11,:] + x[12,:]) * 0.5
        # 10: head
        y[10,:] = (x[7,:] + x[8,:]) * 0.5
        # 9: neck
        y[9,:] = (y[8,:] + y[10,:]) * 0.5
        # 7: spine
        y[7,:] = (y[8,:] + y[0,:]) * 0.5
        # 11: left shoulder
        y[11,:] = x[11,:]
        # 12: left elbow
        y[12,:] = x[13,:]
        # 13: left wrist
        y[13,:] = x[15,:]
        # 14: right shoulder
        y[14,:] = x[12,:]
        # 15: right elbow
        y[15,:] = x[14,:]
        # 16: right wrist
        y[16,:] = x[16,:]
        return y

    def __init__(self, fps):
        self.mediapipe_start = False
        self.mediapipe_euro_filter = None
        self.motionbert_start = False
        self.motionbert_euro_filter = None
        self.t0_timestamp = time.time()
        self.fps = fps

        if os.name == 'posix' and 'WSL_DISTRO_NAME' not in os.environ:
            self.options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="./models/mediapipe/pose_landmarker_heavy.task",delegate=BaseOptions.Delegate.GPU),
                running_mode=VisionRunningMode.VIDEO)
        elif os.name == 'nt' or 'WSL_DISTRO_NAME' in os.environ:
            self.options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="./models/mediapipe/pose_landmarker_heavy.task",delegate=BaseOptions.Delegate.CPU),
                running_mode=VisionRunningMode.VIDEO)

        self.landmarker = PoseLandmarker.create_from_options(self.options)

        # # === motionbert model ===
        # motionbert_args = get_config("motionbert/configs/pose3d/MB_ft_h36m_global_lite.yaml")
        # motionbert_model_backbone = load_backbone(motionbert_args)
        # if torch.cuda.is_available():
        #     motionbert_model_backbone = nn.DataParallel(motionbert_model_backbone)
        #     motionbert_model_backbone = motionbert_model_backbone.cuda()
    
        # motionbert_checkpoint = torch.load("models/motionbert/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin", map_location=lambda storage, loc: storage.cuda())
        # motionbert_model_backbone.load_state_dict(motionbert_checkpoint['model_pos'], strict=True)
        # motionbert_model_pos = motionbert_model_backbone
        # motionbert_model_pos.eval()
        # self.motionbert_model = motionbert_model_pos

    def _normalized_to_pixel_coordinates(self, 
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

    def draw_landmarks(self, image, landmark_list):
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

            landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)

            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

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
        
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


    def draw_landmarks_on_video(self, rgb_image, detection_result):
        annotated_image_2d = np.copy(rgb_image)
        annotated_image_2d = self.draw_landmarks(annotated_image_2d, detection_result.pose_landmarks[0])
        return annotated_image_2d


    def img_to_kpts(self, img):
        frame_timestamp = time.time()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        image_height, image_width, _ = mp_image.numpy_view().shape
        pose_landmarker_result = self.landmarker.detect_for_video(mp_image, int(frame_timestamp*1000))

        if(len(pose_landmarker_result.pose_landmarks)==0):
            return None, None, None

        # result_ndarray = OneEuroFilter.landmark2ndarray(pose_landmarker_result)

        # if not self.mediapipe_start: 
        #     self.mediapipe_euro_filter = OneEuroFilter(t0=self.t0_timestamp, input0=result_ndarray, min_cutoff=1.0, beta=1.5, d_cutoff=self.fps)
        #     self.mediapipe_start = True

        # ndarray_filtered = self.mediapipe_euro_filter(t=frame_timestamp, input=result_ndarray)
        # pose_landmarker_result_filterd = OneEuroFilter.ndarray2landmark(ndarray_filtered, pose_landmarker_result)
        # pose_landmarker_result = pose_landmarker_result_filterd

        pose_landmarks = pose_landmarker_result.pose_landmarks[0]
        pose_world_landmarks = pose_landmarker_result.pose_world_landmarks[0]
        tmp_pose_data = np.array([[lm.x, lm.y, lm.visibility] for lm in pose_landmarks])
        tmp_pose_data[:,0] = np.clip(tmp_pose_data[:,0]*image_width, a_min=0, a_max=image_width - 1)
        tmp_pose_data[:,1] = np.clip(tmp_pose_data[:,1]*image_height, a_min=0, a_max=image_height - 1)
        tmp_pose_world_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_world_landmarks])

        keypoints_pose = tmp_pose_data
        keypoints_world = tmp_pose_world_data

        return keypoints_pose, self.mediapipe2h36m(keypoints_pose), self.mediapipe2h36m(keypoints_world)
        
        # if use MotionBERT
        # vid_size = (image_width, image_height)
        # batch_input = read_keypoints([keypoints_pose], vid_size, scale_range=None, focus=None, data_type="mediapipe")
        # # add one dimension for batch axis
        # batch_input = torch.from_numpy(batch_input[np.newaxis, :])
        # N, T = batch_input.shape[:2]
        # if torch.cuda.is_available():
        #     batch_input = batch_input.cuda()

        # predicted_3d_pos = self.motionbert_model(batch_input)
        # with torch.no_grad():
        #     tmp_kpts_3d_with_score = torch.cat((predicted_3d_pos, batch_input[...,-1:]), dim=-1).cpu().numpy()

        #     if not self.motionbert_start:
        #         self.motionbert_euro_filter = OneEuroFilter(t0=self.t0_timestamp, input0=tmp_kpts_3d_with_score[...,:-1], min_cutoff=1.0, beta=1.5, d_cutoff=self.fps)
        #         self.motionbert_start = True

        #     tmp_kpts_3d_with_score[...,:-1] = self.motionbert_euro_filter(t=frame_timestamp, input=tmp_kpts_3d_with_score[...,:-1])
        
        
        # # shape是(batch_size*frame_size, 17, 3)
        # keypoints_3d_with_score = np.concatenate(tmp_kpts_3d_with_score)

        
        # # keypoints_pose shape: (33, 3)
        # # keypoints_3d_with_score shape: (1, 17, 4)
        # return keypoints_pose, self.mediapipe2h36m(keypoints_pose), keypoints_3d_with_score[0]


    def __del__(self):
        self.landmarker.close()
        