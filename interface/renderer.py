import numpy as np
import dataclasses
import cv2
import enum
from typing import List, Mapping, Optional, Tuple, Union

@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = (224, 224, 224)
    thickness: int = 2
    circle_radius: int = 2


# human3.6m
class PoseLandmarkID(enum.IntEnum):
    """The 17 pose landmarks."""
    HIP = 0
    RIGHT_HIP = 1
    RIGHT_KNEE = 2
    RIGHT_FOOT = 3
    LEFT_HIP = 4
    LEFT_KNEE = 5
    LEFT_FOOT = 6
    SPINE = 7
    THORAX = 8
    NECK = 9
    HEAD = 10
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13
    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16


class Kpts_Renderer:
    def __init__(self, kpts_type):
        self.kpts_type = kpts_type

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
        y[9,:] = (x[8,:] + y[10,:]) * 0.5
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

    def cocowhole2h36m(self, x):
        V, C = x.shape
        y = np.zeros([17,C])
        # 0: hip
        y[0,:] = (x[11,:] + x[12,:]) * 0.5
        # 1: right hip
        y[1,:] = x[12,:]
        # 2: right knee
        y[2,:] = x[14,:]
        # 3: right foot
        y[3,:] = x[16,:]
        # 4: left hip
        y[4,:] = x[11,:]
        # 5: left knee
        y[5,:] = x[13,:]
        # 6: left foot
        y[6,:] = x[15,:]
        # 8: thorax
        y[8,:] = (x[5,:] + x[6,:]) * 0.5
        # 10: head
        y[10,:] = (x[3,:] + x[4,:]) * 0.5
        # 9: neck
        y[9,:] = (y[8,:] + y[10,:]) * 0.5
        # 7: spine
        y[7,:] = (y[8,:] + y[0,:]) * 0.5
        # 11: left shoulder
        y[11,:] = x[5,:]
        # 12: left elbow
        y[12,:] = x[7,:]
        # 13: left wrist
        y[13,:] = x[9,:]
        # 14: right shoulder
        y[14,:] = x[6,:]
        # 15: right elbow
        y[15,:] = x[8,:]
        # 16: right wrist
        y[16,:] = x[10,:]
        return y

    def draw_landmarks(self, image, kpts_pose):
        _BGR_CHANNELS = 3
        _VISIBILITY_THRESHOLD = 0.0
        _PRESENCE_THRESHOLD = 0.0

        if image.shape[2] != _BGR_CHANNELS:
            raise ValueError('Input image must contain three channel bgr data.')

        image_height, image_width, _ = image.shape

        # kpts_pose shape: (33, 3) -> kpts_pose shape: (17, 3)
        if self.kpts_type == "mediapipe":
            kpts_pose = self.mediapipe2h36m(kpts_pose)
        elif self.kpts_type == "vitpose":
            kpts_pose = self.cocowhole2h36m(kpts_pose)

        # image = np.ones((image_width,image_height,3),dtype=np.uint8)*255
        # idx_to_coordinates = {}
        # for idx, landmark in enumerate(landmark_list):
        #     if (landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.presence < _PRESENCE_THRESHOLD):
        #         continue

        #     landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)

        #     if landmark_px:
        #         idx_to_coordinates[idx] = landmark_px

        

        POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
                                      (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
                                      (12, 13), (8 ,14), (14, 15), (15, 16)])

        POSE_LANDMARKS_LEFT = frozenset([
            PoseLandmarkID.LEFT_HIP, PoseLandmarkID.LEFT_KNEE,
            PoseLandmarkID.LEFT_FOOT, PoseLandmarkID.LEFT_SHOULDER, PoseLandmarkID.LEFT_ELBOW,
            PoseLandmarkID.LEFT_WRIST
        ])

        POSE_LANDMARKS_RIGHT = frozenset([
            PoseLandmarkID.RIGHT_HIP, PoseLandmarkID.RIGHT_KNEE,
            PoseLandmarkID.RIGHT_FOOT, PoseLandmarkID.RIGHT_SHOULDER,
            PoseLandmarkID.RIGHT_ELBOW, PoseLandmarkID.RIGHT_WRIST
        ])

        POSE_LANDMARKS_MIDDLE = frozenset([
            PoseLandmarkID.HIP, PoseLandmarkID.SPINE,
            PoseLandmarkID.THORAX, PoseLandmarkID.NECK,
            PoseLandmarkID.HEAD
        ])

        left_spec = DrawingSpec(color=(255, 138, 0), thickness=2)
        right_spec = DrawingSpec(color=(0, 217, 213), thickness=2)
        middle_spec = DrawingSpec(color=(224,224,224), thickness=2)

        landmark_drawing_spec = {}

        for landmark in POSE_LANDMARKS_LEFT:
            landmark_drawing_spec[landmark] = left_spec
        for landmark in POSE_LANDMARKS_RIGHT:
            landmark_drawing_spec[landmark] = right_spec
        for landmark in POSE_LANDMARKS_MIDDLE:
            landmark_drawing_spec[landmark] = middle_spec

        # bones color
        line_drawing_spec = DrawingSpec(color=(224,224,224), thickness=2)

        # Draws the connections if the start and end landmarks are both visible.
        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            cv2.line(image, (int(kpts_pose[start_idx, 0]), int(kpts_pose[start_idx, 1])), (int(kpts_pose[end_idx, 0]),int(kpts_pose[end_idx, 1])), line_drawing_spec.color, line_drawing_spec.thickness)

        # Draws landmark points after finishing the connection lines, which is
        # aesthetically better.
        for idx in range(len(kpts_pose)):
            drawing_spec = landmark_drawing_spec[idx]

            # White circle border
            cv2.circle(image, (int(kpts_pose[idx,0]), int(kpts_pose[idx,1])), int(drawing_spec.circle_radius * 1.2), (224,224,224), drawing_spec.thickness)

            # Fill color into the circle
            cv2.circle(image, (int(kpts_pose[idx,0]), int(kpts_pose[idx,1])), drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)
        
        # bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def draw_landmarks_on_video(self, rgb_image, kpts_pose):
        # kpts_pose shape: (33, 3)
        annotated_image_2d = np.copy(rgb_image)
        annotated_image_2d = self.draw_landmarks(annotated_image_2d, kpts_pose)
        return annotated_image_2d