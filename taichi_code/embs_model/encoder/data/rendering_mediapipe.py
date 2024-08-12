import os 
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import dataclasses
from typing import List, Mapping, Optional, Tuple, Union
import enum

@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = (224, 224, 224)
    thickness: int = 2
    circle_radius: int = 2


class PoseLandmarkID(enum.IntEnum):
    """The 33 pose landmarks."""
    NOSE = 0
    # LEFT_EYE_INNER = 1
    LEFT_EYE = 1
    # LEFT_EYE_OUTER = 3
    # RIGHT_EYE_INNER = 4
    RIGHT_EYE = 2
    # RIGHT_EYE_OUTER = 6
    LEFT_EAR = 3
    RIGHT_EAR = 4
    # MOUTH_LEFT = 9
    # MOUTH_RIGHT = 10
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_PINKY = 11
    RIGHT_PINKY = 12
    LEFT_INDEX = 13
    RIGHT_INDEX = 14
    LEFT_THUMB = 15
    RIGHT_THUMB = 16
    LEFT_HIP = 17
    RIGHT_HIP = 18
    LEFT_KNEE = 19
    RIGHT_KNEE = 20
    LEFT_ANKLE = 21
    RIGHT_ANKLE = 22
    LEFT_HEEL = 23
    RIGHT_HEEL = 24
    LEFT_FOOT_INDEX = 25
    RIGHT_FOOT_INDEX = 26
    # my addition
    THORAX = 27
    HIP = 28

# class PoseLandmarkID(enum.IntEnum):
#     """The 33 pose landmarks."""
#     NOSE = 0
#     # LEFT_EYE_INNER = 1
#     LEFT_EYE = 2 1
#     # LEFT_EYE_OUTER = 3
#     # RIGHT_EYE_INNER = 4
#     RIGHT_EYE = 5 2
#     # RIGHT_EYE_OUTER = 6
#     LEFT_EAR = 7 3
#     RIGHT_EAR = 8 4
#     # MOUTH_LEFT = 9
#     # MOUTH_RIGHT = 10
#     LEFT_SHOULDER = 11 5
#     RIGHT_SHOULDER = 12 6
#     LEFT_ELBOW = 13 7
#     RIGHT_ELBOW = 14 8
#     LEFT_WRIST = 15 9
#     RIGHT_WRIST = 16 10
#     LEFT_PINKY = 17 11
#     RIGHT_PINKY = 18 12
#     LEFT_INDEX = 19 13
#     RIGHT_INDEX = 20 14
#     LEFT_THUMB = 21 15
#     RIGHT_THUMB = 22 16
#     LEFT_HIP = 23 17
#     RIGHT_HIP = 24 18
#     LEFT_KNEE = 25 19
#     RIGHT_KNEE = 26 20
#     LEFT_ANKLE = 27 21
#     RIGHT_ANKLE = 28 22
#     LEFT_HEEL = 29 23
#     RIGHT_HEEL = 30 24
#     LEFT_FOOT_INDEX = 31 25
#     RIGHT_FOOT_INDEX = 32 26

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
            
    # -1 is because this value needs to be <= index, not <= len
    x_px = min(int(np.floor(normalized_x * image_width)), image_width - 1)
    y_px = min(int(np.floor(normalized_y * image_height)), image_height - 1)

    # tuple
    return x_px, y_px


def get_img_from_fig(fig):
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    return img_bgr



def draw_landmarks(image,landmark_ndarray):
    # original connection
    # POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    #                               (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    #                               (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    #                               (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    #                               (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    #                               (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    #                               (29, 31), (30, 32), (27, 31), (28, 32)])

    # my connection
    POSE_CONNECTIONS = frozenset([(0, 1), (1, 3), (0, 2), (2, 4), (5, 6), (5, 7),
                                  (7, 9), (9, 11), (9, 13), (9, 15), (11, 13),
                                  (6, 8), (8, 10), (10, 12), (10, 14), (10, 16),
                                  (12, 14), (5, 17), (6, 18), (17, 18), (17, 19),
                                  (18, 20), (19, 21), (20, 22), (21, 23), (22, 24),
                                  (23, 25), (24, 26), (21, 25), (22, 26)])

    POSE_LANDMARKS_LEFT = frozenset([
        PoseLandmarkID.LEFT_EYE, PoseLandmarkID.LEFT_EAR,
        PoseLandmarkID.LEFT_SHOULDER, PoseLandmarkID.LEFT_ELBOW,
        PoseLandmarkID.LEFT_WRIST, PoseLandmarkID.LEFT_PINKY, PoseLandmarkID.LEFT_INDEX,
        PoseLandmarkID.LEFT_THUMB, PoseLandmarkID.LEFT_HIP, PoseLandmarkID.LEFT_KNEE,
        PoseLandmarkID.LEFT_ANKLE, PoseLandmarkID.LEFT_HEEL,
        PoseLandmarkID.LEFT_FOOT_INDEX
    ])

    POSE_LANDMARKS_RIGHT = frozenset([
        PoseLandmarkID.RIGHT_EYE, PoseLandmarkID.RIGHT_EAR,
        PoseLandmarkID.RIGHT_SHOULDER, PoseLandmarkID.RIGHT_ELBOW, 
        PoseLandmarkID.RIGHT_WRIST, PoseLandmarkID.RIGHT_PINKY, PoseLandmarkID.RIGHT_INDEX,
        PoseLandmarkID.RIGHT_THUMB, PoseLandmarkID.RIGHT_HIP, PoseLandmarkID.RIGHT_KNEE,
        PoseLandmarkID.RIGHT_ANKLE, PoseLandmarkID.RIGHT_HEEL,
        PoseLandmarkID.RIGHT_FOOT_INDEX
    ])

    POSE_LANDMARKS_MIDDLE = frozenset([
        PoseLandmarkID.NOSE, PoseLandmarkID.THORAX, PoseLandmarkID.HIP,
    ])

    # RGB
    left_spec = DrawingSpec(color=(255, 138, 0), thickness=1)
    right_spec = DrawingSpec(color=(0, 217, 213), thickness=1)
    middle_spec = DrawingSpec(color=(35,87,126), thickness=1)

    landmark_drawing_spec = {}

    for landmark in POSE_LANDMARKS_LEFT:
        landmark_drawing_spec[landmark] = left_spec
    for landmark in POSE_LANDMARKS_RIGHT:
        landmark_drawing_spec[landmark] = right_spec
    for landmark in POSE_LANDMARKS_MIDDLE:
        landmark_drawing_spec[landmark] = middle_spec

    # landmark_drawing_spec[PoseLandmarkID.NOSE] = DrawingSpec(color=(224,224,224), thickness=2)
    # Draws the connections if the start and end landmarks are both visible.
    line_drawing_spec = DrawingSpec(color=(224,224,224), thickness=2)

    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(image, landmark_ndarray[start_idx, :-2].astype(np.int32),
            landmark_ndarray[end_idx, :-2].astype(np.int32), line_drawing_spec.color[::-1],
            line_drawing_spec.thickness)
   
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for idx, landmark_px in enumerate(landmark_ndarray):
        drawing_spec = landmark_drawing_spec[idx]
        
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
        cv2.circle(image, landmark_px[:-2].astype(np.int32), circle_border_radius, (224,224,224)[::-1], drawing_spec.thickness)
        
        # Fill color into the circle
        cv2.circle(image, landmark_px[:-2].astype(np.int32), drawing_spec.circle_radius, drawing_spec.color[::-1], drawing_spec.thickness)
    
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr_image


def plot_landmarks(landmark_ndarray, fig, ax):

    # ax.set_box_aspect([1,1,1])
    # ax.xaxis.set_major_locator(MultipleLocator(0.1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    # ax.zaxis.set_major_locator(MultipleLocator(0.1))
    # plotted_landmarks = {}

    # for idx, landmark in enumerate(landmark_ndarray):        
    #     plotted_landmarks[idx] = (landmark.x*100, landmark.y*100+20, landmark.z*100)

    # plotted_landmarks_ndarray = np.array([[landmark[0],landmark[1],landmark[2]] for landmark in plotted_landmarks.values()])
    # all_x,all_y,all_z = plotted_landmarks_ndarray[:,0],plotted_landmarks_ndarray[:,1],plotted_landmarks_ndarray[:,2]
    
    ax.cla()
    ax.view_init(elev=12.0, azim=80.0)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.   
    # plot_radius = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    # mid_x = (all_x.max()+all_x.min())/2.0
    # mid_y = (all_y.max()+all_y.min())/2.0
    # mid_z = (all_z.max()+all_z.min())/2.0
    # ax.set_xlim(mid_x - plot_radius, mid_x + plot_radius)
    # ax.set_ylim(mid_y - plot_radius, mid_y + plot_radius)
    # ax.set_zlim(mid_z - plot_radius, mid_z + plot_radius)

    ax.set_xlim(-256, 256)
    ax.set_ylim(-256, 256)
    ax.set_zlim(-256, 256)

    # 我少連一些臉上的東西
    POSE_CONNECTIONS = frozenset([(0, 1), (1, 3), (0, 2), (2, 4), (5, 7),
                                  (7, 9), (9, 11), (9, 13), (9, 15), (11, 13),
                                  (6, 8), (8, 10), (10, 12), (10, 14), (10, 16),
                                  (12, 14), (17, 19),
                                  (18, 20), (19, 21), (20, 22), (21, 23), (22, 24),
                                  (23, 25), (24, 26), (21, 25), (22, 26),
                                  (28,18), (28,17), (28,27), (27,6), (27,5), (27,0)])

    POSE_LANDMARKS_LEFT = frozenset([
        PoseLandmarkID.LEFT_EYE, PoseLandmarkID.LEFT_EAR,
        PoseLandmarkID.LEFT_SHOULDER, PoseLandmarkID.LEFT_ELBOW,
        PoseLandmarkID.LEFT_WRIST, PoseLandmarkID.LEFT_PINKY, PoseLandmarkID.LEFT_INDEX,
        PoseLandmarkID.LEFT_THUMB, PoseLandmarkID.LEFT_HIP, PoseLandmarkID.LEFT_KNEE,
        PoseLandmarkID.LEFT_ANKLE, PoseLandmarkID.LEFT_HEEL,
        PoseLandmarkID.LEFT_FOOT_INDEX
    ])

    POSE_LANDMARKS_RIGHT = frozenset([
        PoseLandmarkID.RIGHT_EYE, PoseLandmarkID.RIGHT_EAR,
        PoseLandmarkID.RIGHT_SHOULDER, PoseLandmarkID.RIGHT_ELBOW, 
        PoseLandmarkID.RIGHT_WRIST, PoseLandmarkID.RIGHT_PINKY, PoseLandmarkID.RIGHT_INDEX,
        PoseLandmarkID.RIGHT_THUMB, PoseLandmarkID.RIGHT_HIP, PoseLandmarkID.RIGHT_KNEE,
        PoseLandmarkID.RIGHT_ANKLE, PoseLandmarkID.RIGHT_HEEL,
        PoseLandmarkID.RIGHT_FOOT_INDEX
    ])

    POSE_LANDMARKS_MIDDLE = frozenset([
        PoseLandmarkID.NOSE, PoseLandmarkID.THORAX, PoseLandmarkID.HIP,
    ])

    # Draws the connections if the start and end landmarks are both visible.
    for connection in POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]

        if(end_idx in POSE_LANDMARKS_LEFT):
            kpt_color = "#FF8A00"
        elif(end_idx in POSE_LANDMARKS_RIGHT):
            kpt_color = "#00D9E7"
        elif(end_idx in POSE_LANDMARKS_MIDDLE):
            kpt_color = "#23577E"
        landmark_pair = [
            landmark_ndarray[start_idx], landmark_ndarray[end_idx]
        ]

        ax.plot(
            xs = [-landmark_pair[0][0], -landmark_pair[1][0]],
            ys = [-landmark_pair[0][2], -landmark_pair[1][2]],
            zs = [-landmark_pair[0][1], -landmark_pair[1][1]],
            color = kpt_color)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')   

    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])


def draw_3d_landmarks_on_video(detection_result, fig, ax):

    """ 
    Cannot use slicing method, as it will result in the following error
    >  - Layout of the output array img is incompatible with cv::Mat
    >  - Expected Ptr<cv::UMat> for argument 'img'
    """
    plot_landmarks(detection_result, fig, ax)


def render_keypoints_3d(batch_kpts, use_negative=True, fps_in=10):
    batch_size, clip_len = batch_kpts.shape[0]//4, batch_kpts.shape[1]
    if use_negative:
        columns = 4
    else:
        columns = 2

    out_writer_3d = cv2.VideoWriter(f"../logs/{time.time()}_3d.mp4",
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             # (width, height)
                             fps_in, (300*columns, 300*4))  # type: ignore
    assert out_writer_3d.isOpened()

    # width, height
    figsize = (3*columns, 3*4)
    dpi = 100
    fig, axes = plt.subplots(4, columns, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})
    plt.tight_layout()
    batch_kpts[..., 0] = (batch_kpts[..., 0]+0)*(512/2)
    batch_kpts[..., 1] = (batch_kpts[..., 1]+0)*(512/2)
    batch_kpts[..., 2] = (batch_kpts[..., 2]+0)*(512/2)


    for f in range(clip_len):
        for row_idx in range(len(axes)):
            if batch_size == 1:
                ax = axes[row_idx]
                if row_idx == 0:
                    motion_world = batch_kpts[0]
                elif row_idx == 1:
                    motion_world = batch_kpts[0+batch_size]
                elif row_idx == 2:
                    motion_world = batch_kpts[0+batch_size*2]
                else:
                    motion_world = batch_kpts[0+batch_size*3] 
                draw_3d_landmarks_on_video(motion_world[f], fig, ax)
            else:            
                for column_idx in range(len(axes[0])):
                    ax = axes[row_idx][column_idx]
                    if column_idx == 0:
                        motion_world = batch_kpts[row_idx]
                    elif column_idx == 1:
                        motion_world = batch_kpts[row_idx+batch_size]
                    elif column_idx == 2:
                        motion_world = batch_kpts[row_idx+batch_size*2]
                    else:
                        motion_world = batch_kpts[row_idx+batch_size*3]

                    draw_3d_landmarks_on_video(motion_world[f], fig, ax)

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        out_writer_3d.write(img)
    plt.close()
    out_writer_3d.release()



def render_keypoints_3d_for_sample_demo(video_name, anchor, positive, negative, similarity=None, similarity_negative=None, fps_in=10):
    
    batch_size = 1
    clip_len = anchor.shape[0]
    columns = 3
    
    out_writer_3d = cv2.VideoWriter(f"../logs/{video_name}_{time.time()}_3d.mp4",
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             # (width, height)
                             fps_in, (300*columns, 300*batch_size))  # type: ignore
    assert out_writer_3d.isOpened()

    # width, height
    figsize = (3*columns, 3*batch_size)
    dpi = 100
    fig, axes = plt.subplots(batch_size, columns, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})
    plt.tight_layout()

    anchor[..., 0] = (anchor[..., 0]+0)*(512/2)
    anchor[..., 1] = (anchor[..., 1]+0)*(512/2)
    anchor[..., 2] = (anchor[..., 2]+0)*(512/2)
    positive[..., 0] = (positive[..., 0]+0)*(512/2)
    positive[..., 1] = (positive[..., 1]+0)*(512/2)
    positive[..., 2] = (positive[..., 2]+0)*(512/2)
    negative[..., 0] = (negative[..., 0]+0)*(512/2)
    negative[..., 1] = (negative[..., 1]+0)*(512/2)
    negative[..., 2] = (negative[..., 2]+0)*(512/2)


    for f in range(clip_len):
        for row_idx in range(len(axes)):
            if batch_size == 1:
                ax = axes[row_idx]
                ax.view_init(elev=12., azim=80)
                if row_idx == 0:
                    motion_world = anchor
                elif row_idx == 1: 
                    motion_world = positive
                elif row_idx == 2:
                    motion_world = negative

                draw_3d_landmarks_on_video(motion_world[f], fig, ax)

        frame_vis = get_img_from_fig(fig)
        
        if similarity is not None:
            score = str(round(similarity[f],2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(225,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

        if similarity_negative is not None:
            score = str(round(similarity_negative[f],2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(525,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
       
        out_writer_3d.write(frame_vis)
    
    plt.close()
    out_writer_3d.release()



def render_keypoints_3d_for_tensorboard(anchor, positive, negative, similarity=None, similarity_negative=None, embs_similarity=None, embs_similarity_negative=None):
    batch_size = 1
    clip_len = anchor.shape[0]
    columns = 3

    # width, height
    figsize = (3*columns, 3*batch_size)
    dpi = 100
    fig, axes = plt.subplots(batch_size, columns, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})
    plt.tight_layout()

    anchor[..., 0] = (anchor[..., 0]+0)*(512/2)
    anchor[..., 1] = (anchor[..., 1]+0)*(512/2)
    anchor[..., 2] = (anchor[..., 2]+0)*(512/2)
    positive[..., 0] = (positive[..., 0]+0)*(512/2)
    positive[..., 1] = (positive[..., 1]+0)*(512/2)
    positive[..., 2] = (positive[..., 2]+0)*(512/2)
    negative[..., 0] = (negative[..., 0]+0)*(512/2)
    negative[..., 1] = (negative[..., 1]+0)*(512/2)
    negative[..., 2] = (negative[..., 2]+0)*(512/2)

    output_video = []
    for f in range(clip_len):
        for row_idx in range(len(axes)):
            if batch_size == 1:
                ax = axes[row_idx]
                ax.view_init(elev=12., azim=80)
                if row_idx == 0:
                    motion_world = anchor
                elif row_idx == 1: 
                    motion_world = positive
                elif row_idx == 2:
                    motion_world = negative

                draw_3d_landmarks_on_video(motion_world[f], fig, ax)

        frame_vis = get_img_from_fig(fig)
        
        if similarity is not None:
            score = str(round(similarity,2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(225,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

        if similarity_negative is not None:
            score = str(round(similarity_negative,2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(525,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

        if embs_similarity is not None:
            score = str(round(embs_similarity,2))
            frame_vis = cv2.putText(frame_vis, "Embs Similarity: "+score,(225,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

        if embs_similarity_negative is not None:
            score = str(round(embs_similarity_negative,2))
            frame_vis = cv2.putText(frame_vis, "Embs Similarity: "+score,(525,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        
        frame_vis = frame_vis[..., ::-1]
        output_video.append(frame_vis)

    plt.close()
    # N T C H W
    return np.expand_dims(np.transpose(np.array(output_video), (0,3,1,2)), axis=0)
