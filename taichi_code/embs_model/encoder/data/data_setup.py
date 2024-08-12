import os
import torch
from torch.utils.data import DataLoader
import data.utils as kpts_utils
import numpy as np
import re
import time 

def uniformly_sample_rotation_matrix():
    # Uniformly sample azimuth (yaw) between -180 and 180 degrees
    azimuth = np.radians(np.random.uniform(-180, 180))
    # Uniformly sample elevation (pitch) between -10 and 10 degrees
    elevation = np.radians(np.random.uniform(-10, 10))
    # Uniformly sample roll between -10 and 10 degrees
    roll = np.radians(np.random.uniform(-10, 10))
    # Compute individual rotation matrices
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(azimuth), 0, np.sin(azimuth)],
        [0, 1, 0],
        [-np.sin(azimuth), 0, np.cos(azimuth)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation)],
        [0, np.sin(elevation), np.cos(elevation)]
    ])
    # Combine the rotation matrices
    R = R_z @ R_y @ R_x
    return R


def apply_rotation(keypoints_with_score, rotation_matrix):
    keypoints = keypoints_with_score[..., :-1]
    rotated_keypoints = keypoints @ rotation_matrix.T
    keypoints_with_score[..., :-1] = rotated_keypoints

    return keypoints_with_score


def extend_or_clip_video(array, target_length):
    last_frame = array[-1]
    num_to_extend = target_length - array.shape[0]
    if num_to_extend > 0:
        # Duplicate the last frame
        extended_frames = np.repeat(last_frame[np.newaxis, :], num_to_extend, axis=0)
        # Concatenate the original array with the extended frames
        extended_array = np.concatenate((array, extended_frames), axis=0)
    elif num_to_extend < 0:
        extended_array = array[:target_length]
    else:
        extended_array = array
    
    return extended_array

def rotation_based_on_normal(anchor_kpts, positive_kpts, kpts_type):

    # Shape of kpts is
    # (frames_size, kpts_size, channels_size)

    # Use the first frame
    if kpts_type == "motionbert":
        # Shape is (3,)
        anchor_hip_kpt = anchor_kpts[..., 0, 0, :-1]
        anchor_left_shoulder_kpt = anchor_kpts[..., 0, 11, :-1]
        anchor_right_shoulder_kpt = anchor_kpts[..., 0, 14, :-1]

        positive_hip_kpt = positive_kpts[..., 0, 0, :-1]
        positive_left_shoulder_kpt = positive_kpts[..., 0, 11, :-1]
        positive_right_shoulder_kpt = positive_kpts[..., 0, 14, :-1]

    else:
        pass        

    anchor_vector1 = np.array(anchor_left_shoulder_kpt) - np.array(anchor_hip_kpt)
    anchor_vector2 = np.array(anchor_right_shoulder_kpt) - np.array(anchor_hip_kpt)

    positive_vector1 = np.array(positive_left_shoulder_kpt) - np.array(positive_hip_kpt)
    positive_vector2 = np.array(positive_right_shoulder_kpt) - np.array(positive_hip_kpt)

    anchor_normal = np.cross(anchor_vector1, anchor_vector2)  
    anchor_normal = anchor_normal / np.linalg.norm(anchor_normal)
    
    positive_normal = np.cross(positive_vector1, positive_vector2)  
    positive_normal = positive_normal / np.linalg.norm(positive_normal)

    # Project onto the xz plane
    projected_anchor_normal = np.array([anchor_normal[0], 0, anchor_normal[2]])
    projected_positive_normal = np.array([positive_normal[0], 0, positive_normal[2]])

    projected_anchor_normal = projected_anchor_normal / np.linalg.norm(projected_anchor_normal)
    projected_positive_normal = projected_positive_normal / np.linalg.norm(projected_positive_normal)

    dot_product = np.dot(projected_anchor_normal, projected_positive_normal)
    azimuth = np.arccos(dot_product)

    cross_product = np.cross(projected_anchor_normal, projected_positive_normal)
    
    if cross_product[1] > 0:
        azimuth *= -1
    else:
        azimuth *= 1
    
    rotation_matrix = np.array([
        [np.cos(azimuth), 0, np.sin(azimuth)],
        [0, 1, 0],
        [-np.sin(azimuth), 0, np.cos(azimuth)]
    ])

    keypoints = positive_kpts[..., :-1]
    rotated_keypoints = keypoints @ rotation_matrix.T
    positive_kpts[..., :-1] = rotated_keypoints

    return positive_kpts


def create_dataloader(dataset, batch_size, collate_fn, num_workers=0, is_training=False):

    if is_training:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return dataloader


def dataset_collate_fn(batch, form_file_paths_dict, cfg, dataset_type="train"):
    # Number of videos (anchors) in a batch
    batch_size = len(batch)
    
    if cfg.kpts_type_3d=="motionbert":
        from data.rendering import render_keypoints_2d, render_keypoints_2d_for_sample_demo, render_keypoints_3d, render_keypoints_3d_for_sample_demo
        key_01 = "3d"
        key_02 = "2d"
    elif cfg.kpts_type_3d=="mediapipe":
        from data.rendering_mediapipe import render_keypoints_3d, render_keypoints_3d_for_sample_demo
        key_01 = "3d_world"
        key_02 = "3d_pose"
    
    
    # anchors_dict = {f"keypoints_{key_01}":[], f"keypoints_{key_02}":[], "form_ids":[], "camera_ids":[], f"filepath_{key_01}":[]}
    # positives_dict = {f"keypoints_{key_01}":[], f"keypoints_{key_02}":[], "form_ids":[], "camera_ids":[], f"filepath_{key_01}":[]}
    # negatives_dict = {f"keypoints_{key_01}":[], f"keypoints_{key_02}":[], "form_ids":[], "camera_ids":[], f"filepath_{key_01}":[]}

    anchors_dict = {f"keypoints_{key_01}":[], "form_ids":[], "camera_ids":[], f"filepath_{key_01}":[]}
    positives_dict = {f"keypoints_{key_01}":[], "form_ids":[], "camera_ids":[], f"filepath_{key_01}":[]}
    negatives_dict = {f"keypoints_{key_01}":[], "form_ids":[], "camera_ids":[], f"filepath_{key_01}":[]}


    for file_path in batch:
        anchor = file_path
        filename = os.path.basename(file_path)
        anchor_form_id = filename.split("_")[0][1:]
        anchor_camera_id = filename.split("_")[1]

        # select a positive sample
        positive_candidates = [item for item in form_file_paths_dict[anchor_form_id] if item != file_path]
        positive = np.random.choice(positive_candidates)
        filename = os.path.basename(positive)
        positive_form_id = filename.split("_")[0][1:]
        positive_camera_id = filename.split("_")[1]

        # selecting negatives
        pre_candidate = np.random.choice(range(25))
        # There will be 30 types (6 extended types)
        # Since some videos are not complete Tai-Chi movements, additional types are created (only used for training)
        if pre_candidate == 24 and dataset_type=="train":
            form_id_candidates = sorted(list(set([f"{i:02}" for i in range(24,31)]) - {anchor_form_id}))
            if anchor_form_id == "01":
                form_id_candidates.remove("24")
            elif anchor_form_id == "03":
                form_id_candidates.remove("25")
            elif anchor_form_id == "05":
                form_id_candidates.remove("26")
            elif anchor_form_id == "09":
                form_id_candidates.remove("27")
                form_id_candidates.remove("28")
            elif anchor_form_id == "27":
                form_id_candidates.remove("28")
            elif anchor_form_id == "28":
                form_id_candidates.remove("27")
            elif anchor_form_id == "23":
                form_id_candidates.remove("29")
            elif anchor_form_id == "00":
                form_id_candidates.remove("30")

        else:
            form_id_candidates = sorted(list(set([f"{i:02}" for i in range(24)]) - {anchor_form_id}))
            if anchor_form_id == "08":
                form_id_candidates.remove("10")
            elif anchor_form_id == "10":
                form_id_candidates.remove("08")
            elif anchor_form_id == "24":
                form_id_candidates.remove("01")
            elif anchor_form_id == "25":
                form_id_candidates.remove("03")
            elif anchor_form_id == "26":
                form_id_candidates.remove("05")
            elif anchor_form_id == "27":
                form_id_candidates.remove("09")
            elif anchor_form_id == "28":
                form_id_candidates.remove("09")
            elif anchor_form_id == "29":
                form_id_candidates.remove("23")
            elif anchor_form_id == "30":
                form_id_candidates.remove("00")

        negative_form_id = np.random.choice(form_id_candidates)
        negative = np.random.choice(form_file_paths_dict[negative_form_id])

        filename = os.path.basename(negative)
        negative_form_id = filename.split("_")[0][1:]
        negative_camera_id = filename.split("_")[1]

        # anchor = "/home/imlab/Music/All_My_Taichi/datasets/Taichi_Clip/forms_keypoints/02/f02_v00_h14_00199_fps10_3d.npz"
        # positive = "/home/imlab/Music/All_My_Taichi/datasets/Taichi_Clip/forms_keypoints/02/f02_v00_h07_00185_fps10_3d.npz"
        # negative = "/home/imlab/Music/All_My_Taichi/datasets/Taichi_Clip/forms_keypoints/04/f04_v00_h01_00355_fps10_3d.npz"

        anchor_key_01 = np.load(anchor)
        path, filename = os.path.split(anchor)
        # anchor_key_02 = np.load(os.path.join(path, filename.replace(key_01, key_02)))
        anchors_dict[f"keypoints_{key_01}"].append(anchor_key_01["keypoints"])
        # anchors_dict[f"keypoints_{key_02}"].append(anchor_key_02["keypoints"])
        anchors_dict["form_ids"].append(anchor_form_id)
        anchors_dict["camera_ids"].append(anchor_camera_id)
        anchors_dict[f"filepath_{key_01}"].append(anchor)

        positive_key_01 = np.load(positive)
        path, filename = os.path.split(positive)
        # positive_key_02 = np.load(os.path.join(path, filename.replace(key_01, key_02)))
        positives_dict[f"keypoints_{key_01}"].append(positive_key_01["keypoints"])
        # positives_dict[f"keypoints_{key_02}"].append(positive_key_02["keypoints"])
        positives_dict["form_ids"].append(positive_form_id)
        positives_dict["camera_ids"].append(positive_camera_id)
        positives_dict[f"filepath_{key_01}"].append(positive)

        negative_key_01 = np.load(negative)
        path, filename = os.path.split(negative)
        # negative_key_02 = np.load(os.path.join(path, filename.replace(key_01, key_02)))
        negatives_dict[f"keypoints_{key_01}"].append(negative_key_01["keypoints"])
        # negatives_dict[f"keypoints_{key_02}"].append(negative_key_02["keypoints"])
        negatives_dict["form_ids"].append(negative_form_id)
        negatives_dict["camera_ids"].append(negative_camera_id)
        negatives_dict[f"filepath_{key_01}"].append(negative)

    # === Normalize keypoints ===

    # Used for DTW
    anchors_dict[f"normalized_keypoints_{key_01}"] = []
    positives_dict[f"normalized_keypoints_{key_01}"] = []
    negatives_dict[f"normalized_keypoints_{key_01}"] = []

    # The actual input to the model
    anchors_dict[f"normalized_keypoints_first_{key_01}"] = []
    positives_dict[f"normalized_keypoints_first_{key_01}"] = []
    negatives_dict[f"normalized_keypoints_first_{key_01}"] = []

    if cfg.kpts_type_3d=="motionbert":
        for keypoints in anchors_dict[f"keypoints_{key_01}"]:
            normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
            anchors_dict[f"normalized_keypoints_{key_01}"].append(normalized_kpts)
            normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
            anchors_dict[f"normalized_keypoints_first_{key_01}"].append(normalized_kpts)

        for keypoints in positives_dict[f"keypoints_{key_01}"]:
            normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
            positives_dict[f"normalized_keypoints_{key_01}"].append(normalized_kpts)
            normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
            positives_dict[f"normalized_keypoints_first_{key_01}"].append(normalized_kpts)

        for keypoints in negatives_dict[f"keypoints_{key_01}"]:
            normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
            negatives_dict[f"normalized_keypoints_{key_01}"].append(normalized_kpts)
            normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
            negatives_dict[f"normalized_keypoints_first_{key_01}"].append(normalized_kpts)
    
    # if cfg.kpts_type_3d=="mediapipe":
    #     for keypoints in anchors_dict[f"keypoints_{key_01}"]:
    #         normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
    #         anchors_dict[f"normalized_keypoints_{key_01}"].append(normalized_kpts)

    #     for keypoints in positives_dict[f"keypoints_{key_01}"]:
    #         normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
    #         positives_dict[f"normalized_keypoints_{key_01}"].append(normalized_kpts)

    #     for keypoints in negatives_dict[f"keypoints_{key_01}"]:
    #         normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_3d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
    #         negatives_dict[f"normalized_keypoints_{key_01}"].append(normalized_kpts)

    # anchors_dict[f"normalized_keypoints_{key_02}"] = []
    # positives_dict[f"normalized_keypoints_{key_02}"] = []
    # negatives_dict[f"normalized_keypoints_{key_02}"] = []
    # anchors_dict[f"normalized_keypoints_first_{key_02}"] = []
    # positives_dict[f"normalized_keypoints_first_{key_02}"] = []
    # negatives_dict[f"normalized_keypoints_first_{key_02}"] = []

    # for keypoints in anchors_dict[f"keypoints_{key_02}"]:
    #     normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_2d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
    #     anchors_dict[f"normalized_keypoints_{key_02}"].append(normalized_kpts)
    #     normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_2d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
    #     anchors_dict[f"normalized_keypoints_first_{key_02}"].append(normalized_kpts)

    # for keypoints in positives_dict[f"keypoints_{key_02}"]:
    #     normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_2d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
    #     positives_dict[f"normalized_keypoints_{key_02}"].append(normalized_kpts)
    #     normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_2d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
    #     positives_dict[f"normalized_keypoints_first_{key_02}"].append(normalized_kpts)

    # for keypoints in negatives_dict[f"keypoints_{key_02}"]:
    #     normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_2d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="all_frames", scale_type="all_frames")
    #     negatives_dict[f"normalized_keypoints_{key_02}"].append(normalized_kpts)
    #     normalized_kpts = kpts_utils.normalized_keypoints_numpy(keypoints, cfg.kpts_type_2d, raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
    #     negatives_dict[f"normalized_keypoints_first_{key_02}"].append(normalized_kpts)


    # if cfg.kpts_type_3d=="mediapipe":
    #     for idx in range(batch_size):
    #         anchors_dict[f"normalized_keypoints_first_{key_01}"].append(kpts_utils.combine_mediapipe_pose_and_world_numpy(kpts_pose=anchors_dict[f"normalized_keypoints_first_{key_02}"][idx], kpts_world=anchors_dict[f"normalized_keypoints_{key_01}"][idx]))
    #         positives_dict[f"normalized_keypoints_first_{key_01}"].append(kpts_utils.combine_mediapipe_pose_and_world_numpy(kpts_pose=positives_dict[f"normalized_keypoints_first_{key_02}"][idx], kpts_world=positives_dict[f"normalized_keypoints_{key_01}"][idx]))
    #         negatives_dict[f"normalized_keypoints_first_{key_01}"].append(kpts_utils.combine_mediapipe_pose_and_world_numpy(kpts_pose=negatives_dict[f"normalized_keypoints_first_{key_02}"][idx], kpts_world=negatives_dict[f"normalized_keypoints_{key_01}"][idx]))
    
    # # copy is necessary; otherwise, it will affect the following operations
    # if cfg.only_for_sampling_demo and cfg.kpts_type_for_train != "vitpose":
    #     max_length = len(anchors_dict[f"normalized_keypoints_first_{key_01}"][0])
    #     tmp_anchor = extend_or_clip_video(np.copy(anchors_dict[f"normalized_keypoints_first_{key_01}"][0]), max_length)
    #     tmp_positive = extend_or_clip_video(np.copy(positives_dict[f"normalized_keypoints_first_{key_01}"][0]), max_length)
    #     tmp_negative = extend_or_clip_video(np.copy(negatives_dict[f"normalized_keypoints_first_{key_01}"][0]), max_length)
    #     render_keypoints_3d_for_sample_demo("origin_video", tmp_anchor, tmp_positive, tmp_negative, fps_in=cfg.fps)
    
    # if cfg.only_for_sampling_demo and cfg.kpts_type_for_train == "vitpose":
    #     max_length = len(anchors_dict[f"normalized_keypoints_first_{key_02}"][0])
    #     tmp_anchor = extend_or_clip_video(np.copy(anchors_dict[f"normalized_keypoints_first_{key_02}"][0]), max_length)
    #     tmp_positive = extend_or_clip_video(np.copy(positives_dict[f"normalized_keypoints_first_{key_02}"][0]), max_length)
    #     tmp_negative = extend_or_clip_video(np.copy(negatives_dict[f"normalized_keypoints_first_{key_02}"][0]), max_length)
    #     render_keypoints_2d_for_sample_demo("origin_video", tmp_anchor, tmp_positive, tmp_negative, fps_in=cfg.fps)

    # === Normalization ends ===
    
    
    # === Perform Procrustes analysis (rotation) on keypoints ===
    positives_dict[f"rotated_keypoints_{key_01}"] = []
    positives_dict[f"rotated_keypoints_first_{key_01}"] = []

    # For DTW
    for idx in range(batch_size):
        # Align positive with anchor (using Procrustes)
        rotated_kpts = kpts_utils.procrustes_align_keypoints_numpy(kpts_with_score=positives_dict[f"normalized_keypoints_{key_01}"][idx], target_kpts_with_score=anchors_dict[f"normalized_keypoints_{key_01}"][idx], apply_type="first_frame")
        
        # Align positive with anchor (using normal)
        # rotated_kpts = rotation_based_on_normal(anchor_kpts=anchors_dict[f"normalized_keypoints_{key_01}"][idx], positive_kpts=positives_dict[f"normalized_keypoints_{key_01}"][idx], kpts_type=cfg.kpts_type_for_train)
        
        # Do not align positive with anchor
        # rotated_kpts = positives_dict[f"normalized_keypoints_{key_01}"][idx]
        positives_dict[f"rotated_keypoints_{key_01}"].append(rotated_kpts)

    # if cfg.only_for_sampling_demo and cfg.kpts_type_for_train != "vitpose":
    #     if cfg.align_negative_with_anchor:
    #         max_length = len(anchors_dict[f"normalized_keypoints_{key_01}"][0])
    #         tmp_anchor = extend_or_clip_video(np.copy(anchors_dict[f"normalized_keypoints_{key_01}"][0]), max_length)
    #         tmp_positive = extend_or_clip_video(np.copy(positives_dict[f"rotated_keypoints_{key_01}"][0]), max_length)
    #         tmp_negative = extend_or_clip_video(np.copy(negatives_dict[f"rotated_keypoints_{key_01}"][0]), max_length)
    #         render_keypoints_3d_for_sample_demo("rotated_origin_video",tmp_anchor, tmp_positive, tmp_negative, fps_in=cfg.fps)
    #     else:
    #         tmp_anchor = extend_or_clip_video(np.copy(anchors_dict[f"normalized_keypoints_{key_01}"][0]), max_length)
    #         tmp_positive = extend_or_clip_video(np.copy(positives_dict[f"rotated_keypoints_{key_01}"][0]), max_length)
    #         tmp_negative = extend_or_clip_video(np.copy(negatives_dict[f"normalized_keypoints_{key_01}"][0]), max_length)
    #         render_keypoints_3d_for_sample_demo("rotated_origin_video",tmp_anchor, tmp_positive, tmp_negative, fps_in=cfg.fps)

    # === Procrustes analysis (rotation) ends ===


    # === Calculate Frame-to-Frame NP_MPJPEs (Procrustes-aligned Mean Per Joint Position Error) ===
    anchor_positive_dist_matrices = []

    # Used for DTW, with anchor normalized based on all frames, and positive normalized and rotated based on all frames
    for idx in range(batch_size):
        anchor_positive_dist_matrices.append(kpts_utils.compute_sample_to_sample_mpjpe_matrix_numpy(anchors_dict[f"normalized_keypoints_{key_01}"][idx], positives_dict[f"rotated_keypoints_{key_01}"][idx], dist_method=cfg.dtw_dist_method, cfg=cfg))

    # === NP_MPJPEs ends ===


    # === Calculate path using DTW ===
    dtw_path = []
    for idx in range(batch_size):
        dtw_path.append(kpts_utils.dynamic_time_warping_numpy(anchor_positive_dist_matrices[idx]))

    positives_dict[f"rotated_dtw_keypoints_{key_01}"] = []
    anchors_dict[f"normalized_dtw_keypoints_{key_01}"] = []
    positives_dict[f"normalized_dtw_keypoints_{key_01}"] = []
    anchors_dict[f"normalized_dtw_keypoints_{key_02}"] = []
    positives_dict[f"normalized_dtw_keypoints_{key_02}"] = []

    for idx in range(batch_size):
        transposed = np.array(dtw_path[idx]).T
        anchor_frames_path = transposed[0]
        positive_frames_path = transposed[1]

        # positives_dict[f"rotated_dtw_keypoints_{key_01}"].append(positives_dict[f"rotated_keypoints_first_{key_01}"][idx][positive_frames_path])
        anchors_dict[f"normalized_dtw_keypoints_{key_01}"].append(anchors_dict[f"normalized_keypoints_first_{key_01}"][idx][anchor_frames_path])
        positives_dict[f"normalized_dtw_keypoints_{key_01}"].append(positives_dict[f"normalized_keypoints_first_{key_01}"][idx][positive_frames_path])
        # anchors_dict[f"normalized_dtw_keypoints_{key_02}"].append(anchors_dict[f"normalized_keypoints_first_{key_02}"][idx][anchor_frames_path])
        # positives_dict[f"normalized_dtw_keypoints_{key_02}"].append(positives_dict[f"normalized_keypoints_first_{key_02}"][idx][positive_frames_path])
        negatives_dict[f"normalized_keypoints_{key_01}"][idx] = negatives_dict[f"normalized_keypoints_first_{key_01}"][idx]
        # negatives_dict[f"normalized_keypoints_{key_02}"][idx] = negatives_dict[f"normalized_keypoints_first_{key_02}"][idx]
        
    
    if cfg.only_for_sampling_demo and cfg.kpts_type_for_train != "vitpose":
        # just want to print the similarity on the video
        frame_to_frame_similarity = []
        frame_to_frame_similarity_negative = []
        
        frame_to_frame_similarity.append(1.0 - kpts_utils.compute_sample_to_sample_mpjpe_sequence_numpy(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][0], positives_dict[f"rotated_dtw_keypoints_{key_01}"][0], dist_method=cfg.similarity_dist_method, cfg=cfg))

        tmp_negative = extend_or_clip_video(np.copy(negatives_dict[f"normalized_keypoints_{key_01}"][0]), len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][0]))
        frame_to_frame_similarity_negative.append(1.0 - kpts_utils.compute_sample_to_sample_mpjpe_sequence_numpy(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][0], tmp_negative, dist_method=cfg.similarity_dist_method, cfg=cfg))
        
        render_keypoints_3d_for_sample_demo("dtw_origin_video", np.copy(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][0]), np.copy(positives_dict[f"rotated_dtw_keypoints_{key_01}"][0]), tmp_negative, frame_to_frame_similarity[0], frame_to_frame_similarity_negative[0], fps_in=cfg.fps)
    
    # === dtw ends ===


    # === Slicing clips ===
    anchors_clips = [[] for i in range(batch_size)]
    positives_clips = [[] for i in range(batch_size)]
    negatives_clips = [[] for i in range(batch_size)]

    # anchors_clips_for_similarity = [[] for i in range(batch_size)]
    # positives_clips_for_similarity = [[] for i in range(batch_size)]
    # negatives_clips_for_similarity = [[] for i in range(batch_size)]

    for batch_idx in range(batch_size):

        if len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]) < cfg.clip_len:
            new_indices = np.linspace(0, len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]) - 1, cfg.clip_len)
            new_indices = np.round(new_indices).astype('int32')
            anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx] = anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx][new_indices]

        if len(positives_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]) < cfg.clip_len:
            new_indices = np.linspace(0, len(positives_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]) - 1, cfg.clip_len)
            new_indices = np.round(new_indices).astype('int32')
            positives_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx] = positives_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx][new_indices]

        if len(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx]) < cfg.clip_len:
            new_indices = np.linspace(0, len(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx]) - 1, cfg.clip_len)
            new_indices = np.round(new_indices).astype('int32')
            negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx] = negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx][new_indices]


        for frame_idx in range(0, len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]) - cfg.clip_len + 1, cfg.sliding_window_sample_step):
            if frame_idx+cfg.clip_len > len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]):
                anchors_clips[batch_idx].append(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx][range(len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx])-cfg.sliding_window_sample_step, len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]))])
                positives_clips[batch_idx].append(positives_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx][range(len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx])-cfg.sliding_window_sample_step, len(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx]))])
            else:
                anchors_clips[batch_idx].append(anchors_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx][range(frame_idx, frame_idx+cfg.clip_len)])
                positives_clips[batch_idx].append(positives_dict[f"normalized_dtw_keypoints_{key_01}"][batch_idx][range(frame_idx, frame_idx+cfg.clip_len)])

        for frame_idx in range(0, len(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx]) - cfg.clip_len + 1, cfg.sliding_window_sample_step):
            if frame_idx+cfg.clip_len > len(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx]):
                positives_clips[batch_idx].append(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx][range(len(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx])-cfg.sliding_window_sample_step, len(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx]))])
            else:
                negatives_clips[batch_idx].append(negatives_dict[f"normalized_keypoints_{key_01}"][batch_idx][range(frame_idx, frame_idx+cfg.clip_len)])
       
    apn_pairs = []
    # apn_pairs_for_similarity = []

    anchor_form = []
    negative_form = []    

    # negative clips sampling 
    for batch_idx in range(batch_size):
        if cfg.negative_sample_method == "continuous_with_random_start":
            random_start_clip_idx = np.random.choice(range(len(negatives_clips[batch_idx])),1,replace=False)[0]
        else:
            random_start_clip_idx = 0
        
        for anchor_clip_idx in range(len(anchors_clips[batch_idx])):
            if cfg.negative_sample_method == "random":
                selected_clip_idx = np.random.choice(range(len(negatives_clips[batch_idx])), 1, replace=False)[0]
            elif cfg.negative_sample_method == "continuous" or cfg.negative_sample_method == "continuous_with_random_start":
                selected_clip_idx = (anchor_clip_idx + random_start_clip_idx)%len(negatives_clips[batch_idx])
            apn_pairs.append(np.stack((anchors_clips[batch_idx][anchor_clip_idx], positives_clips[batch_idx][anchor_clip_idx], negatives_clips[batch_idx][selected_clip_idx]), axis=0))
            

            anchor_form.append(int(anchors_dict["form_ids"][batch_idx]))
            negative_form.append(int(negatives_dict["form_ids"][batch_idx]))

            # apn_pairs_for_similarity.append(np.stack((anchors_clips_for_similarity[batch_idx][anchor_clip_idx], positives_clips_for_similarity[batch_idx][anchor_clip_idx], negatives_clips_for_similarity[batch_idx][selected_clip_idx]), axis=0))

    # === Slicing clips ends ===


    # === Normalize each clip again ===
    for idx in range(len(apn_pairs)):
        apn_pairs[idx][0] = kpts_utils.normalized_keypoints_numpy(apn_pairs[idx][0], cfg.kpts_type_for_train, raw=False, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
        apn_pairs[idx][1] = kpts_utils.normalized_keypoints_numpy(apn_pairs[idx][1], cfg.kpts_type_for_train, raw=False, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
        apn_pairs[idx][2] = kpts_utils.normalized_keypoints_numpy(apn_pairs[idx][2], cfg.kpts_type_for_train, raw=False, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
        
        # Should the model input be rotated?
        # rotate using Procrustes
        # apn_pairs[idx][1]  = kpts_utils.procrustes_align_keypoints_numpy(kpts_with_score=apn_pairs[idx][1] , target_kpts_with_score=apn_pairs[idx][0] , apply_type="first_frame")
        # rotate using Normal
        # apn_pairs[idx][1] = rotation_based_on_normal(anchor_kpts=apn_pairs[idx][0], positive_kpts=apn_pairs[idx][1], kpts_type=cfg.kpts_type_for_train)
        
    # === normalize ends ===

    ap_similarity = []
    an_similarity = []

    for idx in range(len(apn_pairs)):
        ap_similarity.append(1.0 - kpts_utils.compute_sample_to_sample_mpjpe_sequence_numpy(apn_pairs[idx][0], apn_pairs[idx][1], dist_method=cfg.similarity_dist_method, cfg=cfg, temporal_reduction="mean"))
        # only use an_similarity
        an_similarity.append(1.0 - kpts_utils.compute_sample_to_sample_mpjpe_sequence_numpy(apn_pairs[idx][0], apn_pairs[idx][2], dist_method=cfg.similarity_dist_method, cfg=cfg, temporal_reduction="mean"))


    # === joint, motion, bone, motion of bone ===

    apn_pairs_with_multiple_data_views = []

    for idx in range(len(apn_pairs)):
        anchors_data_views_tuple = []
        positives_data_views_tuple = []
        negatives_data_views_tuple = []

        if "joint" in cfg.data_type_views:
            anchors_data_views_tuple.append(apn_pairs[idx][0])
            positives_data_views_tuple.append(apn_pairs[idx][1])
            negatives_data_views_tuple.append(apn_pairs[idx][2])

        if "angle" in cfg.data_type_views:
            anchors_data_views_tuple.append(kpts_utils.get_angle_from_bone(apn_pairs[idx][0], "motionbert"))
            positives_data_views_tuple.append(kpts_utils.get_angle_from_bone(apn_pairs[idx][1], "motionbert"))
            negatives_data_views_tuple.append(kpts_utils.get_angle_from_bone(apn_pairs[idx][2], "motionbert"))

        if cfg.clip_len > 1 and "motion" in cfg.data_type_views:
            anchors_data_views_tuple.append(kpts_utils.to_motion(apn_pairs[idx][0]))
            positives_data_views_tuple.append(kpts_utils.to_motion(apn_pairs[idx][1]))
            negatives_data_views_tuple.append(kpts_utils.to_motion(apn_pairs[idx][2]))
        
        if "bone" in cfg.data_type_views:
            tmp_anchor_bone = kpts_utils.joint_to_bone(apn_pairs[idx][0], cfg.kpts_type_for_train)
            tmp_positive_bone = kpts_utils.joint_to_bone(apn_pairs[idx][1], cfg.kpts_type_for_train)
            tmp_negative_bone = kpts_utils.joint_to_bone(apn_pairs[idx][2], cfg.kpts_type_for_train)

            anchors_data_views_tuple.append(tmp_anchor_bone)
            positives_data_views_tuple.append(tmp_positive_bone)
            negatives_data_views_tuple.append(tmp_negative_bone)

        if cfg.clip_len > 1 and "bone" in cfg.data_type_views and "motion_bone" in cfg.data_type_views:
            anchors_data_views_tuple.append(kpts_utils.to_motion(tmp_anchor_bone))
            positives_data_views_tuple.append(kpts_utils.to_motion(tmp_positive_bone))
            negatives_data_views_tuple.append(kpts_utils.to_motion(tmp_negative_bone))   
        elif cfg.clip_len > 1 and "motion_bone" in cfg.data_type_views:
            tmp_anchor_bone = kpts_utils.joint_to_bone(apn_pairs[idx][0], cfg.kpts_type_for_train)
            tmp_positive_bone = kpts_utils.joint_to_bone(apn_pairs[idx][1], cfg.kpts_type_for_train)
            tmp_negative_bone = kpts_utils.joint_to_bone(apn_pairs[idx][2], cfg.kpts_type_for_train)
            anchors_data_views_tuple.append(kpts_utils.to_motion(tmp_anchor_bone))
            positives_data_views_tuple.append(kpts_utils.to_motion(tmp_positive_bone))
            negatives_data_views_tuple.append(kpts_utils.to_motion(tmp_negative_bone))   

        
        anchors_data_views = kpts_utils.combine_mutiple_data_view(anchors_data_views_tuple)
        positives_data_views = kpts_utils.combine_mutiple_data_view(positives_data_views_tuple)
        negatives_data_views = kpts_utils.combine_mutiple_data_view(negatives_data_views_tuple)

        # The joints must be retained, otherwise, it will not be possible to draw the skeleton later
        # Adding them here is to avoid affecting the score
        if "joint" not in cfg.data_type_views:
            T, V, C = anchors_data_views.shape
            new_anchors_data_views = np.empty((T, V, C+3))
            new_positives_data_views = np.empty((T, V, C+3))
            new_negatives_data_views = np.empty((T, V, C+3))
            new_anchors_data_views[..., :3] = apn_pairs[idx][0][..., :-1]
            new_positives_data_views[..., :3] = apn_pairs[idx][1][..., :-1]
            new_negatives_data_views[..., :3] = apn_pairs[idx][2][..., :-1]
            new_anchors_data_views[..., 3:] = anchors_data_views
            new_positives_data_views[..., 3:] = positives_data_views
            new_negatives_data_views[..., 3:] = negatives_data_views
            apn_data_views = np.stack((new_anchors_data_views, new_positives_data_views, new_negatives_data_views), axis=0)
        else:
            apn_data_views = np.stack((anchors_data_views, positives_data_views, negatives_data_views), axis=0)
        
        apn_pairs_with_multiple_data_views.append(apn_data_views)

    # shape: (final_batch_size, apn, clip_len, kpts_size, channel_size)
    apn_pairs_with_multiple_data_views = np.array(apn_pairs_with_multiple_data_views)
    
    # === joint, motion, bone, motion of bone ends ===
    
    if cfg.only_for_sampling_demo:
        if cfg.kpts_type_for_train == "motionbert":
            for idx in range(len(anchors_clips[0])):
                tmp_clip_len = len(apn_pairs_with_multiple_data_views[idx,0])
                render_keypoints_3d_for_sample_demo("input_video", apn_pairs_with_multiple_data_views[idx,0][..., [0,1,2,-1]], apn_pairs_with_multiple_data_views[idx,1][..., [0,1,2,-1]], apn_pairs_with_multiple_data_views[idx,2][..., [0,1,2,-1]], np.array([ap_similarity[idx]]*tmp_clip_len), np.array([an_similarity[idx]]*tmp_clip_len), fps_in=cfg.fps)
    
    if cfg.kpts_type_for_train=="motionbert":
        return torch.FloatTensor(apn_pairs_with_multiple_data_views), torch.FloatTensor(ap_similarity), torch.FloatTensor(an_similarity), torch.IntTensor(anchor_form), torch.IntTensor(negative_form)

