import torch
import numpy as np 
import os
from functools import partial 
import warnings

def get_sample_to_sample_matrix_numpy(kpts_for_rows, kpts_for_columns):
    
    # Prevent modifying the original kpts_for_rows
    kpts_for_rows = kpts_for_rows.copy()
    kpts_for_columns = kpts_for_columns.copy()

    kpts_for_rows = kpts_for_rows[:, np.newaxis, ...]
    kpts_for_columns = kpts_for_columns[np.newaxis, ...]
    # kpts_for_rows shape: (sample_size_1, 1, ...)
    # kpts_for_columns shape: (1, sample_size_2, ...)

    kpts_for_rows_shape = list(kpts_for_rows.shape)
    kpts_for_columns_shape = list(kpts_for_columns.shape)
    kpts_for_rows_shape[1] = kpts_for_columns_shape[1]
    kpts_for_columns_shape[0] = kpts_for_rows_shape[0]
    # np.tile replicates to expand, while np.broadcast_to just creates a view
    kpts_for_rows = np.broadcast_to(kpts_for_rows, tuple(kpts_for_rows_shape))
    kpts_for_columns = np.broadcast_to(kpts_for_columns, tuple(kpts_for_columns_shape))
    # kpts_for_rows shape: (sample_size_1, sample_size_2, ...)
    # kpts_for_columns shape: (sample_size_1, sample_size_2, ...)

    matrix_prototype = np.stack((kpts_for_rows, kpts_for_columns), axis=-1)
    # matrix_prototype shape: (sample_size_1, sample_size_2, ..., 2)

    # # Required only if frame_size is present
    # if len(matrix_prototype.shape) > 5:
    #     for row in range(len(matrix_prototype)):
    #         matrix_prototype[row, ..., 1] = procrustes_align_keypoints_numpy(matrix_prototype[row, ..., 1], matrix_prototype[row, ..., 0], "first_frame")

    return matrix_prototype


def normalized_keypoints_numpy(kpts_with_score: np.ndarray, keypoint_type="vitpose", raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="first_frame"):
    
    if keypoint_type == "vitpose":
        valid_indices = np.array(list(range(0, 23)) + [95, 96, 99, 108, 111, 116, 117, 120, 129, 132] + [0, 0])
    elif keypoint_type == "mediapipe":
        valid_indices = np.array([0,2,5,7,8] + list(range(11, 33)) + [0, 0])
    elif keypoint_type == "motionbert":
        valid_indices = np.array(list(range(0, 17)))
    
    # kpts_with_score shape: (keypoints_size, channel_size,)
    if raw:
        kpts_with_score = kpts_with_score[... ,valid_indices,:]

        if keypoint_type == "vitpose":
            kpts_with_score[..., -2, :] = (kpts_with_score[..., 5, :] + kpts_with_score[..., 6, :])*0.5
            kpts_with_score[..., -1, :] = (kpts_with_score[..., 11, :] + kpts_with_score[..., 12, :])*0.5
        elif keypoint_type == "mediapipe":
            kpts_with_score[..., -2, :] = (kpts_with_score[..., 5, :] + kpts_with_score[..., 6, :])*0.5
            kpts_with_score[..., -1, :] = (kpts_with_score[..., 17, :] + kpts_with_score[..., 18, :])*0.5

    if only_raw:
        return kpts_with_score

    # Do not take score (or visibility/confidence)
    # kpts shape: (batch_size, frame_size, keypoints_size, without_score_channel_size,)
    kpts = kpts_with_score[..., :-1]

    offsets = None 
    scales = None 
    
    if keypoint_type == "vitpose":
        # hip_kpt shape: (batch_size, frame_size, without_score_channel_size,)
        thorax_kpt = kpts[..., -2, :]
        hip_kpt = kpts[..., -1, :]
        
    elif keypoint_type == "motionbert":
        # hip_kpt shape: (batch_size, frame_size, without_score_channel_size,)
        thorax_kpt = kpts[..., 8, :]
        hip_kpt = kpts[..., 0, :]

    elif keypoint_type == "mediapipe":
        # hip_kpt shape: (batch_size, frame_size, without_score_channel_size,)
        thorax_kpt = kpts[..., -2, :]
        hip_kpt = kpts[..., -1, :]

    # thorax_to_hip_dist shape: (batch_size, frame_size,)
    thorax_to_hip_dist = np.linalg.norm(thorax_kpt-hip_kpt, ord=2, axis=-1)
    # In pr_vipe, multiplying by 2 ensures that the thorax-to-hip distance is 0.5
    # max_distance shape: (batch_size, frame_size,)
    thorax_to_hip_dist *= 2

    if offset_type == "first_frame":
        offsets = (kpts - hip_kpt[..., [0], np.newaxis, :])
    elif offset_type == "all_frames":
        offsets = (kpts - hip_kpt[..., np.newaxis, :])
    else:
        raise ValueError(f'Do Not Exist This offset_type: {offset_type}')

    if scale_type == "first_frame":
        scales = (thorax_to_hip_dist[..., [0], np.newaxis, np.newaxis] + 1e-6)
    elif scale_type == "all_frames":
        scales = (thorax_to_hip_dist[..., np.newaxis, np.newaxis] + 1e-6)
    else:
        raise ValueError(f'Do Not Exist This scale_type: {scale_type}')


    if only_offset:
        normalized_kpts = offsets
    
    elif only_scale:
        normalized_kpts = normalized_kpts / scales

    else:
        normalized_kpts = offsets / scales

    normalized_kpts[np.isnan(normalized_kpts)] = 1e-6
    kpts_with_score[..., :-1] = normalized_kpts

    return kpts_with_score


def combine_mediapipe_pose_and_world_numpy(kpts_pose, kpts_world, replace=False):
    if replace:
        kpts_world[..., 0:-2] = kpts_pose[..., 0:-2]
    else:
        hip_kpt_pose = kpts_pose[:, -1, :-2]
        hip_kpt_world = kpts_world[:, -1, :-2]
        displacement = (hip_kpt_pose-hip_kpt_world)[..., np.newaxis, :]
        kpts_world[..., :-2] += displacement
    return kpts_world


# This sample_to_sample might be frame_to_frame or anchorclip_to_anotherclip
def compute_sample_to_sample_mpjpe_matrix_numpy(anchor_kpts_with_score: np.ndarray, matched_kpts_with_score: np.ndarray, dist_method="l2_dist", cfg=None, temporal_reduction="mean", threshold=0.3):
    # dist_matrix shape: (anchor_frame_size, matched_frame_size, keypoint_size, channel_size, 2,)
    dist_matrix = get_sample_to_sample_matrix_numpy(anchor_kpts_with_score, matched_kpts_with_score)
    # If the scores of keypoints are too low, the distance between them is not calculated
    # only consider visible keypoints
    mask = np.logical_and(dist_matrix[..., -1, 0] > threshold, dist_matrix[..., -1, 1] > threshold)   
    dist_matrix[~mask] = 0.0
    # binarize (matrix contains only 0 and 1)
    dist_matrix[..., -1, :] = (dist_matrix[..., -1, :] > threshold).astype(np.float32)

    # score_matrix shape: (anchor_frame_size, matched_frame_size, keypoint_size,)
    score_matrix = dist_matrix[..., -1, 0]*dist_matrix[..., -1, 1]
    num_visible_kpts = np.sum(score_matrix, axis=-1)

    if dist_method == "l2_dist":
        # np.linalg.norm(...) shape: (anchor_frame_size, matched_frame_size, keypoint_size,)
        # np.sum(...) shape: (anchor_frame_size, matched_frame_size, )
        
        tmp_l2 = np.linalg.norm(dist_matrix[..., :-1, 0] - dist_matrix[..., :-1, 1], ord=2, axis=-1)
        if cfg.use_score_weight:
            # Multiply by score, to prevent keypoints with low visibility from influencing the overall distance
            # The sum here is for keypoints, not for frames
            # Then calculate the average based on the number of keypoints
            tmp_l2 = tmp_l2 * score_matrix
            dist_matrix = np.sum(tmp_l2, axis=-1) / (np.sum(score_matrix, axis=-1) + 1e-6)
        else:
            dist_matrix = np.mean(tmp_l2, axis=-1)

        dist_matrix[np.isnan(dist_matrix)] = 1.0
        dist_matrix = np.clip(dist_matrix, a_min=0, a_max=1)

        if cfg.use_visibility_weight:
            visibility_weight = np.log(np.clip((1 + (num_visible_kpts - 1) * 10), a_min=1, a_max=None)) / np.log(1+anchor_kpts_with_score.shape[-2]*10)
            dist_matrix = np.clip(dist_matrix * (2.0-visibility_weight), a_min=0, a_max=1)
        # The above tmp_sum_matrix is still distance (is it necessary to convert to similarity?)
        # Considering that DTW requires distance, it seems unnecessary
        # If forced to convert, consider:
        # Simple 1.0 - distance
        # tmp_sum_matrix = 1.0 - tmp_sum_matrix
        # Or reciprocal conversion
        # tmp_sum_matrix = 1.0 / (1.0 + tmp_sum_matrix)

    # Radial Basis Function, RBF
    elif dist_method == "rbf":
        gamma = cfg.rbf_gamma

        tmp_rbf = np.exp(-gamma * np.sum(np.square(dist_matrix[..., :-1, 0] - dist_matrix[..., :-1, 1]), axis=-1))
        if cfg.use_score_weight:
            tmp_rbf = tmp_rbf * score_matrix
            similarity = np.sum(tmp_rbf, axis=-1) / (np.sum(score_matrix, axis=-1) + 1e-6)
        else:
            similarity = np.mean(tmp_rbf, axis=-1)

        if cfg.use_visibility_weight:
            # The more visible keypoints, the higher the similarity
            # (num_visible_kpts - 1) is used because we assume num_visible_kpts will be at least 1
            # Therefore, 1 + (1-1)*10 = 1, and log(1) is 0
            # If all 17 keypoints are visible, the weight is 1, and the fewer keypoints visible, the smaller the weight
            visibility_weight = np.log(np.clip((1 + (num_visible_kpts - 1) * 10), a_min=1, a_max=None)) / np.log(1+anchor_kpts_with_score.shape[-2]*10)
            similarity = similarity * visibility_weight
        similarity[np.isnan(similarity)] = 0
        dist_matrix = np.clip(1.0 - similarity, a_min=0, a_max=1)

    elif dist_method == "cosine":
        tmp_cos = np.sum(dist_matrix[..., :-1, 0] * dist_matrix[..., :-1, 1], axis=-1) / (np.linalg.norm(dist_matrix[..., :-1, 0], axis=-1) * np.linalg.norm(dist_matrix[..., :-1, 1], axis=-1) + 1e-6)
        if cfg.use_score_weight:
            tmp_cos = tmp_cos * score_matrix
            similarity = np.sum(tmp_cos, axis=-1) / (np.sum(score_matrix, axis=-1) + 1e-6)
        else:
            similarity = np.mean(tmp_cos, axis=-1)
        
        if cfg.use_visibility_weight:
            visibility_weight = np.log(np.clip((1 + (num_visible_kpts - 1) * 10), a_min=1, a_max=None)) / np.log(1+anchor_kpts_with_score.shape[-2]*10)
            similarity = similarity * visibility_weight
        similarity[np.isnan(similarity)] = 0
        dist_matrix = np.clip(1.0 - similarity, a_min=0, a_max=1)
    
    if len(dist_matrix.shape) > 2:
        if temporal_reduction == "mean":
            return np.mean(dist_matrix, axis=-1)
        elif temporal_reduction == "sum":
            return np.sum(dist_matrix, axis=-1)        

    return dist_matrix
    


def compute_sample_to_sample_mpjpe_sequence_numpy(anchor_kpts_with_score: np.ndarray, matched_kpts_with_score: np.ndarray, dist_method="l2_dist", cfg=None, temporal_reduction=None, threshold=0.3):
    dist_sequence = np.stack((anchor_kpts_with_score, matched_kpts_with_score), axis=-1)
    # only consider visible keypoints
    mask = np.logical_and(dist_sequence[..., -1, 0] > threshold, dist_sequence[..., -1, 1] > threshold)   
    dist_sequence[~mask] = 0.0
    # binarize (matrix contains only 0 and 1)
    dist_sequence[..., -1, :] = (dist_sequence[..., -1, :] > threshold).astype(np.float32)

    # score_matrix shape: (anchor_frame_size, matched_frame_size, keypoint_size,)
    score_matrix = dist_sequence[..., -1, 0]*dist_sequence[..., -1, 1]
    num_visible_kpts = np.sum(score_matrix, axis=-1)

    if dist_method == "l2_dist":
        # np.linalg.norm(...) shape: (anchor_frame_size, matched_frame_size, keypoint_size,)
        # np.sum(...) shape: (anchor_frame_size, matched_frame_size, )
        
        tmp_l2 =  np.linalg.norm(dist_sequence[..., :-1, 0] - dist_sequence[..., :-1, 1], ord=2, axis=-1)
        if cfg.use_score_weight:
            # Multiply by score, to prevent keypoints with low visibility from influencing the overall distance
            # The sum here is for keypoints, not for frames
            # Then calculate the average based on the number of keypoints
            tmp_l2 = tmp_l2 * score_matrix
            dist_sequence = np.sum(tmp_l2, axis=-1) / (np.sum(score_matrix, axis=-1) + 1e-6)
        else:
            dist_sequence = np.mean(tmp_l2, axis=-1)

        dist_sequence[np.isnan(dist_sequence)] = 1.0
        dist_sequence = np.clip(dist_sequence, a_min=0, a_max=1)

        if cfg.use_visibility_weight:
            visibility_weight = np.log(np.clip((1 + (num_visible_kpts - 1) * 10), a_min=1, a_max=None)) / np.log(1+anchor_kpts_with_score.shape[-2]*10)
            dist_sequence = np.clip(dist_sequence * (2.0-visibility_weight), a_min=0, a_max=1)
        # The above tmp_sum_matrix is still distance (is it necessary to convert to similarity?)
        # Considering that DTW requires distance, it seems unnecessary
        # If forced to convert, consider:
        # Simple 1.0 - distance
        # tmp_sum_matrix = 1.0 - tmp_sum_matrix
        # Or reciprocal conversion
        # tmp_sum_matrix = 1.0 / (1.0 + tmp_sum_matrix)

    # Radial Basis Function, RBF
    elif dist_method == "rbf":
        gamma = cfg.rbf_gamma
        tmp_rbf = np.exp(-gamma * np.sum(np.square(dist_sequence[..., :-1, 0] - dist_sequence[..., :-1, 1]), axis=-1))
        if cfg.use_score_weight:
            tmp_rbf = tmp_rbf * score_matrix
            similarity = np.sum(tmp_rbf, axis=-1) / (np.sum(score_matrix, axis=-1) + 1e-6)
        else:
            similarity = np.mean(tmp_rbf, axis=-1)

        if cfg.use_visibility_weight:
            # The more visible keypoints, the higher the similarity
            # (num_visible_kpts - 1) is used because we assume num_visible_kpts will be at least 1
            # Therefore, 1 + (1-1)*10 = 1, and log(1) is 0
            # If all 17 keypoints are visible, the weight is 1, and the fewer keypoints visible, the smaller the weight
            visibility_weight = np.log(np.clip((1 + (num_visible_kpts - 1) * 10), a_min=1, a_max=None)) / np.log(1+anchor_kpts_with_score.shape[-2]*10)
            similarity = similarity * visibility_weight
        similarity[np.isnan(similarity)] = 0
        dist_sequence = np.clip(1.0 - similarity, a_min=0, a_max=1)
       
    elif dist_method == "cosine":
        tmp_cos = np.sum(dist_sequence[..., :-1, 0] * dist_sequence[..., :-1, 1], axis=-1) / (np.linalg.norm(dist_sequence[..., :-1, 0], axis=-1) * np.linalg.norm(dist_sequence[..., :-1, 1], axis=-1) + 1e-6)
        if cfg.use_score_weight:
            tmp_cos = tmp_cos * score_matrix
            similarity = np.sum(tmp_cos, axis=-1) / (np.sum(score_matrix, axis=-1) + 1e-6)
        else:
            similarity = np.mean(tmp_cos, axis=-1)

        if cfg.use_visibility_weight:
            visibility_weight = np.log(np.clip((1 + (num_visible_kpts - 1) * 10), a_min=1, a_max=None)) / np.log(1+anchor_kpts_with_score.shape[-2]*10)
            similarity = similarity * visibility_weight
        similarity[np.isnan(similarity)] = 0
        similarity = (similarity+1)/2
        dist_sequence = np.clip(1.0 - similarity, a_min=0, a_max=1)

    if temporal_reduction == "mean":
        dist_sequence = np.mean(dist_sequence)  

    return dist_sequence



def procrustes_align_keypoints_numpy(kpts_with_score: np.ndarray, target_kpts_with_score: np.ndarray, apply_type="first_frame"):
    # kpts are already normalized

    def get_rotation_matrix(source_kpts: np.ndarray, target_kpts: np.ndarray):
        target_kpts_shape = list(range(len(target_kpts.shape)))
       
        target_kpts_shape[-2], target_kpts_shape[-1] = target_kpts_shape[-1], target_kpts_shape[-2]
        covariance = np.transpose(target_kpts, tuple(target_kpts_shape)) @ source_kpts

        U, S, Vt = np.linalg.svd(covariance)
        R = U @ Vt
        
        det_R = np.linalg.det(R)
        signs = np.sign(det_R)
        #print(np.sign(np.linalg.det(np.transpose(U,tuple(target_kpts_shape)) @ np.transpose(Vt,tuple(target_kpts_shape)))) == signs)
        signs_shape = list(signs.shape)
        diag_template = np.eye(covariance.shape[-2],covariance.shape[-1])
        diag = np.tile(diag_template, tuple(signs_shape+[1,1]))
        diag[..., S.shape[-1]-1, S.shape[-1]-1] = signs

        R = U @ diag @ Vt

        return R, target_kpts_shape

    # prevent modification of the original kpts_with_score
    kpts_with_score = kpts_with_score.copy()
    target_kpts_with_score = target_kpts_with_score.copy()

    kpts = kpts_with_score[...,:-1]
    target_kpts = target_kpts_with_score[...,:-1]   
    new_indices = None

    if apply_type == "first_frame":
        # only use the first frame; rotate all subsequent frames with this R
        R, target_kpts_shape = get_rotation_matrix(kpts[..., :1, :, :], target_kpts[..., :1, :, :])
    
    elif apply_type == "all_frames":
        if(len(kpts) != len(target_kpts)):
            new_indices = np.linspace(0, len(kpts) - 1, len(target_kpts))
            new_indices = np.round(new_indices).astype('int32') 
            kpts = kpts[new_indices]
            kpts_with_score = kpts_with_score[new_indices]

        R, target_kpts_shape = get_rotation_matrix(kpts, target_kpts)
    else:
        raise ValueError(f'Do Not Exist This apply_type: {apply_type}')


    # if we only want to use y as the rotation axis
    R[..., 1] = [0, 1, 0]
    R[..., 1, :] = [0, 1, 0]
    kpts_transformed = R @ np.transpose(kpts, tuple(target_kpts_shape))
    kpts_with_score[..., :-1]  = np.transpose(kpts_transformed, tuple(target_kpts_shape))
    
    return kpts_with_score


def dynamic_time_warping_numpy(distance):

    def trace_back_dtw_from_end(dtw):
        i = dtw.shape[0] - 2
        j = dtw.shape[1] - 2
        path = [(i, j)]  # End point
        # Traceback
        while i > 0 and j > 0:
            index = np.argmin(np.array([dtw[i, j], dtw[i, j+1], dtw[i+1, j]]))
            if index == 0:
                i -= 1
                j -= 1
            elif index == 1:
                i -= 1
            else:
                j -= 1
            path.append((i, j))

        path.reverse()
        return path

    # m is the length of the anchor video, n is the length of the matched video
    m, n = distance.shape

    # Initialize the DTW matrix with infinity on the first row and first column
    dtw_0 = np.zeros((m+1, n+1))
    dtw_0[0, 1:] = np.inf
    dtw_0[1:, 0] = np.inf

    # This is a shallow copy in numpy, which means that changes in dtw_1 affect dtw_0
    dtw_1 = dtw_0[1:, 1:]
    # values in dtw_0[1:,1:] are replaced with the values from dtw_1
    dtw_1[:, :] = distance

    for i in range(m):
        for j in range(n):
            dtw_1[i, j] += np.min(np.array([dtw_0[i, j], dtw_0[i, j+1], dtw_0[i+1, j]]))

    return trace_back_dtw_from_end(dtw_0)


def get_sample_to_sample_matrix_tensor(kpts_for_rows, kpts_for_columns):

    kpts_for_rows = kpts_for_rows.clone()
    kpts_for_columns = kpts_for_columns.clone()
    kpts_for_rows = kpts_for_rows[:, np.newaxis, ...]
    kpts_for_columns = kpts_for_columns[np.newaxis, ...]
    kpts_for_rows_shape = list(kpts_for_rows.shape)
    kpts_for_columns_shape = list(kpts_for_columns.shape)
    kpts_for_rows_shape[1] = kpts_for_columns_shape[1]
    kpts_for_columns_shape[0] = kpts_for_rows_shape[0]
    kpts_for_rows = torch.broadcast_to(kpts_for_rows, tuple(kpts_for_rows_shape))
    kpts_for_columns = torch.broadcast_to(kpts_for_columns, tuple(kpts_for_columns_shape))
    matrix_prototype = torch.stack((kpts_for_rows, kpts_for_columns), dim=-1)

    return matrix_prototype


def to_motion(data):
    motions = np.zeros_like(data)
    # Along the frame_size axis
    # np.diff is out[i] = a[i+1] - a[i]
    # The final :-1 is just to avoid including it in the score
    # The first :-1 is for the frame dimension
    # motions[:-1, :, :-1] = np.diff(data[:,:,:-1], axis=0)  # Better to have zero at the end ?
    motions[1:, :, :-1] = np.diff(data[:,:,:-1], axis=0)    # Better to have zero at the beginning ?
    # score
    motions[:-1, :, -1] = 0.5*(data[:-1, :, -1]+data[1:, :, -1])
    # For the last frame, since there is no next frame to compute
    # I think its score can remain unchanged
    motions[-1, :, -1] = data[-1, :, -1]
    return motions


def calculate_angles(vec1, vec2):
    # Shape of vec1 is (frames_size, kpts_size, channels_size)
    # Calculate dot product
    dot_product = np.sum(vec1 * vec2, axis=-1)
    
    # Calculate norms
    norm_vec1 = np.linalg.norm(vec1, axis=-1)
    norm_vec2 = np.linalg.norm(vec2, axis=-1)

    norm_vec1[norm_vec1 == 0] = 100
    norm_vec2[norm_vec2 == 0] = 100
        
    cos_theta = np.where(np.logical_or(norm_vec1 == 100, norm_vec2 == 100), 1, dot_product / (norm_vec1 * norm_vec2))

    # Constrain cos_theta between -1 and 1
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate angles (in radians)
    angles = np.arccos(cos_theta)
    
    return angles


def get_angle_from_bone(kpts, kpts_type):
    
    # kpts的shape (frames_size, kpts_size, channels_size)
    angles = np.zeros_like(kpts[..., :2])
    if kpts_type == "motionbert":
        # (source_kpts, target_kpts, source_kpts2, target_kpts2)
        pairs = [[(0,7), (0,4)], [(1,0), (1,2)], [(2,1), (2,3)], [(3,3), None], [(4,0), (4,5)], [(5,4), (5,6)],
                 [(6,6), None], [(7,0), (7,8)], [(8,7), (8,11)], [(9,8), (9,10)], [(10,10), None], [(11,8), (11,12)], [(12,11), (12,13)],
                 [(13,13), None], [(14,8), (14,15)] ,[(15,14), (15,16)], [(16,16), None]]

    vecs1 = np.zeros_like(kpts[..., :-1])
    vecs2 = np.zeros_like(kpts[..., :-1])
    
    for bone1, bone2 in pairs:
        if bone2 is not None:
            vecs1[:, bone1[0]] = kpts[..., bone1[1], :-1] - kpts[..., bone1[0], :-1]
            vecs2[:, bone2[0]] = kpts[..., bone2[1], :-1] - kpts[..., bone2[0], :-1]
            angles[..., bone1[0], -1] = (kpts[..., bone1[1], -1] + kpts[..., bone1[0], -1] + kpts[..., bone2[1], -1] + kpts[..., bone2[0], -1]) / 4
        else:
            vecs1[:, bone1[0]] = kpts[..., bone1[1], :-1] - kpts[..., bone1[0], :-1]
            vecs2[:, bone1[0]] = kpts[..., bone1[1], :-1] - kpts[..., bone1[0], :-1]
            angles[..., bone1[0], -1] = 0

    angles[..., 0] = calculate_angles(vecs1, vecs2)

    return angles


def joint_to_bone(kpts, kpts_type):
    bones = np.zeros_like(kpts)
    if kpts_type == "motionbert":
        # (source_kpts, target_kpts)
        pairs = [(0,0),(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),(8,11),(11,12),
                 (12,13),(8,14),(14,15),(15,16)]
    elif kpts_type == "vitpose":
        pairs = [(33,0),(0,1),(0,2),(1,3),(2,4),(33,5),(33,6),(5,7),(6,8),(7,9),(8,10),(34,11),(34,12),
                 (11,13),(12,14),(13,15),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22),
                 (9,23),(9,24),(24,25),(9,26),(26,27),(10,28),(10,29),(29,30),(10,31),(31,32),(34,33),(34,34)]
    elif kpts_type == "mediapipe":
        pairs = [(27,0),(0,1),(0,2),(1,3),(2,4),(27,5),(27,6),(5,7),(6,8),(7,9),(8,10),(9,11),(10,12),
                 (9,13),(10,14),(9,15),(10,16),(28,17),(28,18),(17,19),(18,20),(19,21),(20,22),
                 (21,23),(22,24),(21,25),(22,26),(28,27),(28,28)]        
    for k1, k2 in pairs:
        bones[..., k2, :-1] = kpts[..., k2, :-1] - kpts[..., k1, :-1]
        # score
        bones[..., k2, -1] = (kpts[..., k2, -1] + kpts[..., k1, -1])/2
    
    return bones



def combine_mutiple_data_view(list_data):
    new_list = []
    new_score = []
    
    for data in list_data:
        new_list.append(data[..., :-1])
        new_score.append(data[..., -1:])
    
    new_score = np.mean(np.array(new_score), axis=0)
    new_list.append(new_score)
    new_data = np.concatenate(tuple(new_list), axis=-1)
    return new_data


def find_negative_from_distance_matrix():
    norm2_rows =  anchors 
    norm2_columns = torch.cat((anchors, positives), dim=0)
    norm2_matrix = get_sample_to_sample_matrix_tensor(norm2_rows, norm2_columns)
    norm2_matrix = torch.linalg.norm(norm2_matrix[..., 0] - norm2_matrix[..., 1], ord=2, dim=-1)
    dist_matrix = DistModel(norm2_matrix)

