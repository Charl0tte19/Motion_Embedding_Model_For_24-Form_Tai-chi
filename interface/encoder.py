import torch
from torch import nn
from models.stgcn import ST_GCN_18
import numpy as np
from models.linear import TemporalSimpleModel

class ST_GCN_Encoder():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_type_views = ("joint","motion","bone", "motion_bone", "angle")
        #self.data_type_views = ("joint","angle")
        #self.data_type_views = ("bone",)
        #self.data_type_views = ("joint","bone", "angle")
        #self.data_type_views = ("joint","motion")
        # self.data_type_views = ("joint","motion","bone")
        self.graph_cfg = {'layout': "motionbert", 'strategy': 'spatial', 'max_hop': 1}
        self.input_channel_size = len(self.data_type_views)*3 + 1 if "angle" not in self.data_type_views else (len(self.data_type_views)-1)*3 + 2
        self.stgcn = ST_GCN_18(in_channels=self.input_channel_size, num_class=24, graph_cfg=self.graph_cfg, edge_importance_weighting=True, dropout=0.5, clip_len=10).to(self.device)
        #self.stgcn = TemporalSimpleModel(in_dim=17*self.input_channel_size, clip_len=10).to(self.device)
        self.stgcn.load_state_dict(torch.load(f'./models/best_stgcn.pth', weights_only=True)["model"])
        self.stgcn.eval()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def normalized_keypoints_numpy(self, kpts_with_score: np.ndarray, keypoint_type="vitpose", raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="first_frame"):

        if keypoint_type == "vitpose":
            valid_indices = np.array(list(range(0, 23)) + [95, 96, 99, 108, 111, 116, 117, 120, 129, 132] + [0, 0])
        elif keypoint_type == "mediapipe":
            valid_indices = np.array([0,2,5,7,8] + list(range(11, 33)) + [0, 0])
        elif keypoint_type == "motionbert":
            valid_indices = np.array(list(range(0, 17)))

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

        kpts = kpts_with_score[... ,:-1]

        offsets = None 
        scales = None 

        if keypoint_type == "vitpose":
            thorax_kpt = kpts[..., -2, :]
            hip_kpt = kpts[..., -1, :]

        elif keypoint_type == "motionbert":
            thorax_kpt = kpts[..., 8, :]
            hip_kpt = kpts[..., 0, :]

        elif keypoint_type == "mediapipe":
            thorax_kpt = kpts[..., -2, :]
            hip_kpt = kpts[..., -1, :]

        thorax_to_hip_dist = np.linalg.norm(thorax_kpt-hip_kpt, ord=2, axis=-1)
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


    def rotation_based_on_normal(self, anchor_kpts, positive_kpts, kpts_type):

        if kpts_type == "motionbert":
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

        # Azimuth is positive: clockwise rotation
        rotation_matrix = np.array([
            [np.cos(azimuth), 0, np.sin(azimuth)],
            [0, 1, 0],
            [-np.sin(azimuth), 0, np.cos(azimuth)]
        ])

        keypoints = positive_kpts[..., :-1]
        rotated_keypoints = keypoints @ rotation_matrix.T
        positive_kpts[..., :-1] = rotated_keypoints

        return positive_kpts


    def to_motion(self, data):
        motions = np.zeros_like(data)
        # motions[:-1, :, :-1] = np.diff(data[:,:,:-1], axis=0)  
        motions[1:, :, :-1] = np.diff(data[:,:,:-1], axis=0)
        # score
        motions[:-1, :, -1] = 0.5*(data[:-1, :, -1]+data[1:, :, -1])
        motions[-1, :, -1] = data[-1, :, -1]
        return motions


    def calculate_angles(self, vec1, vec2):
        dot_product = np.sum(vec1 * vec2, axis=-1)

        norm_vec1 = np.linalg.norm(vec1, axis=-1)
        norm_vec2 = np.linalg.norm(vec2, axis=-1)

        norm_vec1[norm_vec1 == 0] = 100
        norm_vec2[norm_vec2 == 0] = 100

        cos_theta = np.where(np.logical_or(norm_vec1 == 100, norm_vec2 == 100), 1, dot_product / (norm_vec1 * norm_vec2))

        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angles = np.arccos(cos_theta)

        return angles


    def get_angle_from_bone(self, kpts, kpts_type):

        # kpts shape: (frames_size, kpts_size, channels_size)
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

        angles[..., 0] = self.calculate_angles(vecs1, vecs2)

        return angles


    def joint_to_bone(self, kpts, kpts_type):
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

    def combine_mutiple_data_view(self, list_data):
        new_list = []
        new_score = []

        for data in list_data:
            new_list.append(data[..., :-1])
            new_score.append(data[..., -1:])

        new_score = np.mean(np.array(new_score), axis=0)
        new_list.append(new_score)
        new_data = np.concatenate(tuple(new_list), axis=-1)
        return new_data

    def combine_mediapipe_pose_and_world_numpy(self, kpts_pose, kpts_world, replace=False):
        if replace:
            kpts_world[..., 0:-2] = kpts_pose[..., 0:-2]
        else:
            hip_kpt_pose = kpts_pose[:, 0, :-2]
            hip_kpt_world = kpts_world[:, 0, :-2]
            displacement = (hip_kpt_pose-hip_kpt_world)[..., np.newaxis, :]
            kpts_world[..., :-2] += displacement
        return kpts_world


    def cosine_distance(self, p, sp):
        return (1.0 - self.cosine_similarity(p, sp)) / 2.0


    def compute_sample_to_sample_mpjpe_sequence_numpy(self, anchor_kpts_with_score: np.ndarray, matched_kpts_with_score: np.ndarray, dist_method="l2_dist", cfg=None, temporal_reduction=None, threshold=0.3):
        dist_sequence = np.stack((anchor_kpts_with_score, matched_kpts_with_score), axis=-1)
        # only consider visible keypoints
        mask = np.logical_and(dist_sequence[..., -1, 0] > threshold, dist_sequence[..., -1, 1] > threshold)   
        dist_sequence[~mask] = 0.0
        # binarize (matrix contains only 0 and 1)
        dist_sequence[..., -1, :] = (dist_sequence[..., -1, :] > threshold).astype(np.float32)

        # score_matrix shape: (anchor_frame_size, matched_frame_size, keypoint_size,)
        score_matrix = dist_sequence[..., -1, 0]*dist_sequence[..., -1, 1]
        num_visible_kpts = np.sum(score_matrix, axis=-1)

        if dist_method == "cosine":
            tmp_cos = np.sum(dist_sequence[..., :-1, 0] * dist_sequence[..., :-1, 1], axis=-1) / (np.linalg.norm(dist_sequence[..., :-1, 0], axis=-1) * np.linalg.norm(dist_sequence[..., :-1, 1], axis=-1) + 1e-6)
            similarity = np.mean(tmp_cos, axis=-1)
            similarity[np.isnan(similarity)] = 0
            dist_sequence = np.clip((similarity+1.0)/2, a_min=0, a_max=1)
        
        elif dist_method == "rbf":
                gamma = 10.0
                tmp_rbf = np.exp(-gamma * np.sum(np.square(dist_sequence[..., :-1, 0] - dist_sequence[..., :-1, 1]), axis=-1))
                similarity = np.mean(tmp_rbf, axis=-1)

                similarity[np.isnan(similarity)] = 0
                dist_sequence = np.clip(similarity, a_min=0, a_max=1)

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



    def start_encoding(self, webcam_kpts, webcam_kpts_2d, video_kpts):

        normalized_webcam_kpts = self.normalized_keypoints_numpy(webcam_kpts, "motionbert", raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
        normalized_webcam_kpts_2d = self.normalized_keypoints_numpy(webcam_kpts_2d, "motionbert", raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
        normalized_webcam_kpts = self.combine_mediapipe_pose_and_world_numpy(kpts_pose=normalized_webcam_kpts_2d, kpts_world=normalized_webcam_kpts)
        normalized_video_kpts = self.normalized_keypoints_numpy(video_kpts, "motionbert", raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
        
        # Do not perform rotation
        rotated_webcam_kpts = normalized_webcam_kpts
        # rotated_webcam_kpts = self.procrustes_align_keypoints_numpy(kpts_with_score=normalized_webcam_kpts, target_kpts_with_score=normalized_video_kpts, apply_type="first_frame")
        # rotated_webcam_kpts = self.rotation_based_on_normal(anchor_kpts=np.copy(normalized_video_kpts), positive_kpts=np.copy(normalized_webcam_kpts), kpts_type="motionbert")
        # return self.compute_sample_to_sample_mpjpe_sequence_numpy(rotated_webcam_kpts, normalized_video_kpts, dist_method="rbf", temporal_reduction="mean")
        
        webcam_data_views_tuple = []
        video_data_views_tuple = []
        if "joint" in self.data_type_views:
            webcam_data_views_tuple.append(rotated_webcam_kpts)
            video_data_views_tuple.append(normalized_video_kpts)

        if "angle" in self.data_type_views:
            webcam_data_views_tuple.append(self.get_angle_from_bone(rotated_webcam_kpts, "motionbert"))
            video_data_views_tuple.append(self.get_angle_from_bone(normalized_video_kpts, "motionbert"))

        if "motion" in self.data_type_views:
            webcam_data_views_tuple.append(self.to_motion(rotated_webcam_kpts))
            video_data_views_tuple.append(self.to_motion(normalized_video_kpts))

        if "bone" in self.data_type_views:
            tmp_webcam_bone = self.joint_to_bone(rotated_webcam_kpts, "motionbert")
            tmp_video_bone = self.joint_to_bone(normalized_video_kpts, "motionbert")

            webcam_data_views_tuple.append(tmp_webcam_bone)
            video_data_views_tuple.append(tmp_video_bone)   

        if "bone" in self.data_type_views and "motion_bone" in self.data_type_views:
            webcam_data_views_tuple.append(self.to_motion(tmp_webcam_bone))
            video_data_views_tuple.append(self.to_motion(tmp_video_bone)) 
        elif "motion_bone" in self.data_type_views:
            tmp_webcam_bone = self.joint_to_bone(rotated_webcam_kpts, "motionbert")
            tmp_video_bone = self.joint_to_bone(normalized_video_kpts, "motionbert")
            webcam_data_views_tuple.append(self.to_motion(tmp_webcam_bone))
            video_data_views_tuple.append(self.to_motion(tmp_video_bone)) 

        webcam_data_views = self.combine_mutiple_data_view(webcam_data_views_tuple)
        video_data_views = self.combine_mutiple_data_view(video_data_views_tuple) 

        # stgcn version
        batch_kpts = np.stack((webcam_data_views, video_data_views), axis=0)
        batch_kpts = np.expand_dims(batch_kpts, axis=0)
        batch_kpts = torch.FloatTensor(batch_kpts).to(self.device)
        output_embs = self.stgcn(batch_kpts)
        output_embs = torch.squeeze(output_embs, dim=(-2,-1))
        webcam_embs, video_embs = output_embs[:,0], output_embs[:,1]

        # linear version
        # batch_kpts = np.stack((webcam_data_views, video_data_views), axis=0)
        # batch_kpts = torch.FloatTensor(batch_kpts).to(self.device)
        # output_embs = self.stgcn(batch_kpts)
        # webcam_embs, video_embs = output_embs[0], output_embs[1]

        similarity = (self.cosine_similarity(webcam_embs, video_embs)+1)/2

        # from data.rendering import render_keypoints_3d_for_sample_demo
        # render_keypoints_3d_for_sample_demo("origin_video", np.array(rotated_webcam_kpts), np.array(normalized_video_kpts), np.array(normalized_webcam_kpts), fps_in=10)

        return similarity.item()











