import torch
from torch import nn
from models.stgcn import ST_GCN_18
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import os

class ST_GCN_Encoder():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_type_views = ("joint","motion","bone", "motion_bone", "angle")
        self.graph_cfg = {'layout': "motionbert", 'strategy': 'spatial', 'max_hop': 1}
        self.input_channel_size = len(self.data_type_views)*3 + 1 if "angle" not in self.data_type_views else (len(self.data_type_views)-1)*3 + 2
        self.stgcn = ST_GCN_18(in_channels=self.input_channel_size, num_class=24, graph_cfg=self.graph_cfg, edge_importance_weighting=True, dropout=0.5, clip_len=10).to(self.device)
        self.stgcn.load_state_dict(torch.load(f'./models/best_epoch.pth')["model"])
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

    def cosine_distance(self, p, sp):
        return (1.0 - self.cosine_similarity(p, sp)) / 2.0

    def get_img_from_fig(self, fig):
        fig.canvas.draw()
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        return img_bgr

    def render_keypoints_3d_for_sample_demo(self, video_name, anchor, positive, similarity=None, fps_in=10):

        batch_size = 1
        clip_len = anchor.shape[0]
        columns = 2

        video_name = f"./{video_name}_{time.time()}_3d.mp4"
        out_writer_3d = cv2.VideoWriter(video_name,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps_in, (300*columns, 300*batch_size))  # type: ignore
        assert out_writer_3d.isOpened()

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

        anchor = np.transpose(anchor[...,:-1], (1,2,0))
        positive = np.transpose(positive[...,:-1], (1,2,0))

        joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
        joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

        color_mid = "#23577E"
        color_left = "#FF8A00"
        color_right = "#00D9E7"

        for f in range(clip_len):
            for row_idx in range(len(axes)):
                if batch_size == 1:
                    ax = axes[row_idx]
                    ax.view_init(elev=12., azim=80)
                    if row_idx == 0:
                        motion_world = anchor
                    elif row_idx == 1: 
                        motion_world = positive

                    j3d = motion_world[:,:,f]
                    ax.cla()
                    ax.set_xlim(-256, 256)
                    ax.set_ylim(-256, 256)
                    ax.set_zlim(-256, 256)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')   
                    for i in range(len(joint_pairs)):
                        limb = joint_pairs[i]
                        xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
                        if joint_pairs[i] in joint_pairs_left:
                            ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                        elif joint_pairs[i] in joint_pairs_right:
                            ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                        else:
                            ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization

            frame_vis = self.get_img_from_fig(fig)

            if similarity is not None:
                score = str(round(similarity,2))
                frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(225,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

            out_writer_3d.write(frame_vis)
        plt.close()
        out_writer_3d.release()

    def start_encoding(self, anchor_kpts):

        normalized_anchor_kpts = self.normalized_keypoints_numpy(anchor_kpts, "motionbert", raw=True, only_raw=False, only_offset=False, only_scale=False, offset_type="first_frame", scale_type="all_frames")
        anchor_data_views_tuple = []
        if "joint" in self.data_type_views:
            anchor_data_views_tuple.append(normalized_anchor_kpts)

        if "angle" in self.data_type_views:
            anchor_data_views_tuple.append(self.get_angle_from_bone(normalized_anchor_kpts, "motionbert"))

        if "motion" in self.data_type_views:
            anchor_data_views_tuple.append(self.to_motion(normalized_anchor_kpts))

        if "bone" in self.data_type_views:
            tmp_anchor_bone = self.joint_to_bone(normalized_anchor_kpts, "motionbert")
            anchor_data_views_tuple.append(tmp_anchor_bone)

        if "bone" in self.data_type_views and "motion_bone" in self.data_type_views:
            anchor_data_views_tuple.append(self.to_motion(tmp_anchor_bone))
            
        elif "motion_bone" in self.data_type_views:
            tmp_anchor_bone = self.joint_to_bone(normalized_anchor_kpts, "motionbert")
            anchor_data_views_tuple.append(self.to_motion(tmp_anchor_bone))

        anchor_data_views = self.combine_mutiple_data_view(anchor_data_views_tuple)

        batch_kpts = np.stack((anchor_data_views, ), axis=0)
        batch_kpts = np.expand_dims(batch_kpts, axis=0)
        batch_kpts = torch.FloatTensor(batch_kpts).to(self.device)

        with torch.no_grad():
            output_embs = self.stgcn(batch_kpts)
            output_embs = torch.squeeze(output_embs, dim=(-2,-1))
            anchor_embs = output_embs[0,0]

        return anchor_embs.detach().cpu().numpy()


stgcn = ST_GCN_Encoder()
root = os.path.abspath(__file__).split("interface")[0]
saved_folder_name = "Taichi_Clip"
forms_kpts_folder = os.path.join(root,"Taichi_Clip","forms_keypoints")
form_ids = [f"{i:02}" for i in range(24)]
file_count = 0

duration = [11, 24, 7, 26, 6, 29, 24, 26, 12, 21, 8, 7, 10, 7, 10, 14, 14, 16, 7, 6, 14, 8, 9, 9]
clip_motion_embs = {form_id: {clip_id: [] for clip_id in range(duration[form_id])} for form_id in range(24)}

for form_id in form_ids:
    for form_root, form_dirs, form_files in sorted(os.walk(os.path.join(forms_kpts_folder, form_id))):
        for form_file in sorted(form_files):
            if form_file.endswith("3d.npz"):
                _, camera_view, human_id, file_name, _, _ = form_file.split("_")   
                if camera_view not in ["v00","v01","v07"]:
                    continue

                anchor_kpts = np.load(os.path.join(form_root, form_file))["keypoints"]
                
                if(len(anchor_kpts) != duration[int(form_id)]*10):
                                new_indices = np.linspace(0, len(anchor_kpts) - 1, duration[int(form_id)]*10)
                                new_indices = np.round(new_indices).astype('int32')
                                anchor_kpts = anchor_kpts[new_indices]
            
                clip_id = 0
                for idx in range(0, len(anchor_kpts), 10):
                    anchor_embs = stgcn.start_encoding(anchor_kpts[idx:idx+10])
                    clip_motion_embs[int(form_id)][clip_id].append(anchor_embs)
                    clip_id += 1
    

for form_id in clip_motion_embs.keys():
    for clip_id in clip_motion_embs[int(form_id)].keys():
        print(form_id, clip_id, len(np.array(clip_motion_embs[int(form_id)][clip_id])))
        dataset_embs = np.mean(np.array(clip_motion_embs[int(form_id)][clip_id]), axis=0)
        np.save(f'./clip_motion_embs/{form_id}_{clip_id}.npy', dataset_embs)
