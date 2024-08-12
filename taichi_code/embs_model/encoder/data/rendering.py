import os 
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def joints_dict():
    joints = {
        'wholebody': {
            'keypoints': {
                0: 'nose',
                1: 'left_eye',
                2: 'right_eye',
                3: 'left_ear',
                4: 'right_ear',
                5: 'left_shoulder',
                6: 'right_shoulder',
                7: 'left_elbow',
                8: 'right_elbow',
                9: 'left_wrist',
                10: 'right_wrist',
                11: 'left_hip',
                12: 'right_hip',
                13: 'left_knee',
                14: 'right_knee',
                15: 'left_ankle',
                16: 'right_ankle',
                17: 'left_big_toe',
                18: 'left_small_toe',
                19: 'left_heel',
                20: 'right_big_toe',
                21: 'right_small_toe',
                22: 'right_heel',
                23: 'left_thumb4',
                24: 'left_forefinger1',
                25: 'left_forefinger4',
                26: 'left_pinky_finger1',
                27: 'left_pinky_finger4',
                28: 'right_thumb4',
                29: 'right_forefinger1',
                30: 'right_forefinger4',
                31: 'right_pinky_finger1',
                32: 'right_pinky_finger4',
                # my addition
                33: 'thorax',
                34: 'hip'
            },
            'skeleton': [
                [15, 13, "l"], [13, 11, "l"], [16, 14, "r"], [14, 12, "r"],
                [5, 7, "l"], [6, 8, "r"], [7, 9, "l"], [8, 10, "r"], [0, 1, "l"], [0, 2, "r"],
                [1, 3, "l"], [2, 4, "r"], [15, 17, "l"], [15, 18, "l"], [15, 19, "l"],
                [16, 20, "r"], [16, 21, "r"], [16, 22, "r"], [9, 23, "l"], [9, 24, "l"], [9, 26, "l"],
                [24, 25, "l"], [24, 26, "l"], [26, 27, "l"], [25, 27, "l"], [10, 28, "r"], [10, 29, "r"],
                [10, 31, "r"], [29, 31, "r"], [31, 32, "r"], [29, 30, "r"], [30, 32, "r"],
                [6, 33, 'r'], [5, 33, 'l'], [12, 34, 'r'], [11, 34, 'l'], [33, 34, 'm'], [0, 33, 'm'], 
            ]
        }
    }
    return joints


def draw_points(image, points, keypoint, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 150)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))

    for i, pt in enumerate(points[list(keypoint.keys())]):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[0]), int(pt[1])), circle_size, tuple(colors[i % len(colors)]), -1)

    return image


def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0,
                  confidence_threshold=0.5):

    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    for i, joint in enumerate(skeleton):
        pt1, pt2, part = joint
        pt1 = points[pt1]
        pt2 = points[pt2]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            if part == "r":
                image = cv2.line(
                    image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                    (231, 217, 0), 2
                )
            elif part == "l":
                image = cv2.line(
                    image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                    (0, 138, 255), 2
                )   
            else:
                image = cv2.line(
                    image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                    (126, 87, 35), 2
                )                              

    return image


def draw_points_and_skeleton(image, points, keypoint, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5):
    image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                          palette_samples=skeleton_palette_samples, person_index=person_index,
                          confidence_threshold=confidence_threshold)
    image = draw_points(image, points, keypoint, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold)
    return image

def render_keypoints_2d(batch_kpts, kpts_type="vitpose"):

    batch_size, clip_len = batch_kpts.shape[0], batch_kpts.shape[1]
     
    background_img = np.full((512, 512, 3), 255, dtype=np.uint8)
    batch_kpts[..., 0] = batch_kpts[..., 0]*128 + 256
    batch_kpts[..., 1] = batch_kpts[..., 1]*128 + 256
    imgs = []
    for idx in range(batch_size):
        
        if batch_size==1:
            imgs.append(background_img.copy())

        imgs.append(draw_points_and_skeleton(background_img.copy(), batch_kpts[idx, 0],
                                   joints_dict()["wholebody"]['keypoints'], # draw node
                                   joints_dict()["wholebody"]['skeleton'], # draw edge
                                   person_index=0,
                                   points_color_palette='gist_rainbow',
                                   skeleton_color_palette='jet',
                                   points_palette_samples=10,
                                   confidence_threshold=0.3))
    if batch_size==1:
        column_imgs =tuple(imgs[i] for i in range(2))
    else:
        column_imgs =tuple(imgs[i] for i in range(batch_size))
    full_img = np.hstack(column_imgs)
    return full_img


def render_keypoints_2d_for_sample_demo(video_name, anchor, positive, negative, similarity=None, similarity_negative=None, fps_in=10):

    batch_size = 1
    clip_len = anchor.shape[0]
    columns = 3

    out_writer_2d = cv2.VideoWriter(f"../logs/{video_name}_{time.time()}_2d.mp4",
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             # (width, height)
                             fps_in, (512*columns, 512*batch_size))  # type: ignore
    assert out_writer_2d.isOpened()

    background_img = np.full((512, 512, 3), 255, dtype=np.uint8)
    anchor[..., 0] = anchor[..., 0]*128 + 256
    anchor[..., 1] = anchor[..., 1]*128 + 256
    positive[..., 0] = positive[..., 0]*128 + 256
    positive[..., 1] = positive[..., 1]*128 + 256
    negative[..., 0] = negative[..., 0]*128 + 256
    negative[..., 1] = negative[..., 1]*128 + 256

    for f in range(clip_len):
        imgs = []
        imgs.append(draw_points_and_skeleton(background_img.copy(), anchor[f],
                                   joints_dict()["wholebody"]['keypoints'],
                                   joints_dict()["wholebody"]['skeleton'],
                                   person_index=0,
                                   points_color_palette='gist_rainbow',
                                   skeleton_color_palette='jet',
                                   points_palette_samples=10,
                                   confidence_threshold=0.0))

        imgs.append(draw_points_and_skeleton(background_img.copy(), positive[f],
                                   joints_dict()["wholebody"]['keypoints'],
                                   joints_dict()["wholebody"]['skeleton'],
                                   person_index=0,
                                   points_color_palette='gist_rainbow',
                                   skeleton_color_palette='jet',
                                   points_palette_samples=10,
                                   confidence_threshold=0.0))

        imgs.append(draw_points_and_skeleton(background_img.copy(), negative[f],
                                   joints_dict()["wholebody"]['keypoints'],
                                   joints_dict()["wholebody"]['skeleton'],
                                   person_index=0,
                                   points_color_palette='gist_rainbow',
                                   skeleton_color_palette='jet',
                                   points_palette_samples=10,
                                   confidence_threshold=0.0))

        column_imgs =tuple(imgs[i] for i in range(3))
        frame_vis = np.hstack(column_imgs)
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_RGBA2BGR)
        if similarity is not None:
            score = str(round(similarity[f],2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(450,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

        if similarity_negative is not None:
            score = str(round(similarity_negative[f],2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(950,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
        
        out_writer_2d.write(frame_vis)
    
    out_writer_2d.release()


def render_keypoints_2d_for_tensorboard(anchor, positive, negative, similarity=None, similarity_negative=None, embs_similarity=None, embs_similarity_negative=None):
    batch_size = 1
    clip_len = anchor.shape[0]
    columns = 3

    background_img = np.full((256, 256, 3), 255, dtype=np.uint8)
    anchor[..., :-1] = anchor[..., :-1]*64 + 128
    positive[..., :-1] = positive[..., :-1]*64 + 128
    negative[..., :-1] = negative[..., :-1]*64 + 128

    output_video = []                         
    for f in range(clip_len):
        imgs = []
        imgs.append(draw_points_and_skeleton(background_img.copy(), anchor[f],
                                   joints_dict()["wholebody"]['keypoints'],
                                   joints_dict()["wholebody"]['skeleton'],
                                   person_index=0,
                                   points_color_palette='gist_rainbow',
                                   skeleton_color_palette='jet',
                                   points_palette_samples=10,
                                   confidence_threshold=0.0))

        imgs.append(draw_points_and_skeleton(background_img.copy(), positive[f],
                                   joints_dict()["wholebody"]['keypoints'],
                                   joints_dict()["wholebody"]['skeleton'],
                                   person_index=0,
                                   points_color_palette='gist_rainbow',
                                   skeleton_color_palette='jet',
                                   points_palette_samples=10,
                                   confidence_threshold=0.0))

        imgs.append(draw_points_and_skeleton(background_img.copy(), negative[f],
                                   joints_dict()["wholebody"]['keypoints'],
                                   joints_dict()["wholebody"]['skeleton'],
                                   person_index=0,
                                   points_color_palette='gist_rainbow',
                                   skeleton_color_palette='jet',
                                   points_palette_samples=10,
                                   confidence_threshold=0.0))

        column_imgs =tuple(imgs[i] for i in range(3))
        frame_vis = np.hstack(column_imgs)
        #frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_RGBA2BGR)
        if similarity is not None:
            score = str(round(similarity,2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(180,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

        if similarity_negative is not None:
            score = str(round(similarity_negative,2))
            frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(440,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

        if embs_similarity is not None:
            score = str(round(embs_similarity,2))
            frame_vis = cv2.putText(frame_vis, "Embs Similarity: "+score,(180,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

        if embs_similarity_negative is not None:
            score = str(round(embs_similarity_negative,2))
            frame_vis = cv2.putText(frame_vis, "Embs Similarity: "+score,(440,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

        output_video.append(frame_vis)

    # N T C H W
    return np.expand_dims(np.transpose(np.array(output_video), (0,3,1,2)), axis=0)


def get_img_from_fig(fig):
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    return img_bgr


def render_keypoints_3d(batch_kpts, fps_in=10):
    batch_size = batch_kpts.shape[0]//3
    clip_len = batch_kpts.shape[1]
    columns = 3
    
    out_writer_3d = cv2.VideoWriter(f"../logs/{time.time()}_3d.mp4",
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             # (width, height)
                             fps_in, (300*columns, 300*batch_size))  # type: ignore
    assert out_writer_3d.isOpened()

    # width, height
    figsize = (3*columns, 3*batch_size)
    dpi = 100
    fig, axes = plt.subplots(batch_size, columns, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})
    plt.tight_layout()
    batch_kpts[..., 0] = (batch_kpts[..., 0]+0)*(512/2)
    batch_kpts[..., 1] = (batch_kpts[..., 1]+0)*(512/2)
    batch_kpts[..., 2] = (batch_kpts[..., 2]+0)*(512/2)

    batch_kpts = np.transpose(batch_kpts[...,:-1], (0,2,3,1))

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
                    motion_world = batch_kpts[0]
                elif row_idx == 1: 
                    motion_world = batch_kpts[0+batch_size]
                elif row_idx == 2:
                    motion_world = batch_kpts[0+batch_size*2]
                
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
            else:
                for column_idx in range(len(axes[0])):
                    ax = axes[row_idx][column_idx]
                    ax.view_init(elev=12., azim=80)
                    if column_idx == 0:
                        motion_world = batch_kpts[row_idx]
                    elif column_idx == 1:
                        motion_world = batch_kpts[row_idx+batch_size]
                    elif column_idx == 2:
                        motion_world = batch_kpts[row_idx+batch_size*2]
                    
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
        frame_vis = get_img_from_fig(fig)
        out_writer_3d.write(frame_vis)
    plt.close()
    out_writer_3d.release()



def render_keypoints_3d_for_sample_demo(video_name, anchor, positive, negative, similarity=None, similarity_negative=None, fps_in=10, kpts_type="3d"):

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

    if kpts_type == "3d":
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

        anchor = np.transpose(anchor[...,:-1], (1,2,0))
        positive = np.transpose(positive[...,:-1], (1,2,0))
        negative = np.transpose(negative[...,:-1], (1,2,0))

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
                    elif row_idx == 2:
                        motion_world = negative

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

            frame_vis = get_img_from_fig(fig)

            if similarity is not None:
                score = str(round(similarity[f],2))
                frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(225,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

            if similarity_negative is not None:
                score = str(round(similarity_negative[f],2))
                frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(525,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
            out_writer_3d.write(frame_vis)
    
    elif kpts_type == "2d":
        fig, axes = plt.subplots(batch_size, columns, figsize=figsize, dpi=dpi, subplot_kw={})
        plt.tight_layout()

        anchor[..., 0] = (anchor[..., 0]+0)*(512/2)
        anchor[..., 1] = (anchor[..., 1]+0)*(512/2)
        positive[..., 0] = (positive[..., 0]+0)*(512/2)
        positive[..., 1] = (positive[..., 1]+0)*(512/2)
        negative[..., 0] = (negative[..., 0]+0)*(512/2)
        negative[..., 1] = (negative[..., 1]+0)*(512/2)

        anchor = np.transpose(anchor[...,:-1], (1,2,0))
        positive = np.transpose(positive[...,:-1], (1,2,0))
        negative = np.transpose(negative[...,:-1], (1,2,0))

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
                    if row_idx == 0:
                        motion_world = anchor
                    elif row_idx == 1: 
                        motion_world = positive
                    elif row_idx == 2:
                        motion_world = negative

                    j3d = motion_world[:,:,f]
                    ax.cla()
                    ax.set_xlim(-256, 256)
                    ax.set_ylim(-256, 256)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    for i in range(len(joint_pairs)):
                        limb = joint_pairs[i]
                        xs, ys = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(2)]
                        if joint_pairs[i] in joint_pairs_left:
                            ax.plot(xs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                        elif joint_pairs[i] in joint_pairs_right:
                            ax.plot(xs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                        else:
                            ax.plot(xs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization

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



def render_keypoints_3d_for_tensorboard(anchor, positive, negative, similarity=None, similarity_negative=None, embs_similarity=None, embs_similarity_negative=None, kpts_type="3d"):

    batch_size = 1
    clip_len = anchor.shape[0]
    columns = 3
    
    # width, height
    figsize = (3*columns, 3*batch_size)
    dpi = 100

    if kpts_type == "3d":
        fig, axes = plt.subplots(batch_size, columns, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})
        plt.tight_layout()

        anchor[..., :-1] = (anchor[..., :-1]+0)*(512/2)
        positive[..., :-1] = (positive[..., :-1]+0)*(512/2)
        negative[..., :-1] = (negative[..., :-1]+0)*(512/2)

        anchor = np.transpose(anchor[...,:-1], (1,2,0))
        positive = np.transpose(positive[...,:-1], (1,2,0))
        negative = np.transpose(negative[...,:-1], (1,2,0))

        joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
        joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

        color_mid = "#23577E"
        color_left = "#FF8A00"
        color_right = "#00D9E7"

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

            frame_vis = get_img_from_fig(fig)

            if similarity is not None:
                score = str(round(similarity,2))
                frame_vis = cv2.putText(frame_vis, "Coords Similarity: "+score,(225,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

            if similarity_negative is not None:
                score = str(round(similarity_negative,2))
                frame_vis = cv2.putText(frame_vis, "Coords Similarity: "+score,(525,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

            if embs_similarity is not None:
                score = str(round(embs_similarity,2))
                frame_vis = cv2.putText(frame_vis, "Embs Similarity: "+score,(225,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

            if embs_similarity_negative is not None:
                score = str(round(embs_similarity_negative,2))
                frame_vis = cv2.putText(frame_vis, "Embs Similarity: "+score,(525,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

            frame_vis = frame_vis[..., ::-1]
            output_video.append(frame_vis)
    
    elif kpts_type == "2d":
        fig, axes = plt.subplots(batch_size, columns, figsize=figsize, dpi=dpi, subplot_kw={})
        plt.tight_layout()
    
        anchor[..., :-1] = (anchor[..., :-1]+0)*(512/2)
        positive[..., :-1] = (positive[..., :-1]+0)*(512/2)
        negative[..., :-1] = (negative[..., :-1]+0)*(512/2)
    
        anchor = np.transpose(anchor[...,:-1], (1,2,0))
        positive = np.transpose(positive[...,:-1], (1,2,0))
        negative = np.transpose(negative[...,:-1], (1,2,0))
    
        joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
        joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
        color_mid = "#23577E"
        color_left = "#FF8A00"
        color_right = "#00D9E7"
    
        output_video = []
        for f in range(clip_len):
            for row_idx in range(len(axes)):
                if batch_size == 1:
                    ax = axes[row_idx]
                    if row_idx == 0:
                        motion_world = anchor
                    elif row_idx == 1: 
                        motion_world = positive
                    elif row_idx == 2:
                        motion_world = negative
                    
                    j3d = motion_world[:,:,f]
                    ax.cla()
                    ax.set_xlim(-256, 256)
                    ax.set_ylim(-256, 256)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    for i in range(len(joint_pairs)):
                        limb = joint_pairs[i]
                        xs, ys = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(2)]
                        if joint_pairs[i] in joint_pairs_left:
                            ax.plot(xs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                        elif joint_pairs[i] in joint_pairs_right:
                            ax.plot(xs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                        else:
                            ax.plot(xs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                
            frame_vis = get_img_from_fig(fig)
            
            if similarity is not None:
                score = str(round(similarity,2))
                frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(225,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    
            if similarity_negative is not None:
                score = str(round(similarity_negative,2))
                frame_vis = cv2.putText(frame_vis, "Similarity: "+score,(525,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    
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
