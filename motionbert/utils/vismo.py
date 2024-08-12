import numpy as np
import os
import cv2
import copy
from tqdm import tqdm
from motionbert.utils.tools import ensure_dir
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

def render_and_save(motion_input, save_path, keep_imgs=False, fps=25, color="#F96706#FB8D43#FDB381", with_conf=False, draw_face=False):
    ensure_dir(os.path.dirname(save_path))
    motion = copy.deepcopy(motion_input)
    if motion.shape[-1]==3:
        motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 

    motion_world = pixel2world_vis_motion(motion, dim=3)
    motion2video_3d(motion_world, save_path=save_path, keep_imgs=keep_imgs, fps=fps)

def render_for_webcam(motion_input, plt, ax):
    motion = copy.deepcopy(motion_input)
    if motion.shape[-1]==3:
        motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 
        
    motion_world = pixel2world_vis_motion(motion, dim=3)
    return motion2video_3d_webcam(motion_world, plt, ax)

def render_for_two_threading(motion_input, plt, ax):
    motion = copy.deepcopy(motion_input)
    if motion.shape[-1]==3:
        motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 

    motion_world = pixel2world_vis_motion(motion, dim=3)
    return motion2video_3d_for_two_threading(motion_world, plt, ax)


def my_render_fn(motion_input, fig, ax):
    motion = copy.deepcopy(motion_input)
    if motion.shape[-1]==3:
        motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 

    motion_world = pixel2world_vis_motion(motion, dim=3)
    return my_motion_to_video3d_fn(motion_world, fig, ax)

def pixel2world_vis_motion(motion, dim=2, is_tensor=False):
    N = motion.shape[-1]
    if dim==2:
        offset = np.ones([2,N]).astype(np.float32)
    else:
        offset = np.ones([3,N]).astype(np.float32)
        offset[2,:] = 0
    if is_tensor:
        offset = torch.tensor(offset)
    return (motion + offset) * 512 / 2

def get_img_from_fig(fig):
    fig.canvas.draw()
    # draw must be called at least once before this function will work and to update the renderer for any subsequent changes to the Figure.
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    return img_bgr


def motion2video_3d(motion, save_path, fps=25, keep_imgs = False):
    vlen = motion.shape[-1]
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    figsize = (8, 8)
    dpi = 100  
    fig = plt.figure(0, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    width_in_pixels = figsize[0] * dpi
    height_in_pixels = figsize[1] * dpi
    videowriter = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, (width_in_pixels,height_in_pixels))  # type: ignore

    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        ax.cla()
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=12., azim=80)
        #plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
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
        videowriter.write(frame_vis)
    plt.close()
    videowriter.release()


def motion2video_3d_webcam(motion,plt,ax):
    vlen = motion.shape[-1]
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    # width_in_pixels = figsize[0] * dpi
    # height_in_pixels = figsize[1] * dpi
    j3d = motion[:,:,0]
    ax.cla()
    ax.set_xlim(-512, 0)
    ax.set_ylim(-256, 256)
    ax.set_zlim(-512, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
 

    #plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    for i in range(len(joint_pairs)):
        limb = joint_pairs[i]
        xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
        if joint_pairs[i] in joint_pairs_left:
            ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        elif joint_pairs[i] in joint_pairs_right:
            ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        else:
            ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization

    frame_vis = get_img_from_fig(plt.gcf())
    
    plt.imshow(frame_vis)
    plt.pause(.001)
    return frame_vis
    
def motion2video_3d_for_two_threading(motion,plt,ax):
    vlen = motion.shape[-1]
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    frames = []

    for f in range(vlen):
        j3d = motion[:,:,f]
        ax.cla()
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # ax.zaxis.set_ticklabels([])      

        #plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            else:
                ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization

        frame_vis = get_img_from_fig(plt.gcf())
        frames.append(frame_vis)
        # plt.imshow(frame_vis)
        # plt.pause(.001)
    return frames


def my_motion_to_video3d_fn(motion,fig,ax):
    vlen = motion.shape[-1]

    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    
    # color_mid = "#00457E"
    color_mid = "#23577E"
    #color_left = "#02315E"
    color_left = "#FF8A00"
    #color_right = "#2F70AF"
    color_right = "#00D9E7"

    frames = []

    for f in range(vlen):
        j3d = motion[:,:,f]
        ax.cla()
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # ax.zaxis.set_ticklabels([])      

        #plt.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
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
        frames.append(frame_vis)
        # plt.imshow(frame_vis)
        # plt.pause(.001)
    return frames