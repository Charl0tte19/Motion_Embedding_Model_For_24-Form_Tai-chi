# Motion_Embedding_Model_For_24-Form_Tai-chi

Implementation of "Real-Time Feedback via Pose Estimation and Representation Learning for a Tai-Chi Chuan Assisted Learning System"

## Prerequisites

Our experiments were run on Ubuntu server with GeForce RTX 4090. We use Python 3.10 and the required packages are listed as follows.

- pytorch 2.4.0+cu118
- torchvision 2.4.0+cu118 
- ultralytics 8.2.76
- python-ffmpeg 2.0.12
- mediapipe 0.10.14
- tensorboard 2.17.0
- moviepy 1.0.3
- easydict 1.13
- fvcore 0.1.5.post20221221
- pyqt5 5.15.11


Run the following commands to install the required packages:
```
conda create --name motion_embs python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install python-ffmpeg
pip install mediapipe
pip install tensorboard
pip install moviepy
pip install easydict
pip install fvcore
pip install pyqt5
```

## Download models

- Download the motion embedding model from [GoogleDrive](https://drive.google.com/file/d/1zWgl8buEYeOPvHqLf13iBU76yFEHMg9B/view?usp=sharing), which includes the necessary pre-trained pose estimation models.


Alternatively, download the pretrained pose estimation models you want to use from the original sources.
- Download the Mediapipe models from [this page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- Download the VitPose models from [Huggingface](https://huggingface.co/JunkyByte/easy_ViTPose) provided by [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose/tree/main).
- Download the Motionbert model from [Onedrive](https://onedrive.live.com/?authkey=%21ALth1xunGWSXSeA&id=A5438CD242871DF0%21173&cid=A5438CD242871DF0) provided by [MotionBERT](https://github.com/Walter0807/MotionBERT/tree/main).

> [!NOTE]  
> Please refer to the [Directory Structure](https://github.com/Charl0tte19/Motion_Embedding_Model_For_24-Form_Tai-chi/blob/main/README.md#directory-structure) for the locations of these models.

## Download 24-form Tai-Chi dataset
Download the skeleton motion dataset from [GoogleDrive](https://drive.google.com/file/d/1dSQ3Y4Fn2sYuSmV2B0HQLc0n1CaHB3CF/view?usp=sharing) and place the **form_keypoints folder** under [taichi_code/datasets/Taichi_Clip](https://github.com/Charl0tte19/Motion_Embedding_Model_For_24-Form_Tai-chi/tree/main/taichi_code/datasets/Taichi_Clip).


For details about the dataset, please refer to [DATASET.md](https://github.com/Charl0tte19/Motion_Embedding_Model_For_24-Form_Tai-chi/blob/main/taichi_code/datasets/DATASET.md).

## Directory Structure

Please organize the downloaded `models` and `dataset` according to the structure below. 

> [!IMPORTANT]  
> Note that `models/`, `easy_ViTPose/`, `motionbert/` and `forms_keypoints/` need to be copied multiple times.

```
project_root/
│
├── taichi_code/
│   ├── datasets/                                                                               # Folder for skeleton motion data
│   │   ├── Taichi_Clip/                
│   │   │   └── forms_keypoints/                
│   │   │       ├── 00/                                                                         # Folder for form 00 (i.e., Commencing Form)
│   │   │       │   ├── f00_v00_h00_00000_fps10_3d.npz                                          # Example skeleton motion data
│   │   │       │   └── ...Other Commencing Form data              
│   │   │       │                   
│   │   │       └── ...Other forms' folders               
│   │   │                   
│   │   ├── mediapipe_generate_files_list_json.py                                               # Generate a JSON file listing mediapipe data files for batch creation (not used)
│   │   └── motionbert_generate_files_list_json.py                                              # Generate a JSON file listing motionbert data files for batch creation
│   │               
│   ├── embs_model/                                                                             # Code related to motion embeddings model
│   │   ├── encoder/                                                                            # ST-GCN encoder code
│   │   │   ├── configs/                
│   │   │   │   ├── config.py                                                                   # Config file for training model
│   │   │   │   └── test_config.py                                                              # Config file for testing model
│   │   │   │                   
│   │   │   ├── data/                                                                           # Code related to processing skeleton motion data
│   │   │   │   ├── data_setup.py                                                               # Preprocessing code for training batch
│   │   │   │   ├── data_setup_for_test.py                                                      # Preprocessing code for testing batch
│   │   │   │   ├── keypoints_dataset.py                                                        # Code for reading keypoints dataset
│   │   │   │   ├── rendering.py                                                                # Code for rendering motionbert keypoints data
│   │   │   │   ├── rendering_mediapipe.py                                                      # Code for rendering mediapipe keypoints data
│   │   │   │   └── utils.py                                                                    # Code for keypoints data preprocessing (e.g., normalization)
│   │   │   │               
│   │   │   ├── losses/             
│   │   │   │   └── triplet_loss.py                                                             # Code for triplet loss function
│   │   │   │                   
│   │   │   ├── easy_ViTPose/                                                                   # vitpose code (details refer to the original easy_ViTPose repo)
│   │   │   ├── models/                                                                         # Code related to model architecture and pretrained models
│   │   │   │   ├── mediapipe/                                                                  # Pretrained mediapipe models
│   │   │   │   │   ├── pose_landmarker_heavy.task              
│   │   │   │   │   └── ...Other mediapipe models                
│   │   │   │   │               
│   │   │   │   ├── motionbert/                                                                 # Pretrained motionbert models
│   │   │   │   │   └── FT_MB_lite_MB_ft_h36m_global_lite               
│   │   │   │   │       └── best_epoch.bin              
│   │   │   │   │               
│   │   │   │   ├── vitpose/                                                                    # Pretrained vitpose models
│   │   │   │   │   ├── vitpose-b-wholebody.pth             
│   │   │   │   │   └── ...Other vitpose models              
│   │   │   │   │               
│   │   │   │   ├── yolov8s.pt                                                                  # Model for detecting people, used with vitpose
│   │   │   │   ├── gconv.py                                                                    # Code for graph convolutional network
│   │   │   │   ├── graph.py                                                                    # Code for defining graph
│   │   │   │   ├── linear.py                                                                   # Code for Pr-VIPE model (not used)
│   │   │   │   └── stgcn.py                                                                    # Code for ST-GCN model
│   │   │   │               
│   │   │   ├── motionbert/                                                                     # Code for motionbert (details refer to the original motionbert repo)
│   │   │   ├── engine.py                                                                       # Code for defining epoch and iteration in training
│   │   │   ├── engine_for_test.py                                                              # Code for defining epoch and iteration in testing
│   │   │   ├── flops_test.py                                                                   # Calculates model FLOPs and parameter count
│   │   │   ├── test.py                                                                         # Code for model testing
│   │   │   └── train.py                                                                        # Code for model training
│   │   │               
│   │   ├── logs/                                                                               # Training event files (generated by TensorBoard SummaryWriter)
│   │   ├── logs_for_test/                                                                      # Testing event files (generated by TensorBoard SummaryWriter)
│   │   └── results/                                                                            # Trained ST-GCN model
│   │               
│   ├── fps10/                                                                                  # Raw video of 24-form Tai Chi at 10 FPS
│   │   ├── Taichi_Clip/                
│   │   │   ├── 00/                                                                             # Folder for form 00 (i.e., Commencing Form)
│   │   │   │   ├── 00_00_00_0000.mp4                                                           # Example raw video
│   │   │   │   └── ...Other Commencing Form videos                                                         
│   │   │   │                   
│   │   │   └── ...Other forms' folders               
│   │   │               
│   │   └── Taichi_Clip2/                                                                       # Folder for incomplete 24-form Tai Chi raw videos (extra data)
│   │       ├── 24/                                                                             
│   │       │   ├── 24_00_40_1597.mp4               
│   │       │   └── ...Other videos               
│   │       │                   
│   │       └── ...Other incomplete forms' folders               
│   │               
│   └── pose_est_model/                                                                         # Code for converting video to skeleton motion data
│       ├── easy_ViTPose/
│       ├── models/
│       │   ├── mediapipe/
│       │   │   ├── pose_landmarker_heavy.task
│       │   │   └── ...Other mediapipe models 
│       │   │
│       │   ├── motionbert/
│       │   │   └── FT_MB_lite_MB_ft_h36m_global_lite
│       │   │       └── best_epoch.bin
│       │   │
│       │   ├── vitpose/
│       │   │   ├── vitpose-b-wholebody.pth
│       │   │   └── ...Other vitpose models 
│       │   │
│       │   └── yolov8s.pt
│       │
│       ├── motionbert/
│       ├── examples/                                                                           # Example videos                                                                   
│       ├── imgs/                                                                               # frame imgs from one example video
│       ├── median_filter_inference_generate_form_keypoints_for_all_video.py                    # fps10 videos -> vitpose keypoints -> motionbert keypoints, with median filter
│       ├── median_filter_mediapipe_to_motionbert_generate_form_keypoints_for_all_video.py      # fps10 videos -> mediapipe keypoints -> motionbert keypoints, with median filter
│       ├── oneeuro_filter_mediapipe_to_motionbert_generate_form_keypoints_for_all_video.py     # fps10 videos -> mediapipe keypoints -> motionbert keypoints, with one-euro filter
│       └── video_to_imgs_using_opencv.py                                                       # Code for converting video to frame images
│
└── interface/
    ├── Taichi_Clip/
    │   ├── forms_keypoints/
    │   │   ├── dataset_embs/                                                                   # Average motion embeddings for each form clip from the entire dataset
    │   │   │   ├── 0_0.npy                                                                     # Motion embedding for clip 0 of form 00
    │   │   │   └── ...Other form clip embeddings
    │   │   │
    │   │   ├── 00/                                                                             # Skeleton motion data is also placed here for averaging motion embeddings
    │   │   │   ├── f00_v00_h00_00000_fps10_3d.npz
    │   │   │   └── ...Other Commencing Form data
    │   │   │
    │   │   └── ...Other forms' folders
    │   │
    │   └── teacher_keypoints/                                                                  # Motion embeddings for instructor's form clips
    │       ├── teacher_embs/
    │       │   ├── 0_0.npy
    │       │   └── ...Other form clip embeddings
    │       │
    │       ├── 00/                                                                             # Instructor's skeleton motion data
    │       │   └── f00_v00_h07_00012_fps10_3d.npz
    │       │
    │       └── Other forms' folders
    │
    ├── easy_ViTPose/
    ├── models/
    │   ├── mediapipe/
    │   │   ├── pose_landmarker_heavy.task
    │   │   └── ... (something...)
    │   │
    │   ├── motionbert/
    │   │   └── FT_MB_lite_MB_ft_h36m_global_lite
    │   │       └── best_epoch.bin
    │   │
    │   ├── vitpose/
    │   │   ├── vitpose-b-wholebody.pth
    │   │   └── ... (something...)
    │   │
    │   ├── yolov8s.pt
    │   ├── best_stgcn.pth
    │   ├── gconv.py
    │   ├── graph.py
    │   ├── linear.py
    │   └── stgcn.py
    │  
    ├── motionbert/
    ├── clip_motion_embs/                                                                       # Initial storage location for generated form clip motion embeddings
    ├── clip_video/                                                                             # User's Tai Chi motion clips, categorized by score rating
    │   ├── bad/
    │   ├── good/
    │   └── ok/
    │
    ├── teacher_data/                                                                           # Instructor's Tai Chi data
    │   ├── 07_00_16_0527.mp4                                                                   # Instructional video displayed on the interface
    │   ├── f07_v00_h16_00649_fps10_2d.npz                                                      # Instructor's 2d skeleton motion data
    │   ├── f07_v00_h16_00649_fps10_3d.npz                                                      # Instructor's 3d skeleton motion data
    │   └── t_pose.npz                                                                          # T-Pose for preliminary movement (not used)
    │
    ├── compare_with_dataset_embs.py                                                            # Compares dataset_embs with teacher_embs for motion similarity
    ├── each_form_embedding.py                                                                  # Generates dataset_embs or teacher_embs, saved in clip_motion_embs/
    ├── encoder.py                                                                              # Converts user's and instructor's skeleton motion data to motion embeddings
    ├── frame_to_skeleton.py                                                                    # Converts user video frames from webcam to MediaPipe skeleton data
    ├── renderer.py                                                                             # Renders skeletons on the interface
    └── webcam.py                                                                               # Interface code, written in PyQt5
```

## Reproduce the results

### - Convert 24-Form Tai Chi Videos to MotionBERT Keypoints Data (No need to execute)
```
cd ./taichi_code/pose_est_model/
python median_filter_inference_generate_form_keypoints_for_all_video.py --model ./models/vitpose/vitpose-b-wholebody.pth --model-name b --save-video --fps 10

( usage: median_filter_inference_generate_form_keypoints_for_all_video.py [--model MODEL] [--model-name {s,b,l,h}] [--save-video] [--fps FPS] )
```

### - Generate a JSON file listing MotionBERT data files for batch creation
```
cd ./taichi_code/datasets/
python motionbert_generate_files_list_json.py

( usage: motionbert_generate_files_list_json.py [--seed SEED] )
```

### - Train Motion Embedding Model
```
cd ./taichi_code/embs_model/
python train.py
```

### - Test Motion Embedding Model
```
cd ./taichi_code/embs_model/
python test.py
```

### - Start the Interface
```
cd ./interface/
python webcam.py
```

## Result
<p>
  <img src="https://raw.githubusercontent.com/Charl0tte19/Motion_Embedding_Model_For_24-Form_Tai-chi/main/figs/vis_01.gif" alt="Description" width="200" style="display: inline-block;"/>
  <img src="https://raw.githubusercontent.com/Charl0tte19/Motion_Embedding_Model_For_24-Form_Tai-chi/main/figs/vis_02.gif" alt="Description" width="200" style="display: inline-block;"/>
</p>

## Acknowledgements
This repo is based on [ST-GCN](https://github.com/yysijie/st-gcn), [MotionBERT](https://github.com/Walter0807/MotionBERT/tree/main), [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose),
[bpe](https://github.com/chico2121/bpe/tree/master), [Pr-VIPE](https://github.com/google-research/google-research/tree/master/poem/pr_vipe) and [pyskl](https://github.com/kennymckormick/pyskl). 

Thanks to the original authors for their awesome works!

