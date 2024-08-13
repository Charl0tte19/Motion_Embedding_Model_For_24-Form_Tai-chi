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


Note that `models/`, `easy_ViTPose/`, `motionbert/` and `forms_keypoints/` need to be copied multiple times.

```
project_root/
│
├── taichi_code/
│   ├── datasets/
│   │   └── Taichi_Clip/
│   │       └── forms_keypoints/
│   │           ├── 00/
│   │           │   ├── f00_v00_h00_00000_fps10_3d.npz
│   │           │   └── ... (something...)
│   │           │   
│   │           └── ... (something...)
│   │
│   ├── embs_model/
│   │   ├── encoder/
│   │   │   ├── configs/
│   │   │   │   ├── config.py
│   │   │   │   └── test_config.py
│   │   │   │   
│   │   │   ├── data/
│   │   │   │   ├── data_setup.py
│   │   │   │   ├── data_setup_for_test.py
│   │   │   │   ├── keypoints_dataset.py
│   │   │   │   ├── rendering.py
│   │   │   │   ├── rendering_mediapipe.py
│   │   │   │   └── utils.py
│   │   │   │
│   │   │   ├── losses/
│   │   │   │   └── triplet_loss.py
│   │   │   │    
│   │   │   ├── easy_ViTPose/
│   │   │   ├── models/
│   │   │   │   ├── mediapipe/
│   │   │   │   │   ├── pose_landmarker_heavy.task
│   │   │   │   │   └── ... (something...)
│   │   │   │   │
│   │   │   │   ├── motionbert/
│   │   │   │   │   └── FT_MB_lite_MB_ft_h36m_global_lite
│   │   │   │   │       └── best_epoch.bin
│   │   │   │   │
│   │   │   │   ├── vitpose/
│   │   │   │   │   ├── vitpose-b-wholebody.pth
│   │   │   │   │   └── ... (something...)
│   │   │   │   │
│   │   │   │   ├── yolov8s.pt
│   │   │   │   ├── gconv.py
│   │   │   │   ├── graph.py
│   │   │   │   ├── linear.py
│   │   │   │   └── stgcn.py
│   │   │   │
│   │   │   ├── motionbert/
│   │   │   ├── engine.py
│   │   │   ├── engine_for_test.py
│   │   │   ├── flops_test.py
│   │   │   ├── test.py
│   │   │   └── train.py
│   │   │
│   │   ├── logs/
│   │   ├── logs_for_test/
│   │   └── results/
│   │ 
│   ├── fps10/
│   │   ├── Taichi_Clip/
│   │   │   ├── 00/
│   │   │   │   ├── 00_00_00_0000.mp4
│   │   │   │   └── ... (something...)
│   │   │   │   
│   │   │   └── ... (something...)
│   │   │
│   │   └── Taichi_Clip2/
│   │       ├── 24/
│   │       │   ├── 24_00_40_1597.mp4
│   │       │   └── ... (something...)
│   │       │   
│   │       └── ... (something...)
│   │
│   └── pose_est_model/
│       ├── easy_ViTPose/
│       ├── models/
│       │   ├── mediapipe/
│       │   │   ├── pose_landmarker_heavy.task
│       │   │   └── ... (something...)
│       │   │
│       │   ├── motionbert/
│       │   │   └── FT_MB_lite_MB_ft_h36m_global_lite
│       │   │       └── best_epoch.bin
│       │   │
│       │   ├── vitpose/
│       │   │   ├── vitpose-b-wholebody.pth
│       │   │   └── ... (something...)
│       │   │
│       │   └── yolov8s.pt
│       │
│       ├── motionbert/
│       ├── examples/
│       ├── imgs/
│       ├── median_filter_inference_generate_form_keypoints_for_all_video.py
│       ├── median_filter_mediapipe_to_motionbert_generate_form_keypoints_for_all_video.py
│       ├── oneeuro_filter_mediapipe_to_motionbert_generate_form_keypoints_for_all_video.py
│       └── video_to_imgs_using_opencv.py
│
└── interface/
    ├── Taichi_Clip/
    │   ├── forms_keypoints/
    │   │   ├── dataset_embs/
    │   │   │   ├── 0_0.npy
    │   │   │   └── ... (something...)
    │   │   │
    │   │   ├── 00/
    │   │   │   ├── f00_v00_h00_00000_fps10_3d.npz
    │   │   │   └── ... (something...)
    │   │   │
    │   │   └── ... (something...)
    │   │
    │   └── teacher_keypoints/
    │       ├── teacher_embs/
    │       │   ├── 0_0.npy
    │       │   └── ... (something...)
    │       │
    │       ├── 00/
    │       │   └── f00_v00_h07_00012_fps10_3d.npz
    │       │
    │       └── ... (something...)
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
    ├── clip_motion_embs/
    ├── clip_video/
    │   ├── bad/
    │   ├── good/
    │   └── ok/
    │
    ├── teacher_data/
    │   ├── 07_00_16_0527.mp4
    │   ├── f07_v00_h16_00649_fps10_2d.npz
    │   ├── f07_v00_h16_00649_fps10_3d.npz
    │   └── t_pose.npz
    │
    ├── compare_with_dataset_embs.py
    ├── each_form_embedding.py
    ├── encoder.py
    ├── frame_to_skeleton.py
    ├── renderer.py
    └── webcam.py
```


