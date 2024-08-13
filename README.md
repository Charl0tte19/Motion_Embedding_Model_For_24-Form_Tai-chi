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

Download the VitPose models from [Huggingface](https://huggingface.co/JunkyByte/easy_ViTPose) provided by easy_ViTPose.

