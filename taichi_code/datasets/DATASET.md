## Dataset Description

#### File Naming Convention

Files are named according to the following convention:

**`{form_id}_{view_id}_{person_id}_{file_id}_{fps}_2d.npz`** for [COCO-WholeBody
](https://github.com/jin-s13/COCO-WholeBody/tree/master) 2D keypoints and **`{form_id}_{view_id}_{person_id}_{file_id}_{fps}_3d.npz`** for [MotionBERT](https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/hm36.py#L32) 3D keypoints.

- **`{form_id}`**: Identifier for the Tai-Chi form (e.g., `f01` for Tai-Chi form 01)
- **`{view_id}`**: Identifier for the view angle (e.g., `v02` for view angle 02)
- **`{person_id}`**: Identifier for the person in the video (e.g., `h03` for person 03)
- **`{file_id}`**: Unique file ID for distinguishing different files
- **`{fps}`**: Frames per second of the original video (e.g., `fps10` for videos recorded at 10 fps)

#### File Content

Each `.npz` file contains:

- **`frame_ith`**: A ndarray of frame indices. Shape: `(number of frames,)`
- **`keypoints`**: A ndarray with keypoint data. Shape: `(number of frames, number of keypoints, number of channels)`

This structure captures the skeleton data and their corresponding video frames.

