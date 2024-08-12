import os

class Training_Config:
    # Use class variables

    # For demonstration purposes only, to show how to select a, p, n
    only_for_sampling_demo = False
    # num_workers = os.cpu_count()
    num_workers = int(os.cpu_count()/2)
    # num_workers = 0
    batch_size = 1
    clip_len = 10
    num_epochs = 1
    learning_rate = 1e-2
    kpts_type_3d = "motionbert"
    kpts_type_2d = "vitpose"
    kpts_type_for_train = "motionbert"
    
    # ("joint","motion","bone","motion_bone", "angle")
    data_type_views = ("joint","motion","bone","motion_bone", "angle")

    # Method for DTW distance
    dtw_dist_method = "cosine"  # rbf  
    similarity_dist_method = "cosine" # rbf
    # Whether to reduce the contribution of keypoints due to the estimation cofidence
    use_score_weight = False
    # Whether to reduce similarity due to fewer visible points
    use_visibility_weight = False
    rbf_gamma = 10 # 5, 10, 50

    # FPS of the videos
    fps = 10
    # stride of sliding window 
    sliding_window_sample_step = clip_len
    # Method for selecting negative samples for anchor and positive
    negative_sample_method = "continuous_with_random_start" # random or continuous or continuous_with_random_start

    use_scheduler = False


    # For loss
    triplet_margin = 0.2 # [0.2, 0.5, 1.0, None]
    triplet_method = "cosine"  # cosine, rbf
    # avg loss or sum loss
    reduction = "sum"


