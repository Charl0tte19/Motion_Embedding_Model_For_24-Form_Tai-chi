import os
import json
import argparse
import random
import math
import re 

def main(args):
    random.seed(args.seed)
    root = os.path.abspath(__file__).split("datasets")[0]
    test_file_paths = []
    train_file_paths = []
    test_form_file_paths_dict = {f"{i:02}": [] for i in range(31)}
    train_form_file_paths_dict = {f"{i:02}": [] for i in range(31)}

    # Taichi
    print("Taichi dataset")
    forms_keypoint_folder = os.path.join(root, "datasets", "Taichi_Clip", "forms_keypoints_mediapipe")
    form_ids = [f"{i:02}" for i in range(24)]
    form_files_dict = {f"{i:02}": {} for i in range(24)}

    for form_id in form_ids:
        view_count_dict = {}
        for form_root, form_dirs, form_files in sorted(os.walk(os.path.join(forms_keypoint_folder, form_id))):
            for form_file in sorted(form_files):
                if form_file.endswith(f"{args.keypoint_type}.npz"):
                    view_id = form_file.split("_")[1]
                    # only front-view recordings
                    if view_id not in ["v00", "v01", "v07"]:
                        continue

                    if view_id not in form_files_dict[form_id]:
                        form_files_dict[form_id][view_id] = [os.path.join(form_root,form_file)]
                    else:
                        form_files_dict[form_id][view_id].append(os.path.join(form_root,form_file))

                if args.remove_video and form_file.endswith(".mp4"):
                    os.remove(os.path.join(form_root,form_file))


    for form_id in form_ids:
        for view_id in form_files_dict[form_id].keys():
            tmp = random.sample(form_files_dict[form_id][view_id], k=math.ceil(len(form_files_dict[form_id][view_id]) / 10.0))
            test_file_paths.extend(tmp)
            test_form_file_paths_dict[form_id].extend(tmp)

            tmp = [item for item in form_files_dict[form_id][view_id] if item not in tmp]
            train_file_paths.extend(tmp)
            train_form_file_paths_dict[form_id].extend(tmp)



    form_ids = [f"{i:02}" for i in range(24,31)]
    form_files_dict = {f"{i:02}": {} for i in range(24,31)}

    for form_id in form_ids:
        view_count_dict = {}
        for form_root, form_dirs, form_files in sorted(os.walk(os.path.join(forms_keypoint_folder, form_id))):
            for form_file in sorted(form_files):
                if form_file.endswith(f"{args.keypoint_type}.npz"):
                    view_id = form_file.split("_")[1]
                    # only front-view recordings
                    if view_id not in ["v00", "v01", "v07"]:
                        continue

                    if view_id not in form_files_dict[form_id]:
                        form_files_dict[form_id][view_id] = [os.path.join(form_root,form_file)]
                    else:
                        form_files_dict[form_id][view_id].append(os.path.join(form_root,form_file))

                if args.remove_video and form_file.endswith(".mp4"):
                    os.remove(os.path.join(form_root,form_file))


    for form_id in form_ids:
        for view_id in form_files_dict[form_id].keys():
            train_file_paths.extend(form_files_dict[form_id][view_id])
            train_form_file_paths_dict[form_id].extend(form_files_dict[form_id][view_id])



    print("test_file_paths",len(test_file_paths))
    print("train_file_paths",len(train_file_paths))
    print("")

    with open(os.path.join(root, "datasets", f"mediapipe_test_file_paths_{args.keypoint_type}.json"), 'w') as f:
        json.dump({"all_file_paths": test_file_paths, "form_file_paths_dict": test_form_file_paths_dict}, f)
    print('>>> Saved test json')

    with open(os.path.join(root, "datasets", f"mediapipe_train_file_paths_{args.keypoint_type}.json"), 'w') as f:
        json.dump({"all_file_paths": train_file_paths, "form_file_paths_dict": train_form_file_paths_dict}, f)
    print('>>> Saved train json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--root-path', type=str, default='/mnt/c/Users/user/Music',
    #                     help='root path')

    parser.add_argument('--keypoint-type', type=str, default='3d_world',
                        help='keypoint type')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    parser.add_argument('--remove-video', default=False, action='store_true',
                        help='remove video')

    args = parser.parse_args()

    main(args)

    