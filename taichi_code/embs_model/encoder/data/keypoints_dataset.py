import os
import json
from torch.utils.data.dataset import Dataset

def read_keypoints_json_file(json_file_path):
    with open(json_file_path, "r") as file:
        json_file = json.load(file)    
    all_file_paths = json_file["all_file_paths"]
    form_file_paths_dict = {}
    for key, value in json_file["form_file_paths_dict"].items():
        form_file_paths_dict[key] = value

    return all_file_paths, form_file_paths_dict


class Keypoints_Paths_Dataset(Dataset):
    def __init__(self, json_3d_file_path=None):
        assert json_3d_file_path, 'No json file'
        self.all_3d_file_paths, self.form_3d_file_paths_dict = read_keypoints_json_file(json_3d_file_path)
        
    def __len__(self):
        return len(self.all_3d_file_paths)

    def __getitem__(self, idx):
        return self.all_3d_file_paths[idx]

