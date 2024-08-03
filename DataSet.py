import os
import torch
from torch.utils.data import Dataset
from ImgData import ImgData


class WLSDataSet(Dataset):
    def __init__(self, test_case_list):
        self.test_case_list = test_case_list

    def __getitem__(self, index):
        lp_img_land = ImgData(self.test_case_list[index][0]).img_landmarks
        g_img_land_list = []
        file_list = [f for f in os.listdir(self.test_case_list[index][1]) if os.path.isfile(os.path.join(self.test_case_list[index][1], f))]
        for f in file_list:
            g_img_path = os.path.isfile(os.path.join(self.test_case_list[index][1], f))
            if self.test_case_list[index][0] != g_img_path:
                g_img_land = ImgData(g_img_path).img_landmarks
                g_img_land_list.append(g_img_land)
        return lp_img_land, g_img_land_list

    def __len__(self):
        return len(self.test_case_list)
