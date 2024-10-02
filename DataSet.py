import os

import torch
from torch.utils.data import Dataset
from ImgData import ImgData
from model.WLS import WLS


class WLSDataSet(Dataset):
    def __init__(self, test_case_list):
        self.test_case_list = test_case_list

    def __getitem__(self, index):
        lp_img_land = ImgData(self.test_case_list[index][0]).img_landmarks_tensor
        g_img_land_list = []
        file_list = [f for f in os.listdir(self.test_case_list[index][1]) if
                     os.path.isfile(os.path.join(self.test_case_list[index][1], f))]
        for f in file_list:
            g_img_path = os.path.join(self.test_case_list[index][1], f)
            file_name, file_extension = os.path.splitext(g_img_path)
            if file_extension == '.png':
                if os.path.basename(self.test_case_list[index][0]) != os.path.basename(g_img_path):
                    g_img_land = ImgData(g_img_path).img_landmarks_tensor
                    g_img_land_list.append(g_img_land)
        if len(g_img_land_list) == 0:
            print("no g_img: " + self.test_case_list[index][0])
        return lp_img_land, g_img_land_list

    def __len__(self):
        return len(self.test_case_list)


class ASFFDataSet(Dataset):
    def __init__(self, test_case_list, wls_weight_path):
        self.test_case_list = test_case_list
        self.g_img_select_model = WLS()
        self.g_img_select_model.w = torch.load(wls_weight_path)['wls_weight']

    def __getitem__(self, index):
        lp_img = ImgData(self.test_case_list[index][0])
        hp_img = ImgData(self.test_case_list[index][2])

        lp_img_land = lp_img.img_landmarks_tensor
        g_img_land_list = []
        g_img_list = []
        file_list = [f for f in os.listdir(self.test_case_list[index][1]) if
                     os.path.isfile(os.path.join(self.test_case_list[index][1], f))]
        for f in file_list:
            g_img_path = os.path.join(self.test_case_list[index][1], f)
            file_name, file_extension = os.path.splitext(g_img_path)
            if file_extension == '.png':
                if os.path.basename(self.test_case_list[index][0]) != os.path.basename(g_img_path):
                    g_img = ImgData(g_img_path)
                    g_img_list.append(g_img)
                    g_img_land = g_img.img_landmarks_tensor
                    g_img_land_list.append(g_img_land)
        if len(g_img_land_list) == 0:
            print("no g_img: " + self.test_case_list[index][0])
            exit(-1)
        else:
            _, g_img_index = self.g_img_select_model.compute_loss(lp_img_land, g_img_land_list)
        return {
            "lp_img_tensor": lp_img.img_tensor,
            "g_img_tensor": g_img_list[g_img_index].img_tensor,
            "lp_land_bin_img_tensor": lp_img.get_bin_img_tensor(),
            "lp_landmarks_tensor": lp_img.img_landmarks_tensor,
            "g_img_landmarks_tensor": g_img_list[g_img_index].img_landmarks_tensor,
            "hp_img_tensor": hp_img.img_tensor
        }

    def __len__(self):
        return len(self.test_case_list)


class TruthImgDataSet(Dataset):
    def __init__(self, test_case_list):
        self.test_case_list = test_case_list

    def __getitem__(self, index):
        img = ImgData(self.test_case_list[index][2])
        return img.img_tensor, 1

    def __len__(self):
        return len(self.test_case_list)


if __name__ == '__main__':
    from util import DirectoryUtils

    train_data_set_path = DirectoryUtils.select_file("train data list csv")
    wls_weight_path = DirectoryUtils.select_file("wls weight path")

    data_list = DirectoryUtils.read_list_from_csv(train_data_set_path)
    asff_train_data = ASFFDataSet(data_list, wls_weight_path)
    for data in asff_train_data:
        for d in data:
            print(d, data[d].shape)
