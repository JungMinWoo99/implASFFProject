from tqdm import tqdm
from util import DirectoryUtils
from downscaling.DownScaling import *
import os


def save_lq_img(hq_img_dataset_path, lq_img_dataset_path):
    DirectoryUtils.copy_subfolders(hq_img_dataset_path, lq_img_dataset_path)
    img_subfolder_list = [f for f in os.listdir(hq_img_dataset_path) if
                          os.path.isdir(os.path.join(hq_img_dataset_path, f))]

    for folder_name in tqdm(img_subfolder_list):
        hq_img_sub_folder_dir = os.path.join(hq_img_dataset_path, folder_name)
        lq_img_sub_folder_dir = os.path.join(lq_img_dataset_path, folder_name)
        img_file_list = [f for f in os.listdir(hq_img_sub_folder_dir) if
                         os.path.isfile(os.path.join(hq_img_sub_folder_dir, f))]
        for f in img_file_list:
            hq_img_file_path = os.path.join(hq_img_sub_folder_dir, f)
            lq_img_file_path = os.path.join(lq_img_sub_folder_dir, f)
            file_name, file_extension = os.path.splitext(hq_img_file_path)
            if file_extension == '.png':
                if os.path.exists(lq_img_file_path):
                    continue
                else:
                    lq_img = degradation_model(hq_img_file_path, motion_blur_kernels=motion_blur_kernels)
                    cv2.imwrite(lq_img_file_path, cv2.cvtColor(lq_img, cv2.COLOR_RGB2BGR))
