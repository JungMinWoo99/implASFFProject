from Data.ImgData import ImgData
from tqdm import tqdm
from util import DirectoryUtils
import os
import numpy as np
import face_alignment  # pip install face-alignment or conda install -c 1adrianb face_alignment

FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

def ext_landmarks(img_mat, img_path=''):
    idx = 0
    try:
        img_landmarks = FaceDetection.get_landmarks_from_image(img_mat)
    except:
        print('Error in detecting this face {}. Continue...'.format(img_path))
    if img_landmarks is None:
        print('Warning: No face is detected in {}. Continue...'.format(img_path))
    elif len(img_landmarks) > 3:
        hights = []
        for l in img_landmarks:
            hights.append(l[8, 1] - l[19, 1])  # choose the largest face
        idx = hights.index(max(hights))
        print(
            'Warning: Too many faces are detected in img, only handle the largest one...')
    selected_landmarks = img_landmarks[idx]
    return selected_landmarks

def save_landmarks(img_dataset_path, land_dataset_path):
    DirectoryUtils.copy_subfolders(img_dataset_path, land_dataset_path)
    img_subfolder_list = [f for f in os.listdir(img_dataset_path) if os.path.isdir(os.path.join(img_dataset_path, f))]

    for folder_name in tqdm(img_subfolder_list):
        img_sub_folder_dir = os.path.join(img_dataset_path, folder_name)
        land_sub_folder_dir = os.path.join(land_dataset_path, folder_name)
        img_file_list = [f for f in os.listdir(img_sub_folder_dir) if
                         os.path.isfile(os.path.join(img_sub_folder_dir, f))]
        for f in img_file_list:
            img_file_path = os.path.join(img_sub_folder_dir, f)
            land_file_path = os.path.join(land_sub_folder_dir, f)[:-4] + '.npy'
            file_name, file_extension = os.path.splitext(img_file_path)
            if file_extension == '.png':
                img = ImgData(img_file_path)
                resized_img_mat = img.img_mat

                selected_landmarks = ext_landmarks(resized_img_mat, img.img_path)

                np.save(land_file_path, selected_landmarks)
