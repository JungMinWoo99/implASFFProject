from ImgData import ImgData
from tqdm import tqdm
import DirectoryUtils
import os
import numpy as np
import face_alignment  # pip install face-alignment or conda install -c 1adrianb face_alignment

FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

img_dataset_path = DirectoryUtils.select_folder()
land_dataset_path = DirectoryUtils.select_folder()
DirectoryUtils.copy_subfolders(img_dataset_path, land_dataset_path)
img_subfolder_list = [f for f in os.listdir(img_dataset_path) if os.path.isdir(os.path.join(img_dataset_path, f))]

for folder_name in tqdm(img_subfolder_list):
    img_sub_folder_dir = os.path.join(img_dataset_path, folder_name)
    land_sub_folder_dir = os.path.join(land_dataset_path, folder_name)
    img_file_list = [f for f in os.listdir(img_sub_folder_dir) if os.path.isfile(os.path.join(img_sub_folder_dir, f))]
    for f in img_file_list:
        img_file_path = os.path.join(img_sub_folder_dir, f)
        land_file_path = os.path.join(land_sub_folder_dir, f)[:-4] + '.npy'
        file_name, file_extension = os.path.splitext(img_file_path)
        if file_extension == '.png':
            img = ImgData(img_file_path)
            if os.path.exists(land_file_path):
                continue
            else:
                idx = 0
                try:
                    img_landmarks = FaceDetection.get_landmarks_from_image(img.img_mat)
                except:
                    print('Error in detecting this face {}. Continue...'.format(img.img_path))
                if img_landmarks is None:
                    print('Warning: No face is detected in {}. Continue...'.format(img.img_path))
                elif len(img_landmarks) > 3:
                    hights = []
                    for l in img_landmarks:
                        hights.append(l[8, 1] - l[19, 1])  # choose the largest face
                    idx = hights.index(max(hights))
                    print(
                        'Warning: Too many faces are detected in {}, only handle the largest one...'.format(
                            img.img_path))
                selected_landmarks = img_landmarks[idx]
                np.save(land_file_path, selected_landmarks)
