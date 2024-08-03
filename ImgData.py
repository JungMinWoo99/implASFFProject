from constant import *
import numpy as np
import torch
import cv2
import face_alignment  # pip install face-alignment or conda install -c 1adrianb face_alignment
from torchvision.transforms.functional import normalize

FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')


class ImgData:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_mat = self._read_and_resize_img()
        self.img_tensor = self._get_img_tensor()
        self.img_landmarks, self.img_landmarks_tensor = self._get_img_landmarks()

    def _read_and_resize_img(self):
        img_mat = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img_mat.ndim == 2:
            img_mat = cv2.cvtColor(img_mat, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)  # RGB
        if img_mat.shape[0] > g_output_img_size or img_mat.shape[1] > g_output_img_size:
            img_mat = cv2.resize(img_mat, (g_output_img_size, g_output_img_size), interpolation=cv2.INTER_AREA)
        return img_mat

    def _get_img_landmarks(self):
        idx = 0
        try:
            img_landmarks = FaceDetection.get_landmarks_from_image(self.img_mat)
        except:
            print('Error in detecting this face {}. Continue...'.format(self.img_path))
        if img_landmarks is None:
            print('Warning: No face is detected in {}. Continue...'.format(self.img_path))
        elif len(img_landmarks) != 1:
            hights = []
            for l in img_landmarks:
                hights.append(l[8, 1] - l[19, 1])  # choose the largest face
            idx = hights.index(max(hights))
            print('Warning: Too many faces are detected in {}, only handle the largest one...'.format(self.img_path))
        selected_landmarks = img_landmarks[idx]
        img_landmarks_tensor = torch.from_numpy(selected_landmarks).transpose(0, 1).to(default_device)
        return selected_landmarks, img_landmarks_tensor

    def _get_img_tensor(self):
        normalized_img_mat = self.img_mat.transpose((2, 0, 1)) / 255.0
        img_tensor = torch.from_numpy(normalized_img_mat).float()
        normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        img_tensor = img_tensor.unsqueeze(0).to(default_device)
        return img_tensor

    def get_bin_img_tensor(self):
        l_point_scale = 2
        landmarks_bin_img_tensor = torch.zeros((1, g_output_img_size, g_output_img_size)).to(default_device)
        for landmark in self.img_landmarks:
            x, y = landmark
            if x-l_point_scale >= 0 and x+l_point_scale < g_output_img_size and y-l_point_scale >= 0 and y+l_point_scale < g_output_img_size:
                landmarks_bin_img_tensor[0, int(x-l_point_scale):int(x+l_point_scale), int(y-l_point_scale):int(y+l_point_scale)] = 1
        return landmarks_bin_img_tensor




if __name__ == '__main__':
    test_img = ImgData("E:\code_depository\depository_python\FSR_project\ImpASFF\sample\i1.png")
    img_arr = (test_img.get_bin_img_tensor().transpose(0, 2).cpu().numpy() * 255).astype(np.uint8)
    bgr_test_img = cv2.cvtColor(test_img.img_mat, cv2.COLOR_RGB2BGR)
    cv2.imshow('origin', bgr_test_img)
    print(bgr_test_img.shape)
    print(img_arr.shape)
    cv2.imshow('land', img_arr)
    cv2.waitKey()
