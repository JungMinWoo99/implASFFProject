import torch
from torch import nn
import torch.optim as optim
import cv2
import numpy as np
import os.path as osp
import time
import face_alignment  # pip install face-alignment or conda install -c 1adrianb face_alignment
import argparse
import os
import json
import math
from torchvision.transforms.functional import normalize

FaceDetection = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                             device='cuda' if torch.cuda.is_available() else 'cpu')

g_output_img_size = 256
g_landmarks_num = 68


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
        img_landmarks_tensor = torch.from_numpy(selected_landmarks).transpose(0, 1)
        return selected_landmarks, img_landmarks_tensor

    def _get_img_tensor(self):
        normalized_img_mat = self.img_mat.transpose((2, 0, 1)) / 255.0
        img_tensor = torch.from_numpy(normalized_img_mat).float()
        normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor


class WLS:
    def __init__(self, init_w = None):
        if init_w is None:
            rand_tensor = torch.rand(g_landmarks_num) * 0.01
        else:
            rand_tensor = init_w
        self.w = rand_tensor.requires_grad_(True)  # define weight
        self.w_diag = torch.diag(self.w)
        print(self.w_diag)

    def compute_loss(self, L_d, L_g_list):
        distance_list = []
        for L_k in L_g_list:
            # cal affine transform
            h_L_k = torch.cat([L_k, torch.ones(1, g_landmarks_num)], dim=0)  # to homogeneous coordinate
            inner_matrix = h_L_k @ self.w_diag @ h_L_k.transpose(0, 1)
            inverse_matrix = torch.inverse(inner_matrix)
            A = L_d @ self.w_diag @ h_L_k.transpose(0, 1) @ inverse_matrix
            error = A @ h_L_k - L_d
            affine_distance = torch.sum(self.w * torch.norm(error, dim=0)**2.0)
            distance_list.append(affine_distance)
        print(distance_list)
        shortest_distance = min(distance_list)
        min_index = distance_list.index(shortest_distance)
        del distance_list[min_index]
        distance_tensor = torch.tensor(distance_list)
        loss_tensor = torch.max(torch.zeros_like(distance_tensor), 1.0 - (distance_tensor - shortest_distance))
        loss = torch.sum(loss_tensor)
        return loss, min_index


def cal_sim(lq_landmarks, ref_landmark):
    weight_eye = 1.01
    weight_mouth = 1.49

    # lq landmark process    (68,2)
    lq_landmarks_eye = np.concatenate([lq_landmarks[17:27, :], lq_landmarks[36:48, :]], axis=0)  # eye+eyebrow landmark

    # ref landmark process
    Ref_Ab = np.insert(ref_landmark, 2, 1, -1)
    Ref_Ab_eye = np.concatenate([Ref_Ab[17:27, :], Ref_Ab[36:48, :]], axis=0)  # eyebrow+eye (22, 3)
    Ref_Ab_mouth = Ref_Ab[48:, :]  # mouth #(20,2)

    # eye
    result_Ab_eye = np.dot(np.dot(np.linalg.inv(np.dot(Ref_Ab_eye.T, Ref_Ab_eye)), Ref_Ab_eye.T),
                           lq_landmarks_eye)  # (3, 2)
    # mouth
    result_Ab_mouth = np.dot(np.dot(np.linalg.inv(np.dot(Ref_Ab_mouth.T, Ref_Ab_mouth)), Ref_Ab_mouth.T),
                             lq_landmarks[48:, :])  # (3, 2)

    ref_landmark_align_eye = np.dot(Ref_Ab_eye, result_Ab_eye.reshape([3, 2]))  # transposed eye landmark (22, 2)
    ref_landmark_align_mouth = np.dot(Ref_Ab_mouth,
                                      result_Ab_mouth.reshape([3, 2]))  # transposed mouth landmark (20, 2)
    Sim = weight_eye * np.linalg.norm(ref_landmark_align_eye - lq_landmarks_eye) + weight_mouth * np.linalg.norm(
        ref_landmark_align_mouth - lq_landmarks[48:, :])
    return Sim


if __name__ == '__main__':
    torch.set_printoptions(precision=10)

    # 파일 경로 설정
    l_d_file = "./l_d.json"
    l_g_list_file = "./l_g_list.json"

    # 파일이 존재하지 않을 때만 데이터 생성
    if not (os.path.exists(l_d_file) and os.path.exists(l_g_list_file)):
        # 데이터 생성 코드
        lq_img_data = ImgData("./TestExamples/LQCrop/i3.png")
        ref_img_set = []
        for i in range(9):
            ref_img_set.append(ImgData("./TestExamples/HQReferences/Obama/o{}.png".format(i + 1)))
        L_d = lq_img_data.img_landmarks_tensor
        L_g_list = []
        for img_data in ref_img_set:
            L_g_list.append(img_data.img_landmarks_tensor)

        # 데이터를 파일에 저장
        with open(l_d_file, 'w') as f:
            json.dump(L_d.tolist(), f)

        with open(l_g_list_file, 'w') as f:
            json.dump([item.tolist() for item in L_g_list], f)
    else:
        # 파일에서 데이터 불러오기
        with open(l_d_file, 'r') as f:
            L_d = torch.tensor(json.load(f))

        with open(l_g_list_file, 'r') as f:
            L_g_list = [torch.tensor(item) for item in json.load(f)]

    tensor = torch.zeros(68)
    tensor[17:27] = 1.01**0.5
    tensor[36:48] = 1.01**0.5
    tensor[48:68] = 1.49**0.5

    wls = WLS(init_w=tensor)
    ret, answer = wls.compute_loss(L_d, L_g_list)
    print(ret)

    test_list = []
    for ref_L in L_g_list:
        test_list.append(cal_sim(L_d.transpose(0, 1).numpy(), ref_L.transpose(0, 1).numpy()))
    print(test_list)

    """
    optimizer = optim.Adam([wls.w], lr=0.0001)

    for i in range(10000):
        optimizer.zero_grad()
        ret, answer = wls.compute_loss(L_d, L_g_list)
        print(ret)
        if ret == 0:
            print(answer)
            print(wls.w)
            break
        ret.backward()
        optimizer.step()
    """
