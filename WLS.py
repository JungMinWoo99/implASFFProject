from constant import *
import torch
from torch import nn
import torch.optim as optim
import cv2
import numpy as np
import os.path as osp
import time
import argparse
import os
import json
from ImgData import ImgData


class WLS:
    def __init__(self, init_w=None):
        if init_w is None:
            rand_tensor = torch.rand(g_landmarks_num) * 0.01
        else:
            rand_tensor = init_w
        self.w = rand_tensor.to(default_device).requires_grad_(True)  # define weight
        self.w_diag = torch.diag(self.w).to(default_device)

    def compute_loss(self, L_d, L_g_list):
        distance_list = []
        for L_k in L_g_list:
            # cal affine transform
            h_L_k = torch.cat([L_k, torch.ones(1, g_landmarks_num).to(default_device)],
                              dim=0)  # to homogeneous coordinate
            inner_matrix = h_L_k @ self.w_diag @ h_L_k.transpose(0, 1)
            inverse_matrix = torch.inverse(inner_matrix)
            A = L_d @ self.w_diag @ h_L_k.transpose(0, 1) @ inverse_matrix
            error = A @ h_L_k - L_d
            affine_distance = torch.sum(self.w * torch.norm(error, dim=0) ** 2.0)
            distance_list.append(affine_distance)
        shortest_distance = min(distance_list)
        min_index = distance_list.index(shortest_distance)
        del distance_list[min_index]
        distance_tensor = torch.tensor(distance_list).to(default_device)
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
    from DataSet import WLSDataSet
    import DirectoryUtils
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    train_data_set_path = DirectoryUtils.select_file()
    test_data_set_path = DirectoryUtils.select_file()
    wls = WLS()
    train_data_list = DirectoryUtils.read_list_from_csv(train_data_set_path)
    test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
    wls_train_data = WLSDataSet(train_data_list)
    wls_test_data = WLSDataSet(test_data_list)


    def custom_collate_fn(batch):
        return batch[0]  # 배치 차원을 제거


    train_dataloader = DataLoader(
        wls_train_data,  # 위에서 생성한 데이터 셋
        batch_size=1,
        shuffle=True,  # 데이터들의 순서는 섞어서 분할
        collate_fn=custom_collate_fn
    )

    test_dataloader = DataLoader(
        wls_test_data,  # 위에서 생성한 데이터 셋
        batch_size=1,
        shuffle=True,  # 데이터들의 순서는 섞어서 분할
        collate_fn=custom_collate_fn
    )

    mode = 'run'
    if mode == 'train':
        optimizer = optim.Adam([wls.w], lr=0.0001)

        loss_list = []
        epoch = 20
        for e in range(epoch):
            for data in tqdm(train_dataloader, desc='Processing Batches'):
                optimizer.zero_grad()
                loss, answer = wls.compute_loss(data[0], data[1])
                loss.backward()
                optimizer.step()
            loss_sum = 0
            for data in tqdm(test_dataloader, desc='calculate loss'):
                loss, answer = wls.compute_loss(data[0], data[1])
                loss_sum += loss.item()
            loss_avg = loss_sum / len(wls_test_data)
            loss_list.append(loss_avg)
            # 가중치 텐서 저장
            torch.save(wls.w, 'weight_tensor{}.pth'.format(e))

        import matplotlib.pyplot as plt
        # 꺾은선 그래프 그리기
        plt.plot([i for i in range(1, epoch+1)], loss_list, marker='o', linestyle='-')

        # 각 데이터 포인트에 값 표시
        for i, value in enumerate(loss_list):
            plt.plot([i for i in range(1, epoch + 1)], loss_list, marker='o', linestyle='-')
            plt.annotate(f'{value:.5f}', (i, loss_list[i]), textcoords="offset points", xytext=(0, 10), ha='center')

        # 그래프 제목과 축 레이블
        plt.title('loss avg')
        plt.xlabel('epoch')
        plt.ylabel('loss')

        # 그래프 저장
        plt.savefig('line_plot_with_values.png')

        # 그래프 보여주기
        plt.show()
    elif mode == 'run':
        wls.w = torch.load('weight_tensor15.pth')
        while True:
            test_case = int(input())
            print(test_data_list[test_case])
            ret, answer = wls.compute_loss(wls_test_data[test_case][0], wls_test_data[test_case][1])
            print(answer)

'''
if __name__ == '__main123__':
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
    tensor[17:27] = 1.01 ** 0.5
    tensor[36:48] = 1.01 ** 0.5
    tensor[48:68] = 1.49 ** 0.5

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
'''
