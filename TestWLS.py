from DataSet import WLSDataSet
from util import DirectoryUtils
from torch.utils.data import DataLoader
from model.WLS import WLS
import torch
import numpy as np
from constant import *

def cal_sim(L_d, L_g_list): # 원작자가 구현한 WLS
    distance_list = []
    lq_landmarks = L_d.transpose(0, 1).cpu().numpy()
    for L_k in L_g_list:
        ref_landmark = L_k.transpose(0, 1).cpu().numpy()
        weight_eye = 1.01
        weight_mouth = 1.49

        # lq landmark process    (68,2)
        lq_landmarks_eye = np.concatenate([lq_landmarks[17:27, :], lq_landmarks[36:48, :]],
                                          axis=0)  # eye+eyebrow landmark

        # ref landmark process
        Ref_Ab = np.insert(ref_landmark, 2, 1, -1)
        Ref_Ab_eye = np.concatenate([Ref_Ab[17:27, :], Ref_Ab[36:48, :]], axis=0)  # eyebrow+eye (22, 3)
        Ref_Ab_mouth = Ref_Ab[48:, :]  # mouth #(20,3)

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
        distance_list.append(Sim)
    min_index = distance_list.index(min(distance_list))
    return min_index

test_data_set_path = DirectoryUtils.select_file("test data list csv")
wls_weight_path = DirectoryUtils.select_file("wls weight path")

wls = WLS()
wls.w = torch.load(wls_weight_path)['wls_weight']

test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
wls_test_data = WLSDataSet(test_data_list)


def custom_collate_fn(batch):
    return batch[0]  # 배치 차원을 제거


test_dataloader = DataLoader(
    wls_test_data,  # 위에서 생성한 데이터 셋
    batch_size=1,
    shuffle=False,  # 데이터들의 순서는 섞어서 분할
    collate_fn=custom_collate_fn
)

while True:
    test_case = int(input())
    print(test_data_list[test_case])
    ret, answer = wls.compute_loss(wls_test_data[test_case][0], wls_test_data[test_case][1])
    test_answer = cal_sim(wls_test_data[test_case][0], wls_test_data[test_case][1])
    print(ret, answer)
    print(test_answer)
