from model.AdaIN import AdaIN, AdaIN2
from Data.DataSet import ASFFDataSet
import os
import cv2
import numpy as np
import torch
import util.DirectoryUtils as DirectoryUtils


def tensor_to_img_mat(tensor):
    # 텐서를 numpy 배열로 변환하고 데이터 타입을 uint8로 변환
    img_out = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # RGB를 BGR로 변환 (cv2는 BGR을 사용)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    return img_out


def tensor_to_img(tensor):
    img_out = tensor * 0.5 + 0.5
    img_out = torch.clip(img_out, 0, 1) * 255.0
    return img_out


test_data_set_path = DirectoryUtils.select_file("test data list csv")
wls_weight_path = DirectoryUtils.select_file("wls weight path")

test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
asff_test_data = ASFFDataSet(test_data_list, wls_weight_path)

adain = AdaIN2()

idx = 0
while True:
    case_num = int(input('실행할 케이스 번호를 입력하세요. '))
    data = asff_test_data[case_num]

    I_g = tensor_to_img(data['g_img_tensor'].unsqueeze(0))
    L_g = data['g_img_landmarks_tensor'].unsqueeze(0)
    I_hq = tensor_to_img(data['hp_img_tensor'].unsqueeze(0))
    L_d = data['lp_landmarks_tensor'].unsqueeze(0)

    I_transformed = adain(I_g, I_hq)

    g_img = tensor_to_img_mat(I_g.squeeze(0))
    hp_img = tensor_to_img_mat(I_hq.squeeze(0))
    transformed_img = tensor_to_img_mat(I_transformed.squeeze(0))

    h_concat_img = cv2.hconcat([g_img, hp_img, transformed_img])

    # 연결된 이미지 저장
    cv2.imwrite('adain_test{}.jpg'.format(idx), h_concat_img)
    idx += 1
