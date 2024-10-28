import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import time
import os.path as osp
import os

fea_map_out_path = './TestExamples/FeatureMap'
c_time = time.strftime("%m-%d_%H-%M", time.localtime())
fea_map_out_path = osp.join(fea_map_out_path + '_' + c_time)
os.makedirs(fea_map_out_path, exist_ok=True)

def visualize_feature_map(feature_map, decripsion):  # 특징 맵 시각화 함수
    num_channels = feature_map.shape[1]

    # num_channels가 정사각형 배열로 가능한지 확인
    sqrt_channels = int(math.sqrt(num_channels))

    if sqrt_channels ** 2 == num_channels:
        # 정사각형 배열로 가능한 경우
        rows = cols = sqrt_channels
    else:
        # 정사각형 배열로 불가능한 경우, 두 개의 정사각형 배열로 나눔
        sqrt_channels = int(math.sqrt(num_channels/2))
        rows = sqrt_channels
        cols = sqrt_channels * 2

    # 동적으로 figsize 계산 (가로 2, 세로 2 배율 기준)
    fig_width = cols
    fig_height = rows

    plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(decripsion)
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map[0, i].detach().cpu().numpy(), cmap='viridis')  # 컬러맵 설정
        plt.axis('off')
    # 그래프 간격 조정
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, bottom=0.05, right=0.95, top=0.95)  # 가로, 세로 간격 조정
    plt.savefig(osp.join(fea_map_out_path, decripsion + '.png'))
    plt.close()
