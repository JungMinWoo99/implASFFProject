import numpy as np
import matplotlib.pyplot as plt
import torch
import math


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

    plt.figure(figsize=(10, 5))
    plt.suptitle(decripsion)
    # 전체 채널 중 고르게 8개 선택
    #indices = np.linspace(0, num_channels - 1, 16, dtype=int)
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map[0, i].detach().cpu().numpy(), cmap='viridis')  # 컬러맵 설정
        plt.axis('off')
    # 그래프 간격 조정
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 가로, 세로 간격 조정
    plt.show()
