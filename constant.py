import torch

default_device = torch.device('cuda')

batch_size = 8

g_output_img_size = 256
g_landmarks_num = 68

eps = 1e-10

LReLu_negative_slope = 0.2
