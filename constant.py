import torch

default_device = torch.device('cuda')

g_batch_size = 8

g_output_img_size = 256
g_landmarks_num = 68

eps = 1e-10

LReLu_negative_slope = 0.2

img_list_csv_file_name = 'img_list.csv'

tradeoff_parm_mse = 300
tradeoff_parm_perc = 5
tradeoff_parm_style = 1
tradeoff_parm_adv = 2
