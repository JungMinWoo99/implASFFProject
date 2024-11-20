import cv2
import numpy as np
import model.Loss as L
import torch
from torchvision.transforms.functional import normalize
from constant import *
from util.DirectoryUtils import select_file


def get_img_tensor(img_mat):
    normalized_img_mat = img_mat.transpose((2, 0, 1)) / 255.0
    img_tensor = torch.from_numpy(normalized_img_mat).float()
    normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    img_tensor = img_tensor.to(default_device)
    return img_tensor


while True:
    input_path = select_file("Select the image file where the input, guide, and output of the neural network are combined.")
    if input_path == None:
        break
    stacked_image = cv2.imread(input_path)
    if(stacked_image.shape[0] > g_output_img_size):
        stacked_image = cv2.resize(stacked_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, _, hq, save_out = np.hsplit(stacked_image, 4)
    cv2.imshow("hp",hq)
    cv2.imshow("save",save_out)
    hq = cv2.cvtColor(hq, cv2.COLOR_BGR2RGB)
    save_out = cv2.cvtColor(save_out, cv2.COLOR_BGR2RGB)
    I_h = get_img_tensor(save_out).unsqueeze(0)
    I_truth = get_img_tensor(hq).unsqueeze(0)
    loss = L.ASFFGLoss(I_h, I_truth, torch.tensor(0.0))

    print("\nmse_loss: " + str((tradeoff_parm_mse * loss["mse_loss"])) + \
          "\nperc_loss: " + str((tradeoff_parm_perc * loss["perc_loss"])) + \
          "\nstyle_loss: " + str((tradeoff_parm_style * loss["style_loss"])) + \
          "\ntotal_loss: " + str((loss["total_loss"])))
