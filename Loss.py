import PreTrainedVGGFace
import torch
from constant import *

VGGFaceModel = PreTrainedVGGFace.vgg_face_dag()


def ASFFMSELoss(I_h, I_truth):
    mse = torch.mean((I_h - I_truth) ** 2)
    return mse


def GetVGGFaceFea(I_h, I_truth):
    I_h_fea_list = list(VGGFaceModel(I_h))
    I_truth_fea_list = list(VGGFaceModel(I_truth))
    return I_h_fea_list, I_truth_fea_list


def ASFFPercLoss(I_h_fea_list, I_truth_fea_list):
    def cal_diff(I_h_fea: torch.Tensor, I_truth_fea: torch.Tensor):
        norm = torch.mean(torch.linalg.norm(I_h_fea - I_truth_fea) ** 2)
        return norm

    ret = 0
    for I_h_fea, I_truth_fea in zip(I_h_fea_list, I_truth_fea_list):
        ret = ret + cal_diff(I_h_fea, I_truth_fea)
    return ret


def ASFFStyleLoss(I_h_fea_list, I_truth_fea_list):
    def ToGramMatrix(tensor):
        shape = list(tensor.shape)
        new_shape = shape[:-2] + [shape[-2] * shape[-1]]
        gram = torch.reshape(tensor, new_shape)
        return gram

    def cal_diff(I_h_fea: torch.Tensor, I_truth_fea: torch.Tensor):
        norm = torch.mean(torch.linalg.norm(ToGramMatrix(I_h_fea) - ToGramMatrix(I_truth_fea)) ** 2)
        return norm

    ret = 0
    for I_h_fea, I_truth_fea in zip(I_h_fea_list, I_truth_fea_list):
        ret = ret + cal_diff(I_h_fea, I_truth_fea)
    return ret


def ASFFadvDLoss(I_h_D_output, I_truth_D_output):
    I_truth_E = torch.mean(torch.minimum(torch.zeros_like(I_truth_D_output), -1 + I_truth_D_output))
    I_h_E = torch.mean(torch.minimum(torch.zeros_like(I_h_D_output), -1 - I_h_D_output))
    return I_truth_E + I_h_E


def ASFFadvGLoss(I_h_D_output):
    I_d_E = torch.mean(I_h_D_output)
    return I_d_E


def ASFFGLoss(I_h, I_truth, I_h_D_output):
    batch_size = I_h.shape[0]
    G_loss_sum = torch.zeros(1).to(default_device)
    for n in range(batch_size):
        I_h_fea_list, I_truth_fea_list = GetVGGFaceFea(I_h[n], I_truth[n])
        G_loss_sum = G_loss_sum + tradeoff_parm_perc * ASFFPercLoss(I_h_fea_list, I_truth_fea_list) + \
                      tradeoff_parm_style * ASFFStyleLoss(I_h_fea_list, I_truth_fea_list)
    G_loss = G_loss_sum / batch_size + tradeoff_parm_adv * ASFFadvGLoss(I_h_D_output) + tradeoff_parm_mse * ASFFMSELoss(
        I_h, I_truth)

    return G_loss


def ASFFDLoss(I_h_D_output, I_truth_D_output):
    D_loss = ASFFadvDLoss(I_h_D_output, I_truth_D_output)
    return D_loss
