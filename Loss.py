import torch

import PreTrainedVGGFace
from torchvision.transforms.functional import normalize
from torchvision.transforms import Resize
from constant import *

# VGGFaceModel = PreTrainedVGGFace.vgg_face_dag("vgg_face_dag.pth")
VGGFaceModel = PreTrainedVGGFace.vgg_m_face_bn_dag("vgg_m_face_bn_dag.pth")
resize_transform = Resize((VGGFaceModel.meta['imageSize'][0], VGGFaceModel.meta['imageSize'][1]))

def ASFFMSELoss(I_h, I_truth):
    mse = torch.mean((I_h - I_truth) ** 2)
    return mse


def GetVGGFaceFea(I_h, I_truth):
    def asff_output_to_vgg_input(asff_out):
        vgg_input = torch.tensor([]).to(default_device)
        for i in range(asff_out.size(0)):
            vgg_input = torch.cat((vgg_input, normalize(asff_out[i], VGGFaceModel.meta['mean'], VGGFaceModel.meta['std']).unsqueeze(0)), 0)
        return vgg_input

    I_h_fea_list = list(VGGFaceModel(asff_output_to_vgg_input(I_h)))
    I_truth_fea_list = list(VGGFaceModel(asff_output_to_vgg_input(I_truth)))
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
        fea_2d = torch.reshape(tensor, new_shape)
        gram = fea_2d @ fea_2d.transpose(1, 0)
        return gram

    def cal_diff(I_h_fea: torch.Tensor, I_truth_fea: torch.Tensor):
        I_h_gram = ToGramMatrix(I_h_fea)
        I_truth_gram = ToGramMatrix(I_truth_fea)
        norm = torch.mean((I_h_gram - I_truth_gram) ** 2)
        return norm

    ret = 0
    for I_h_fea, I_truth_fea in zip(I_h_fea_list, I_truth_fea_list):
        batch_size = I_h_fea.shape[0]
        for b in range(batch_size):
            ret = ret + cal_diff(I_h_fea[b], I_truth_fea[b])/batch_size
    return ret


def ASFFadvDLoss(I_h_D_output, I_truth_D_output):
    I_truth_E = torch.mean(torch.minimum(torch.zeros_like(I_truth_D_output), -1 + I_truth_D_output))
    I_h_E = torch.mean(torch.minimum(torch.zeros_like(I_h_D_output), -1 - I_h_D_output))
    return I_truth_E + I_h_E


def ASFFadvGLoss(I_h_D_output):
    I_d_E = torch.mean(I_h_D_output)
    return I_d_E


def ASFFGLoss(I_h, I_truth, I_h_D_output):
    resize_I_h = resize_transform(I_h)
    resize_I_truth = resize_transform(I_truth)
    I_h_fea_list, I_truth_fea_list = GetVGGFaceFea(resize_I_h, resize_I_truth)
    G_style_loss = ASFFStyleLoss(I_h_fea_list, I_truth_fea_list)
    G_perc_loss = ASFFPercLoss(I_h_fea_list, I_truth_fea_list)
    G_adv_loss = ASFFadvGLoss(I_h_D_output)
    G_mse_loss = ASFFMSELoss(I_h, I_truth)

    G_total_loss = tradeoff_parm_style * G_style_loss + tradeoff_parm_perc * G_perc_loss + tradeoff_parm_adv * G_adv_loss + tradeoff_parm_mse * G_mse_loss
    return {
        "mse_loss": G_mse_loss,
        "perc_loss": G_perc_loss,
        "style_loss": G_style_loss,
        "adv_loss": G_adv_loss,
        "total_loss": G_total_loss
    }


def ASFFDLoss(I_h_D_output, I_truth_D_output):
    D_loss = ASFFadvDLoss(I_h_D_output, I_truth_D_output)
    return D_loss


if __name__ == '__main__':
    # 원본 리스트를 텐서로 변환
    array = torch.tensor([1, 2, 3])

    # unsqueeze를 사용해 차원을 확장한 후, repeat로 복사
    expanded_tensor = array.unsqueeze(1).repeat(1, 5 * 5).reshape(3, 5, 5).unsqueeze(1).repeat(2,1,1,1)

    print(expanded_tensor)
    # print("style loss test")
