import sys
import os

import torch

# 모듈화 이후 지울 코드
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import PreTrainedVGGFace

import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from torchvision.transforms import Resize
from constant import *

mse_loss = torch.nn.MSELoss()
perc_loss = torch.nn.MSELoss()
style_loss = torch.nn.MSELoss()


def output_to_img(tensor):
    img_out = tensor * 0.5 + 0.5
    img_out = torch.clip(img_out, 0, 1) * 255.0
    return img_out


def output_denormalization(tensor):
    d_tensor = tensor * 0.5 + 0.5
    ret = torch.clip(d_tensor, 0, 1)
    return ret


def GetVGGFaceFea_ver1(I_h, I_truth):
    if not hasattr(GetVGGFaceFea_ver1, "VGGFaceModel"):
        GetVGGFaceFea_ver1.VGGFaceModel = PreTrainedVGGFace.vgg_face_dag("./pretrained_weight/vgg_face_dag.pth").eval()
    if not hasattr(GetVGGFaceFea_ver1, "resize_transform"):
        GetVGGFaceFea_ver1.resize_transform = Resize((GetVGGFaceFea_ver1.VGGFaceModel.meta['imageSize'][0],
                                                      GetVGGFaceFea_ver1.VGGFaceModel.meta['imageSize'][1]))
    img_h = output_to_img(I_h)
    img_truth = output_to_img(I_truth)
    resize_img_h = GetVGGFaceFea_ver1.resize_transform(img_h)
    resize_img_truth = GetVGGFaceFea_ver1.resize_transform(img_truth)

    def asff_output_to_vgg_input(asff_out):
        vgg_input = torch.tensor([]).to(default_device)
        for i in range(asff_out.size(0)):
            vgg_input = torch.cat(
                (vgg_input, normalize(asff_out[i], GetVGGFaceFea_ver1.VGGFaceModel.meta['mean'],
                                      GetVGGFaceFea_ver1.VGGFaceModel.meta['std']).unsqueeze(0)),
                0)
        return vgg_input

    I_h_fea_list = list(GetVGGFaceFea_ver1.VGGFaceModel(asff_output_to_vgg_input(resize_img_h)))
    I_truth_fea_list = list(GetVGGFaceFea_ver1.VGGFaceModel(asff_output_to_vgg_input(resize_img_truth)))
    return I_h_fea_list, I_truth_fea_list


def GetVGGFaceFea_ver2(I_h, I_truth):
    if not hasattr(GetVGGFaceFea_ver2, "VGGFaceModel"):
        GetVGGFaceFea_ver2.VGGFaceModel = PreTrainedVGGFace.vggface("./pretrained_weight/vggface-9d491dd7c30312.pth").eval()
    if not hasattr(GetVGGFaceFea_ver2, "resize_transform"):
        GetVGGFaceFea_ver2.resize_transform = Resize((GetVGGFaceFea_ver2.VGGFaceModel.meta['imageSize'][0],
                                                      GetVGGFaceFea_ver2.VGGFaceModel.meta['imageSize'][1]))
    resize_I_h = GetVGGFaceFea_ver2.resize_transform(I_h)
    resize_I_truth = GetVGGFaceFea_ver2.resize_transform(I_truth)

    def asff_output_to_vgg_input(asff_out):
        vgg_input = torch.tensor([]).to(default_device)
        for i in range(asff_out.size(0)):
            vgg_input = torch.cat(
                (vgg_input, normalize(asff_out[i], GetVGGFaceFea_ver2.VGGFaceModel.meta['mean'],
                                      GetVGGFaceFea_ver2.VGGFaceModel.meta['std']).unsqueeze(0)),
                0)
        return vgg_input

    I_h_fea_list = list(GetVGGFaceFea_ver2.VGGFaceModel(asff_output_to_vgg_input(resize_I_h)))
    I_truth_fea_list = list(GetVGGFaceFea_ver2.VGGFaceModel(asff_output_to_vgg_input(resize_I_truth)))
    return I_h_fea_list, I_truth_fea_list


def GetVGGFaceFea_ver3(I_h, I_truth):
    if not hasattr(GetVGGFaceFea_ver3, "VGGFaceModel"):
        GetVGGFaceFea_ver3.VGGFaceModel = PreTrainedVGGFace.vgg_face_resnet_ver("./pretrained_weight/resnet50_ft_weight.pkl").eval()
    if not hasattr(GetVGGFaceFea_ver3, "resize_transform"):
        GetVGGFaceFea_ver3.resize_transform = Resize((GetVGGFaceFea_ver3.VGGFaceModel.meta['imageSize'][0],
                                                      GetVGGFaceFea_ver3.VGGFaceModel.meta['imageSize'][1]))
    img_h = output_to_img(I_h)
    img_truth = output_to_img(I_truth)
    resize_img_h = GetVGGFaceFea_ver3.resize_transform(img_h)
    resize_img_truth = GetVGGFaceFea_ver3.resize_transform(img_truth)

    def asff_output_to_vgg_input(asff_out):
        vgg_input = torch.tensor([]).to(default_device)
        for i in range(asff_out.size(0)):
            vgg_input = torch.cat(
                (vgg_input, normalize(asff_out[i], GetVGGFaceFea_ver3.VGGFaceModel.meta['mean'],
                                      GetVGGFaceFea_ver3.VGGFaceModel.meta['std']).unsqueeze(0)),
                0)
        return vgg_input

    I_h_fea_list = list(GetVGGFaceFea_ver3.VGGFaceModel(asff_output_to_vgg_input(resize_img_h)))
    I_truth_fea_list = list(GetVGGFaceFea_ver3.VGGFaceModel(asff_output_to_vgg_input(resize_img_truth)))
    return I_h_fea_list, I_truth_fea_list


def ASFFMSELoss(I_h, I_truth):
    ret = mse_loss(I_h, I_truth)
    if ret.isnan():
        ret = torch.tensor(eps).to(default_device)
    return ret


def ASFFPercLoss(I_h_fea_list, I_truth_fea_list):
    def cal_diff(I_h_fea: torch.Tensor, I_truth_fea: torch.Tensor):
        ret = perc_loss(I_h_fea, I_truth_fea)
        if ret.isnan():
            ret = torch.tensor(eps).to(default_device)
        return ret

    ret = 0
    for I_h_fea, I_truth_fea in zip(I_h_fea_list, I_truth_fea_list):
        ret = ret + cal_diff(I_h_fea, I_truth_fea)
    return ret


def ASFFStyleLoss(I_h_fea_list, I_truth_fea_list):  # 논문이 수식이 잘 적용되지 않는 것 같음
    def ToGramMatrix(tensor):
        (b, c, h, w) = tensor.size()
        features = tensor.view(b, c, w * h)
        features_transpose = features.transpose(1, 2)
        gram = torch.bmm(features_transpose, features)
        return gram

    def cal_diff(I_h_fea: torch.Tensor, I_truth_fea: torch.Tensor):
        (b, c, h, w) = I_h_fea.size()
        I_h_gram = ToGramMatrix(I_h_fea)
        I_truth_gram = ToGramMatrix(I_truth_fea)
        ret = style_loss(I_h_gram, I_truth_gram) / (b * c * h * w)
        if ret.isnan():
            ret = torch.tensor(eps).to(default_device)
        return ret

    ret = 0
    for I_h_fea, I_truth_fea in zip(I_h_fea_list, I_truth_fea_list):
        ret = ret + cal_diff(I_h_fea, I_truth_fea)
    return ret


def ASFFadvLoss(pred_real, pred_fake=None):
    if pred_fake is not None:
        loss_real = F.relu(1 - pred_real).mean()
        loss_fake = F.relu(1 + pred_fake).mean()
        return loss_real + loss_fake
    else:
        loss = -pred_real.mean()
        return loss


def ASFFGLoss(I_h, I_truth, I_h_D_output):
    normalized_I_h = normalize(I_h, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    normalized_I_truth = normalize(I_truth, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    I_h_fea_list, I_truth_fea_list = GetVGGFaceFea_ver3(I_h, I_truth)
    G_style_loss = ASFFStyleLoss(I_h_fea_list, I_truth_fea_list)
    G_perc_loss = ASFFPercLoss(I_h_fea_list, I_truth_fea_list)
    G_adv_loss = ASFFadvLoss(I_h_D_output)
    G_mse_loss = ASFFMSELoss(normalized_I_h, normalized_I_truth)

    G_total_loss = tradeoff_parm_style * G_style_loss + tradeoff_parm_perc * G_perc_loss + tradeoff_parm_adv * G_adv_loss + tradeoff_parm_mse * G_mse_loss
    return {
        "mse_loss": G_mse_loss,
        "perc_loss": G_perc_loss,
        "style_loss": G_style_loss,
        "adv_loss": G_adv_loss,
        "total_loss": G_total_loss
    }
