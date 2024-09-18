import torch
import torch.nn as nn
import torch.nn.utils as utils
import constant
import MLS
import AdaIN
import numpy as np


class DilateResBlock(nn.Module):
    def __init__(self, input_ch, kernel, stride, dilation):
        super(DilateResBlock, self).__init__()
        padding_size = ((kernel - 1) // 2) * dilation
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(input_ch, input_ch, kernel_size=kernel, stride=stride, dilation=dilation,
                                          padding=padding_size)),
            nn.BatchNorm2d(input_ch, eps=constant.eps),
            nn.LeakyReLU(constant.LReLu_negative_slope),
            utils.spectral_norm(nn.Conv2d(input_ch, input_ch, kernel_size=kernel, stride=stride, dilation=dilation,
                                          padding=padding_size))
        )

    def forward(self, x):
        out = x + self.Model(x)
        return out


class Conv_BN_LReLU(nn.Module):
    def __init__(self, input_ch, output_ch, kernel, stride, bias):
        super(Conv_BN_LReLU, self).__init__()
        padding_size = ((kernel - 1) // 2)
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=stride,
                                          padding=padding_size, bias=bias)),
            nn.BatchNorm2d(output_ch, eps=constant.eps),
            nn.LeakyReLU(constant.LReLu_negative_slope)
        )

    def forward(self, x):
        return self.Model(x)


class Conv_LReLU(nn.Module):
    def __init__(self, input_ch, output_ch, kernel, stride, bias):
        super(Conv_LReLU, self).__init__()
        padding_size = ((kernel - 1) // 2)
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=stride,
                                          padding=padding_size, bias=bias)),
            nn.LeakyReLU(constant.LReLu_negative_slope)
        )

    def forward(self, x):
        return self.Model(x)


class ASFF(nn.Module):
    def __init__(self):
        super(ASFF, self).__init__()
        self.pointwise_conv_layer1_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, bias=True)
        self.pointwise_conv_layer1_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, bias=True)
        self.pointwise_conv_layer1_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, bias=True)
        self.conv_layer1_1 = Conv_BN_LReLU(192, 128, 3, 1, False)
        self.conv_layer1_2 = Conv_BN_LReLU(128, 128, 3, 1, False)

        self.conv_layer2_1 = Conv_BN_LReLU(128, 128, 3, 1, True)
        self.conv_layer2_2 = Conv_BN_LReLU(128, 128, 3, 1, True)
        self.pointwise_conv_layer2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias=True)
        self.pointwise_conv_layer2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias=True)

    def forward(self, fea_d, fea_gwa, fea_l):
        fea_m = self.conv_layer1_2(self.conv_layer1_1(torch.concat((self.pointwise_conv_layer1_1(fea_d),
                                                                    self.pointwise_conv_layer1_2(fea_gwa),
                                                                    self.pointwise_conv_layer1_3(fea_l)), dim=1)))
        F_d_fea_d = self.pointwise_conv_layer2_1(self.conv_layer2_1(fea_d))
        F_g_fea_gwa = self.pointwise_conv_layer2_2(self.conv_layer2_2(fea_gwa))
        fea_d_new = fea_m * (F_g_fea_gwa - F_d_fea_d) + F_d_fea_d
        return fea_d_new


class ImgFeatureExtractionBlock(nn.Module):
    def __init__(self):
        super(ImgFeatureExtractionBlock, self).__init__()
        self.conv_layer1 = Conv_BN_LReLU(3, 64, 3, 1, True)
        self.drb_layer1 = DilateResBlock(64, 3, 1, 7)
        self.drb_layer2 = DilateResBlock(64, 3, 1, 5)
        self.conv_layer2 = Conv_BN_LReLU(64, 128, 3, 2, True)
        self.drb_layer3 = DilateResBlock(128, 3, 1, 7)
        self.drb_layer4 = DilateResBlock(128, 3, 1, 3)
        self.conv_layer3 = Conv_BN_LReLU(128, 128, 3, 2, True)
        self.drb_layer5 = DilateResBlock(128, 3, 1, 3)
        self.drb_layer6 = DilateResBlock(128, 3, 1, 1)
        self.conv_layer4 = Conv_LReLU(128, 128, 3, 1, True)

    def forward(self, img):
        fea1 = self.drb_layer2(self.drb_layer1(self.conv_layer1(img)))
        fea2 = self.drb_layer4(self.drb_layer3(self.conv_layer2(fea1)))
        fea3 = self.drb_layer6(self.drb_layer5(self.conv_layer3(fea2)))
        fea4 = self.conv_layer4(fea3)
        return fea4


class BinFeatureExtractionBlock(nn.Module):
    def __init__(self):
        super(BinFeatureExtractionBlock, self).__init__()
        self.Model = nn.Sequential(
            Conv_LReLU(1, 64, 9, 2, False),
            Conv_LReLU(64, 64, 3, 1, False),
            Conv_LReLU(64, 64, 7, 1, False),
            Conv_LReLU(64, 128, 3, 1, False),
            Conv_LReLU(128, 128, 5, 2, False),
            Conv_LReLU(128, 128, 3, 1, False),
            Conv_LReLU(128, 128, 3, 1, False),
            Conv_LReLU(128, 128, 3, 1, False),
            Conv_LReLU(128, 128, 3, 1, False),
            Conv_LReLU(128, 128, 3, 1, False),
        )

    def forward(self, l_d):
        return self.Model(l_d)


class FeatureFusionBlock(nn.Module):
    def __init__(self):
        super(FeatureFusionBlock, self).__init__()
        self.ASFFBlock1 = ASFF()
        self.ASFFBlock2 = ASFF()
        self.ASFFBlock3 = ASFF()
        self.ASFFBlock4 = ASFF()

    def forward(self, fea_d, fea_gwa, fea_l):
        fea_d_new1 = self.ASFFBlock1(fea_d, fea_gwa, fea_l)
        fea_d_new2 = self.ASFFBlock2(fea_d_new1, fea_gwa, fea_l)
        fea_d_new3 = self.ASFFBlock3(fea_d_new2, fea_gwa, fea_l)
        fea_c = self.ASFFBlock4(fea_d_new3, fea_gwa, fea_l)
        return fea_c


class ReconstructionBlock(nn.Module):
    def __init__(self):
        super(ReconstructionBlock, self).__init__()
        padding_size = (3 - 1) // 2
        self.Model = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=padding_size, bias=True),
            DilateResBlock(256, 3, 1, 1),
            DilateResBlock(256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=padding_size, bias=True),
            DilateResBlock(128, 3, 1, 1),
            DilateResBlock(128, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=padding_size, bias=True),
            DilateResBlock(64, 3, 1, 1),
            DilateResBlock(64, 3, 1, 1),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=padding_size, bias=True),
            nn.Tanh()
        )

    def forward(self, fea_c):
        re_img_h = self.Model(fea_c)
        return re_img_h


class ASFFNet(nn.Module):
    def __init__(self):
        super(ASFFNet, self).__init__()
        self.img_d_fea_ex = ImgFeatureExtractionBlock()
        self.img_g_fea_ex = ImgFeatureExtractionBlock()
        self.bin_d_fea_ex = BinFeatureExtractionBlock()
        self.mls = MLS.MLS()
        self.adain = AdaIN.AdaIN()
        self.fea_fusion = FeatureFusionBlock()
        self.recon = ReconstructionBlock()

    def forward(self, img_d, img_g, bin_land_d, land_d, land_g):
        fea_d = self.img_d_fea_ex(img_d)
        fea_g = self.img_g_fea_ex(img_g)
        fea_l_d = self.bin_d_fea_ex(bin_land_d)
        fea_gwa = self.adain(self.mls(fea_g, land_d, land_g), fea_d)
        f_c = self.fea_fusion(fea_d, fea_gwa, fea_l_d)
        recon_img = self.recon(f_c)
        return recon_img


def tensor_to_img(tensor):
    img_out = tensor * 0.5 + 0.5
    img_out = torch.clip(img_out, 0, 1) * 255.0
    return img_out


# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
    import os
    import logging
    from DataSet import ASFFDataSet
    from util import DirectoryUtils
    from constant import *
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    import torch.optim as optim
    from tqdm import tqdm
    import Loss
    from ASFFDiscriminator import SNGANDiscriminator
    from ASFFDiscriminator import DCGANDiscriminator

    use_subset = True

    train_logger = logging.getLogger('train_logger')
    train_logger.setLevel(logging.INFO)
    file_handler1 = logging.FileHandler('train.log')
    file_handler1.setFormatter(logging.Formatter('%(message)s'))
    train_logger.addHandler(file_handler1)

    eval_logger = logging.getLogger('eval_logger')
    eval_logger.setLevel(logging.INFO)
    file_handler2 = logging.FileHandler('eval.log')
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    eval_logger.addHandler(file_handler2)

    torch.autograd.set_detect_anomaly(True)

    asffnetG = ASFFNet().to(default_device)
    asffnetD = DCGANDiscriminator().to(default_device)

    asffnetG.apply(weights_init)
    asffnetD.apply(weights_init)

    train_data_set_path = DirectoryUtils.select_file("train data list csv")
    test_data_set_path = DirectoryUtils.select_file("test data list csv")
    wls_weight_path = DirectoryUtils.select_file("wls weight path")

    train_data_list = DirectoryUtils.read_list_from_csv(train_data_set_path)
    test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
    asff_train_data = ASFFDataSet(train_data_list, wls_weight_path)
    asff_test_data = ASFFDataSet(test_data_list, wls_weight_path)

    if use_subset:
        # 훈련 데이터 중 30%만 사용
        train_size = int(0.3 * len(asff_train_data))
        val_size = len(asff_train_data) - train_size

        # 데이터 분할
        asff_train_data, _ = random_split(asff_train_data, [train_size, val_size])

        # 테스트 데이터 중 30%만 사용
        train_size = int(0.3 * len(asff_test_data))
        val_size = len(asff_test_data) - train_size

        # 데이터 분할
        asff_test_data, _ = random_split(asff_test_data, [train_size, val_size])


    train_dataloader = DataLoader(
        asff_train_data,  # 위에서 생성한 데이터 셋
        batch_size=g_batch_size,
        shuffle=True,  # 데이터들의 순서는 섞어서 분할
    )

    test_dataloader = DataLoader(
        asff_test_data,  # 위에서 생성한 데이터 셋
        batch_size=g_batch_size,
        shuffle=True,  # 데이터들의 순서는 섞어서 분할
    )

    optimizerD = optim.Adam(asffnetD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(asffnetG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def get_loss(data):
        F_r = asffnetG(data['lp_img_tensor'], data['g_img_tensor'], data['lp_land_bin_img_tensor'],
                       data['lp_landmarks_tensor'], data['g_img_landmarks_tensor'])
        I_h = tensor_to_img(F_r)
        I_truth = tensor_to_img(data['hp_img_tensor'])

        fake_validity = asffnetD(I_h.detach())
        real_validity = asffnetD(I_truth.detach())
        G_loss = Loss.ASFFGLoss(I_h, I_truth, fake_validity.detach())
        D_loss = Loss.ASFFDLoss(fake_validity, real_validity)
        return G_loss, D_loss

    epoch = 10
    for e in range(epoch):
        asffnetG.train()
        asffnetD.train()
        mem_snp_num = 1
        for data in tqdm(train_dataloader, desc='Processing Batches'):
            G_loss, D_loss = get_loss(data)

            train_logger.info("\n train loss {}".format(e) + \
                         "\nmse_loss: " + str((tradeoff_parm_mse * G_loss["mse_loss"])) + \
                         "\nperc_loss: " + str((tradeoff_parm_perc * G_loss["perc_loss"])) + \
                         "\nstyle_loss: " + str((tradeoff_parm_style * G_loss["style_loss"])) + \
                         "\nadv_loss: " + str((tradeoff_parm_adv * G_loss["adv_loss"])) + \
                         "\ntotal_loss: " + str((G_loss["total_loss"])))

            optimizerG.zero_grad()
            G_loss["total_loss"].backward()
            optimizerG.step()

            optimizerD.zero_grad()
            D_loss.backward()
            optimizerD.step()

        torch.cuda.empty_cache()

        with torch.no_grad():
            asffnetG.eval()
            asffnetD.eval()
            G_loss_sum_dict = {
                "mse_loss": 0,
                "perc_loss": 0,
                "style_loss": 0,
                "adv_loss": 0,
                "total_loss": 0
            }
            D_loss_sum = 0
            for data in tqdm(test_dataloader, desc='calculate loss'):
                G_loss, D_loss = get_loss(data)

                for key, value in G_loss_sum_dict.items():
                    G_loss_sum_dict[key] += G_loss[key].item()
                D_loss_sum += D_loss.item()

                eval_logger.info("\n eval loss {}".format(e) + \
                             "\nmse_loss: " + str((tradeoff_parm_mse * G_loss["mse_loss"])) + \
                             "\nperc_loss: " + str((tradeoff_parm_perc * G_loss["perc_loss"])) + \
                             "\nstyle_loss: " + str((tradeoff_parm_style * G_loss["style_loss"])) + \
                             "\nadv_loss: " + str((tradeoff_parm_adv * G_loss["adv_loss"])) + \
                             "\ntotal_loss: " + str((G_loss["total_loss"])))

            G_loss_avg = {
                "mse_loss": 0,
                "perc_loss": 0,
                "style_loss": 0,
                "adv_loss": 0,
                "total_loss": 0
            }
            D_loss_avg = 0

            for key, value in G_loss_sum_dict.items():
                G_loss_avg[key] = G_loss_sum_dict[key] / len(asff_test_data)
            D_loss_avg = D_loss_sum / len(asff_test_data)

            print("\nmse_loss: " + str((tradeoff_parm_mse * G_loss_avg["mse_loss"])) + \
                  "\nperc_loss: " + str((tradeoff_parm_perc * G_loss_avg["perc_loss"])) + \
                  "\nstyle_loss: " + str((tradeoff_parm_style * G_loss_avg["style_loss"])) + \
                  "\nadv_loss: " + str((tradeoff_parm_adv * G_loss_avg["adv_loss"])) + \
                  "\ntotal_loss: " + str((G_loss_avg["total_loss"])))

            # 가중치 텐서 저장
            torch.save({
                'epoch': e + 1,
                'gen_state_dict': asffnetG.state_dict(),
                'dis_state_dict': asffnetD.state_dict(),
                'gen_optimizer': optimizerG.state_dict(),
                'dis_optimizer': optimizerD.state_dict(),
                'g_loss': G_loss_avg,
                'd_loss': D_loss_avg
            }, 'asff_train_log{}.pth'.format(e + 1))

    from util.PrintTrainLog import *

    print_asff_log()
