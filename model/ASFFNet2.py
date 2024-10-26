import torch
import torch.nn as nn
import torch.nn.utils as utils
import constant
import model.MLS as MLS
import model.AdaIN as AdaIN


class ASFFNet(nn.Module):
    def __init__(self):
        super(ASFFNet, self).__init__()
        self.img_d_fea_ex = ImgFeatureExtractionBlock()
        self.img_g_fea_ex = ImgFeatureExtractionBlock()
        self.bin_d_fea_ex = BinFeatureExtractionBlock()
        self.mls = MLS.MLS()
        self.adain = AdaIN.AdaIN()
        self.gwa_cond_fea_ex = ComponentConditionBlock()
        self.fea_fusion = FeatureFusionBlock()
        self.ms_fea_ex = MSDilateBlock()
        self.recon = ReconstructionBlock()

    def forward(self, img_d, img_g, bin_land_d, land_d, land_g):
        fea_d = self.img_d_fea_ex(img_d)
        fea_g = self.img_g_fea_ex(img_g)
        fea_l_d = self.bin_d_fea_ex(bin_land_d)
        # fea_gwa = self.adain(self.mls(fea_g, land_d, land_g), fea_d)
        fea_gwa = self.mls(self.adain(fea_g, fea_d), land_d, land_g)
        fea_gwa_cond = self.gwa_cond_fea_ex(fea_gwa)
        fea_c = self.fea_fusion(fea_d, fea_gwa_cond, fea_l_d)
        fea_c_ms = self.ms_fea_ex(fea_c)
        recon_img = self.recon(fea_c_ms + fea_d)
        return recon_img, {
            "lq_fea": fea_d,
            "ref_fea": fea_g,
            "bin_fea": fea_l_d,
            "warped_ref_fea": fea_gwa_cond,
            "asff_fea": fea_c_ms + fea_d
        }


class ImgFeatureExtractionBlock(nn.Module):
    def __init__(self):
        super(ImgFeatureExtractionBlock, self).__init__()
        self.Model = nn.Sequential(
            SNConv_BN_LReLU(3, 64, 3, 1, True),
            DilateResBlock(64, 3, 1, [7, 5]),
            DilateResBlock(64, 3, 1, [7, 5]),
            SNConv_BN_LReLU(64, 128, 3, 2, True),
            DilateResBlock(128, 3, 1, [5, 3]),
            DilateResBlock(128, 3, 1, [5, 3]),
            SNConv_BN_LReLU(128, 256, 3, 2, True),
            DilateResBlock(256, 3, 1, [3, 1]),
            DilateResBlock(256, 3, 1, [3, 1]),
            SNConv_LReLU(256, 128, 3, 1, True)
        )

    def forward(self, img):
        img_fea = self.Model(img)
        return img_fea


class DilateResBlock(nn.Module):
    def __init__(self, input_ch, kernel, stride, dilation):
        super(DilateResBlock, self).__init__()
        padding_size = [((kernel - 1) // 2) * d for d in dilation]
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(input_ch, input_ch, kernel_size=kernel, stride=stride, dilation=dilation[0],
                                          padding=padding_size[0])),
            nn.BatchNorm2d(input_ch, eps=constant.eps),
            nn.LeakyReLU(constant.LReLu_negative_slope),
            utils.spectral_norm(nn.Conv2d(input_ch, input_ch, kernel_size=kernel, stride=stride, dilation=dilation[1],
                                          padding=padding_size[1]))
        )

    def forward(self, x):
        out = x + self.Model(x)
        return out


class BinFeatureExtractionBlock(nn.Module):
    def __init__(self):
        super(BinFeatureExtractionBlock, self).__init__()
        self.Model = nn.Sequential(
            Conv_LReLU(1, 16, 9, 2, False),
            Conv_LReLU(16, 32, 7, 1, False),
            Conv_LReLU(32, 64, 5, 2, False),
            Conv_LReLU(64, 128, 3, 1, False),
            Conv_LReLU(128, 128, 3, 1, False),
            nn.Sigmoid()
        )

    def forward(self, l_d):
        return self.Model(l_d)


class ComponentConditionBlock(nn.Module):  # gwa_fea의 조건적 특성을 추출
    def __init__(self):
        super(ComponentConditionBlock, self).__init__()
        self.Model = nn.Sequential(
            SNConv_LReLU(128, 128, 1, 1, True),
            UpResBlock(128),
            SNConv_LReLU(128, 128, 1, 1, True),
            UpResBlock(128),
            SNConv_LReLU(128, 128, 1, 1, True),
        )

    def forward(self, gwa_fea):
        gwa_up_fea = self.Model(gwa_fea)
        return gwa_up_fea


class UpResBlock(nn.Module):  # 업샘플링을 위한 잔차 블록 정의
    def __init__(self, c):
        super(UpResBlock, self).__init__()
        self.Model = nn.Sequential(
            SNConv_LReLU(c, c, 3, 1, True),
            SNConv_LReLU(c, c, 3, 1, True),
        )

    def forward(self, x):
        out = x + self.Model(x)
        return out  # 입력과 모델 결과를 더하여 출력


class FeatureFusionBlock(nn.Module):
    def __init__(self):
        super(FeatureFusionBlock, self).__init__()
        self.ASFFBlock1 = ASFFBlock()
        self.ASFFBlock2 = ASFFBlock()
        self.ASFFBlock3 = ASFFBlock()
        self.ASFFBlock4 = ASFFBlock()
        self.SFFBlock = SFFLayer()

    def forward(self, fea_d, fea_gwa, fea_l):
        fea_d_new1 = self.ASFFBlock1(fea_d, fea_gwa, fea_l)
        fea_d_new2 = self.ASFFBlock2(fea_d_new1, fea_gwa, fea_l)
        fea_d_new3 = self.ASFFBlock3(fea_d_new2, fea_gwa, fea_l)
        fea_d_new4 = self.ASFFBlock4(fea_d_new3, fea_gwa, fea_l)
        fea_c = self.SFFBlock(fea_d_new4, fea_gwa, fea_l)
        return fea_c


class ASFFBlock(nn.Module):
    def __init__(self):
        super(ASFFBlock, self).__init__()
        self.sff0 = SFFLayer()
        self.conv0 = nn.Sequential(
            SNConv_LReLU(128, 128, 5, 1, True),
            SNConv_LReLU(128, 128, 3, 1, True),
        )
        self.sff1 = SFFLayer()
        self.conv1 = nn.Sequential(
            SNConv_LReLU(128, 128, 5, 1, True),
            SNConv_LReLU(128, 128, 3, 1, True),
        )

    def forward(self, fea_d, fea_gwa, fea_l):
        fea1_1 = self.sff0(fea_d, fea_gwa, fea_l)
        fea1_2 = self.conv0(fea1_1)
        fea2_1 = self.sff1(fea1_2, fea_gwa, fea_l)
        fea2_2 = self.conv1(fea2_1)
        return fea_d + fea2_2


class SFFLayer(nn.Module):
    def __init__(self):
        super(SFFLayer, self).__init__()
        self.MaskModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(192, 128, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(128, eps=constant.eps),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
        )
        self.MaskConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(128, 64, 1, 1)),
        )
        self.DegradedModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
            nn.BatchNorm2d(128, eps=constant.eps),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
        )
        self.DegradedConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(128, 64, 1, 1)),
        )
        self.RefModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
            nn.BatchNorm2d(128, eps=constant.eps),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
        )
        self.RefConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(128, 64, 1, 1)),
        )

    def forward(self, fea_d, fea_gwa, fea_l):
        F_d_fea_d = self.DegradedModel(fea_d)
        F_g_fea_gwa = self.RefModel(fea_gwa)

        fea_l_concat = self.MaskConcat(fea_l)
        F_d_fea_d_concat = self.DegradedConcat(F_d_fea_d)
        F_g_fea_gwa_concat = self.RefConcat(F_g_fea_gwa)

        fea_m = self.MaskModel(torch.cat((fea_l_concat, F_d_fea_d_concat, F_g_fea_gwa_concat), 1))
        fea_d_mask = F_d_fea_d + (F_g_fea_gwa - F_d_fea_d) * fea_m
        return fea_d_mask  # 마스크를 사용하여 복원 특징 융합


class MSDilateBlock(nn.Module):  #
    def __init__(self):
        super(MSDilateBlock, self).__init__()
        self.conv1 = SNDilateConv_LReLU(128, 64, 3, 1, 4, True)
        self.conv2 = SNDilateConv_LReLU(128, 64, 3, 1, 3, True)
        self.conv3 = SNDilateConv_LReLU(128, 64, 3, 1, 2, True)
        self.conv4 = SNDilateConv_LReLU(128, 64, 3, 1, 1, True)
        self.convi = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(256, 128, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
        )

    def forward(self, fea_c):
        fea_ms1 = self.conv1(fea_c)
        fea_ms2 = self.conv2(fea_c)
        fea_ms3 = self.conv3(fea_c)
        fea_ms4 = self.conv4(fea_c)
        fea_ms_cat = torch.cat([fea_ms1, fea_ms2, fea_ms3, fea_ms4], 1)
        fea_ms = self.convi(fea_ms_cat) + fea_c
        return fea_ms


class ReconstructionBlock(nn.Module):
    def __init__(self):
        super(ReconstructionBlock, self).__init__()
        self.Model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SNConv_BN_LReLU(128, 128, 3, 1, True),
            UpDilateResBlock(128, [1, 2]),
            SNConv_BN_LReLU(128, 64, 3, 1, True),
            UpDilateResBlock(64, [1, 2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            UpDilateResBlock(64, [1, 1]),
            utils.spectral_norm(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.Tanh()
        )

    def forward(self, fea_c):
        re_img_h = self.Model(fea_c)
        return re_img_h


class UpDilateResBlock(nn.Module):
    def __init__(self, c, dilation):
        super(UpDilateResBlock, self).__init__()
        self.Model0 = nn.Sequential(
            SNDilateConv_LReLU(c, c, 3, 1, dilation[0], True),
            SNDilateConv_LReLU(c, c, 3, 1, dilation[0], True),
        )
        self.Model1 = nn.Sequential(
            SNDilateConv_LReLU(c, c, 3, 1, dilation[0], True),
            SNDilateConv_LReLU(c, c, 3, 1, dilation[0], True),
        )

    def forward(self, x):
        out = x + self.Model0(x)
        out2 = out + self.Model1(out)
        return out2


class SNConv_BN_LReLU(nn.Module):
    def __init__(self, input_ch, output_ch, kernel, stride, bias):
        super(SNConv_BN_LReLU, self).__init__()
        padding_size = ((kernel - 1) // 2)
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=stride,
                                          padding=padding_size, bias=bias)),
            nn.BatchNorm2d(output_ch, eps=constant.eps),
            nn.LeakyReLU(constant.LReLu_negative_slope)
        )

    def forward(self, x):
        return self.Model(x)


class SNDilateConv_LReLU(nn.Module):
    def __init__(self, input_ch, output_ch, kernel, stride, dilation, bias):
        super(SNDilateConv_LReLU, self).__init__()
        padding_size = ((kernel - 1) // 2) * dilation
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=stride, dilation=dilation,
                                          padding=padding_size, bias=bias)),
            nn.LeakyReLU(constant.LReLu_negative_slope)
        )

    def forward(self, x):
        return self.Model(x)


class SNConv_LReLU(nn.Module):
    def __init__(self, input_ch, output_ch, kernel, stride, bias):
        super(SNConv_LReLU, self).__init__()
        padding_size = ((kernel - 1) // 2)
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=stride,
                                          padding=padding_size, bias=bias)),
            nn.LeakyReLU(constant.LReLu_negative_slope)
        )

    def forward(self, x):
        return self.Model(x)


class Conv_LReLU(nn.Module):
    def __init__(self, input_ch, output_ch, kernel, stride, bias):
        super(Conv_LReLU, self).__init__()
        padding_size = ((kernel - 1) // 2)
        self.Model = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=stride,
                      padding=padding_size, bias=bias),
            nn.LeakyReLU(constant.LReLu_negative_slope)
        )

    def forward(self, x):
        return self.Model(x)


def tensor_to_img(tensor):
    img_out = tensor * 0.5 + 0.5
    img_out = torch.clip(img_out, 0, 1) * 255.0
    return img_out


def tensor_denormalization(tensor):
    d_tensor = tensor * 0.5 + 0.5
    ret = torch.clip(d_tensor, 0, 1)
    return ret


# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
