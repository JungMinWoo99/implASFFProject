import torch
import torch.nn as nn
import torch.nn.utils as utils
import constant
import model.MLS as MLS
import model.AdaIN as AdaIN

class interpolation(nn.Module):
    def __init__(self, size = None, scale_factor = None, mode = 'nearest', align_corners = False):
        super(interpolation, self).__init__()
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.__interp = nn.functional.interpolate

    def forward(self, x):
        return self.__interp(x, scale_factor = self.scale_factor, mode = self.mode, align_corners = self.align_corners)

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
            interpolation(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            #nn.PixelShuffle(2),
            #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=padding_size, bias=True),
            DilateResBlock(128, 3, 1, 1),
            DilateResBlock(128, 3, 1, 1),
            interpolation(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 32, 3, 1, 1, bias=True),
            #nn.PixelShuffle(2),
            #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=padding_size, bias=True),
            DilateResBlock(32, 3, 1, 1),
            DilateResBlock(32, 3, 1, 1),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=padding_size, bias=True),
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
        #fea_gwa = self.adain(self.mls(fea_g, land_d, land_g), fea_d)
        fea_gwa = self.mls(self.adain(fea_g, fea_d), land_d, land_g)
        f_c = self.fea_fusion(fea_d, fea_gwa, fea_l_d)
        recon_img = self.recon(f_c)
        return recon_img, {
        "lq_fea": fea_d,
        "ref_fea": fea_g,
        "bin_fea": fea_l_d,
        "warped_ref_fea": fea_gwa,
        "asff_fea": f_c
        }


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

