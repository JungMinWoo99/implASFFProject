import torch
import torch.nn as nn
import torch.nn.utils as utils
import constant
import model.MLS as MLS
import model.AdaIN as AdaIN


class ASFFNet(nn.Module):
    def __init__(self):
        super(ASFFNet, self).__init__()
        model = []
        SFF_Num = 4  # SFF 블록의 개수 설정
        for i in range(SFF_Num):
            model.append(ASFFBlock(128))
        model.append(FeatureFusion(128))
        self.sff_branch = nn.Sequential(*model)
        self.MSDilate = MSDilateBlock(128, dilation=[4, 3, 2, 1])

        self.mls = MLS.MLS()
        self.adain = AdaIN.AdaIN()

        # UpModel 정의
        self.UpModel = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            UpDilateResBlock(128, [1, 2]),
            utils.spectral_norm(nn.Conv2d(128, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),
            UpDilateResBlock(64, [1, 2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            UpDilateResBlock(64, [1, 1]),
            utils.spectral_norm(nn.Conv2d(64, 3, 3, 1, 1)),
            nn.Tanh()
        )

        self.LQ_Model = FeatureExDilatedResNet(64)
        self.Ref_Model = FeatureExDilatedResNet(64)
        self.Mask_CModel = MaskConditionModel(128)
        self.Component_CModel = ComponentConditionModel(128)

    def forward(self, LQ, Ref, Mask, lq_landmark, ref_landmark):
        LQ_Feature = self.LQ_Model(LQ)
        Ref_Feature = self.Ref_Model(Ref)

        DownScale = LQ.size(2) / LQ_Feature.size(2)
        lq_landmark_D = torch.transpose(lq_landmark / DownScale, 1, 2)
        ref_landmark_D = torch.transpose(ref_landmark / DownScale, 1, 2)

        MLS_Ref_Feature = self.mls(adaptive_instance_normalization_4D(Ref_Feature, LQ_Feature), lq_landmark_D,
                                   ref_landmark_D)
        MaskC = self.Mask_CModel(Mask)
        ComponentC = self.Component_CModel(MLS_Ref_Feature)

        Fea = self.sff_branch((LQ_Feature, ComponentC, MaskC))
        MSFea = self.MSDilate(Fea)

        out = self.UpModel(MSFea + LQ_Feature)

        return out, {
            "lq_fea": LQ_Feature,
            "ref_fea": Ref_Feature,
            "bin_fea": MaskC,
            "warped_ref_fea": MLS_Ref_Feature,
            "asff_fea": MSFea + LQ_Feature
        }

    def get_upmodel_outputs(self, x):
        """UpModel의 각 레이어 출력을 저장하는 함수."""
        feature_maps = []  # 중간 출력 저장

        for layer in self.UpModel:
            x = layer(x)
            feature_maps.append(x.clone().detach().cpu())  # 출력 저장

        return x, feature_maps


class FeatureExDilatedResNet(nn.Module):
    def __init__(self, ngf=64):
        super(FeatureExDilatedResNet, self).__init__()
        self.conv1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(3, ngf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )
        self.stem1_1 = resnet_block(ngf, dilation=[7, 5])
        self.stem1_2 = resnet_block(ngf, dilation=[7, 5])

        self.conv2 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(ngf, ngf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        self.stem2_1 = resnet_block(ngf * 2, dilation=[5, 3])
        self.stem2_2 = resnet_block(ngf * 2, dilation=[5, 3])

        self.conv3 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2),
        )

        self.stem3_1 = resnet_block(ngf * 4, dilation=[3, 1])
        self.stem3_2 = resnet_block(ngf * 4, dilation=[3, 1])

        self.conv4 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, img):  # 입력 이미지에 대한 특징 추출
        feature1 = self.stem1_2(self.stem1_1(self.conv1(img)))
        #visualize_feature_map(feature1,"Feature Extraction layer conv1 Visualizing")  # 특징 맵 시각화                                      #성능 측정시 주석 요망
        feature2 = self.stem2_2(self.stem2_1(self.conv2(feature1)))
        #visualize_feature_map(feature2,"Feature Extraction layer conv2 Visualizing")  # 특징 맵 시각화                                      #성능 측정시 주석 요망
        feature3 = self.stem3_2(self.stem3_1(self.conv3(feature2)))
        #visualize_feature_map(feature3,"Feature Extraction layer conv3 Visualizing")  # 특징 맵 시각화                                      #성능 측정시 주석 요망
        feature4 = self.conv4(feature3)
        #visualize_feature_map(feature4,"Feature Extraction layer conv4 Visualizing")  # 특징 맵 시각화                                      #성능 측정시 주석 요망
        return feature4


class MaskConditionModel(nn.Module):
    def __init__(self, out_channels):
        super(MaskConditionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 9, 2, 4, 1, bias=False),  # 입력 채널을 16으로 변환
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 7, 1, 3, 1, bias=False),  # 채널을 32로 변환
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 5, 2, 2, 1, bias=False),  # 채널을 64로 변환
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, out_channels, 3, 1, 1, 1, bias=False),  # 출력 채널 설정
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, 1, bias=False),  # 최종 출력 채널
            nn.Sigmoid()  # 활성화 함수 적용
        )

    def forward(self, x):
        return self.model(x)  # 마스크 조건 모델을 통해 특징 추출


def MSconvU(in_channels, out_channels, conv_layer, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        utils.spectral_norm(
            conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                       padding=((kernel_size - 1) // 2) * dilation, bias=bias)),
        nn.LeakyReLU(0.2)  # 비선형 활성화 함수 적용
    )


class MSDilateBlock(nn.Module):  # 다중 팽창 합성곱 블록 정의
    def __init__(self, in_channels, conv_layer=nn.Conv2d, kernel_size=3, dilation=[7, 5, 3, 1], bias=True):
        super(MSDilateBlock, self).__init__()
        self.conv1 = MSconvU(in_channels, in_channels // 2, conv_layer, kernel_size, dilation=dilation[0], bias=bias)
        self.conv2 = MSconvU(in_channels, in_channels // 2, conv_layer, kernel_size, dilation=dilation[1], bias=bias)
        self.conv3 = MSconvU(in_channels, in_channels // 2, conv_layer, kernel_size, dilation=dilation[2], bias=bias)
        self.conv4 = MSconvU(in_channels, in_channels // 2, conv_layer, kernel_size, dilation=dilation[3], bias=bias)
        self.convi = nn.Sequential(
            utils.spectral_norm(conv_layer(in_channels * 2, in_channels * 1, kernel_size=kernel_size, stride=1,
                                           padding=(kernel_size - 1) // 2, bias=bias)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(conv_layer(in_channels * 1, in_channels, kernel_size=kernel_size, stride=1,
                                           padding=(kernel_size - 1) // 2, bias=bias)),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out  # 입력과 팽창 합성곱 결과를 합하여 출력


class ComponentConditionModel(nn.Module):  # 컴포넌트 조건 모델 정의
    def __init__(self, in_channel):
        super(ComponentConditionModel, self).__init__()
        self.model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1)),
            nn.LeakyReLU(0.2),
            UpResBlock(in_channel),
            utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1)),
            nn.LeakyReLU(0.2),
            UpResBlock(in_channel),
            utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)  # 컴포넌트 조건 모델을 통해 특징 추출


class SFFLayer(nn.Module):  # SFF 레이어 정의
    def __init__(self, in_channels):
        super(SFFLayer, self).__init__()
        self.MaskModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels // 2 * 3, in_channels, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
        )
        self.MaskConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1, 1)),
        )
        self.DegradedModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.DegradedConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1, 1)),
        )
        self.RefModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.RefConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1, 1)),
        )

    def forward(self, X):
        # X[0] LQ feature, X[1]: Ref Feature, X[2]: MaskFeature
        MaskC = X[2]
        DegradedF = self.DegradedModel(X[0])
        RefF = self.RefModel(X[1])

        DownMask = self.MaskConcat(MaskC)
        DownDegraded = self.DegradedConcat(DegradedF)
        DownRef = self.RefConcat(RefF)

        ConcatMask = torch.cat((DownMask, DownDegraded, DownRef), 1)
        MaskF = self.MaskModel(ConcatMask)

        return DegradedF + (RefF - DegradedF) * MaskF  # 마스크를 사용하여 복원 특징 융합


class FeatureFusion(nn.Module):  # 마지막 SFF 레이어 정의
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.MaskModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels // 2 * 3, in_channels, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)),
        )
        self.MaskConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1, 1)),
        )
        self.DegradedModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.DegradedConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1, 1)),
        )
        self.RefModel = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
        )
        self.RefConcat = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1, 1)),
        )

    def forward(self, X):
        # X[0] LQ feature, X[1]: Ref Feature, X[2]: MaskFeature
        MaskC = X[2]
        DegradedF = self.DegradedModel(X[0])
        RefF = self.RefModel(X[1])

        DownMask = self.MaskConcat(MaskC)
        DownDegraded = self.DegradedConcat(DegradedF)
        DownRef = self.RefConcat(RefF)

        ConcatMask = torch.cat((DownMask, DownDegraded, DownRef), 1)
        MaskF = self.MaskModel(ConcatMask)
        fused_feature = DegradedF + (RefF - DegradedF) * MaskF
        return fused_feature  # 마지막 SFF 레이어에서 복원 특징 융합


class ASFFBlock(nn.Module):  # ASFF 블록 정의
    def __init__(self, in_channels=128):
        super(ASFFBlock, self).__init__()
        self.sff0 = SFFLayer(in_channels)
        self.conv0 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 5, 1, 2)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.sff1 = SFFLayer(in_channels)
        self.conv1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 5, 1, 2)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, X):
        fea1_1 = self.sff0(X)
        fea1_2 = self.conv0(fea1_1)
        fea2_1 = self.sff1((fea1_2, X[1], X[2]))
        fea2_2 = self.conv1(fea2_1)
        return (X[0] + fea2_2, X[1], X[2])  # SFF와 합성곱을 통과한 특징 결합


def adaptive_instance_normalization_4D(content_feat, style_feat):  # Adaptive Instance Normalization (AdaIN) 정의
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_4D(style_feat)
    content_mean, content_std = calc_mean_std_4D(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    fused_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return fused_feat  # 스타일 특징을 콘텐츠 특징에 적용


def calc_mean_std_4D(feat, eps=1e-5):  # 특징 맵의 평균과 표준편차 계산
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std  # 평균과 표준편차 반환


def resnet_block(in_channels, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1, 1],
                 bias=True):
    return ResnetBlock(in_channels, conv_layer, norm_layer, kernel_size, dilation, bias=bias)  # ResNet 블록 정의


class ResnetBlock(nn.Module):  # ResNet 블록 정의
    def __init__(self, in_channels, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1, 1],
                 bias=True):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            utils.spectral_norm(
                conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0],
                           padding=((kernel_size - 1) // 2) * dilation[0], bias=bias)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(
                conv_layer(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1],
                           padding=((kernel_size - 1) // 2) * dilation[1], bias=bias)),
        )

    def forward(self, x):
        out = self.stem(x) + x
        return out  # 입력과 블록 결과를 더하여 출력


class UpResBlock(nn.Module):  # 업샘플링을 위한 잔차 블록 정의
    def __init__(self, dim):
        super(UpResBlock, self).__init__()
        self.Model = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, 1)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = x + self.Model(x)
        return out  # 입력과 모델 결과를 더하여 출력


class UpDilateResBlock(nn.Module):  # 업샘플링을 위한 팽창 잔차 블록 정의
    def __init__(self, dim, dilation=[2, 1]):
        super(UpDilateResBlock, self).__init__()
        self.Model0 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, ((3 - 1) // 2) * dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, ((3 - 1) // 2) * dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
        )
        self.Model1 = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, ((3 - 1) // 2) * dilation[1], dilation[1])),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, ((3 - 1) // 2) * dilation[1], dilation[1])),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = x + self.Model0(x)
        out2 = out + self.Model1(out)
        return out2  # 팽창 잔차 블록을 통해 특징 추출 및 결합


if __name__ == '__main__':
    print('Test')  # 테스트 실행


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


def adaptive_instance_normalization_4D(content_feat, style_feat):  #
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_4D(style_feat)
    content_mean, content_std = calc_mean_std_4D(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
