import torch.nn as nn

"""SNGANDiscriminator"""


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self,in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad))

        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class SNGANDiscriminator(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(SNGANDiscriminator, self).__init__()
        self.activation = activation
        self.block1 = OptimizedDisBlock(3, 64)
        self.block2 = DisBlock(64, 128, activation=activation, downsample=True)
        self.block3 = DisBlock(128, 256, activation=activation, downsample=True)
        self.block4 = DisBlock(256, 512, activation=activation, downsample=True)
        self.block5 = DisBlock(512, 1024, activation=activation, downsample=False)

        self.l6 = nn.utils.spectral_norm(nn.Linear(1024, 1, bias=False))

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l6(h)

        return output

'''DCGANDiscriminator'''


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        ndf = 16
        nc = 3
        self.main = nn.Sequential(
            # input is ``(nc) x 256 x 256``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 128 x 128``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 64 x 64``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 32 x 32``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 16 x 16``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*16) x 8 x 8``
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*32) x 4 x 4``
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, l):
        return self.main(l)

if __name__ == '__main123__':
    from DataSet import TruthImgDataSet
    from util import DirectoryUtils
    from constant import *
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from tqdm import tqdm
    import Loss

    asffnetD = DCGANDiscriminator().to(default_device)

    train_data_set_path = DirectoryUtils.select_file("train data list csv")
    test_data_set_path = DirectoryUtils.select_file("test data list csv")

    train_data_list = DirectoryUtils.read_list_from_csv(train_data_set_path)
    test_data_list = DirectoryUtils.read_list_from_csv(test_data_set_path)
    d_train_data = TruthImgDataSet(train_data_list)
    d_test_data = TruthImgDataSet(test_data_list)

    optimizerD = optim.Adam(asffnetD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    train_dataloader = DataLoader(
        d_train_data,  # 위에서 생성한 데이터 셋
        batch_size=g_batch_size,
        shuffle=True,  # 데이터들의 순서는 섞어서 분할
    )

    test_dataloader = DataLoader(
        d_test_data,  # 위에서 생성한 데이터 셋
        batch_size=g_batch_size,
        shuffle=True,  # 데이터들의 순서는 섞어서 분할
    )

    loss_list = []
    epoch = 10
    for e in range(epoch):
        asffnetD.train()
        for data in tqdm(train_dataloader, desc='Processing Batches'):
            d_output = asffnetD(data[0])
            loss = Loss.ASFFadvDLoss(d_output,)

            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()

        asffnetD.eval()
        loss_sum = 0
        for data in tqdm(test_dataloader, desc='calculate loss'):
            I_h = asffnetG(data['lp_img_tensor'], data['g_img_tensor'], data['lp_land_bin_img_tensor'],
                           data['lp_landmarks_tensor'], data['g_img_landmarks_tensor'])
            loss = Loss.ASFFGLoss(I_h, data['hp_img_tensor'])
        loss_avg = loss_sum / len(asff_test_data)
        loss_list.append(loss_avg)
        # 가중치 텐서 저장
        torch.save(asffnetG.state_dict(), 'asff_weight_tensor{}.pth'.format(e))

    import matplotlib.pyplot as plt

    # 꺾은선 그래프 그리기
    plt.plot([i for i in range(1, epoch + 1)], loss_list, marker='o', linestyle='-')

    # 각 데이터 포인트에 값 표시
    for i, value in enumerate(loss_list):
        plt.plot([i for i in range(1, epoch + 1)], loss_list, marker='o', linestyle='-')
        plt.annotate(f'{value:.5f}', (i, loss_list[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    # 그래프 제목과 축 레이블
    plt.title('loss avg')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # 그래프 저장
    plt.savefig('line_plot_with_values.png')

    # 그래프 보여주기
    plt.show()
