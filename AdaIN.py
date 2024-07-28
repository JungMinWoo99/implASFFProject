import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, content, style):
        # content와 style의 차원은 [N, C, H, W] 형식이어야 함
        assert content.size()[:2] == style.size()[:2], "Content and Style should have the same batch and channel size"

        # content의 평균과 표준편차 계산
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True) + self.epsilon

        # style의 평균과 표준편차 계산
        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True) + self.epsilon

        # content의 정규화 및 style의 분포 적용
        normalized_content = (content - content_mean) / content_std
        stylized_content = normalized_content * style_std + style_mean

        return stylized_content


if __name__ == '__main__':
    # 예시 사용법
    content = torch.randn(1, 512, 64, 64)  # 예시 content tensor
    style = torch.randn(1, 512, 64, 64)  # 예시 style tensor

    adain = AdaIN()
    output = adain(content, style)

    print(output.shape)  # 출력: torch.Size([1, 512, 64, 64])
