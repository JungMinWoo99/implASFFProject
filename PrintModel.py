from torchsummary import summary
from model.ASFFNet import ASFFNet

model = ASFFNet().to('cuda')
summary(model, [(1, 3, 256, 256), (1, 3, 256, 256), (1, 3, 256, 256), (1, 68, 2), (1, 68, 2)])
