import torch

# .pth 파일 경로
file_path = 'weight_tensor.pth'

# .pth 파일 읽기
checkpoint = torch.load(file_path)

print(checkpoint)
