import torch

# CUDA가 사용 가능한지 확인합니다.
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# CUDA 버전을 확인합니다.
cuda_version = torch.version.cuda
print(f"CUDA version: {cuda_version}")

# 사용 가능한 GPU의 수를 출력합니다.
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # 각 GPU의 이름을 출력합니다.
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("No CUDA-compatible GPU found.")
