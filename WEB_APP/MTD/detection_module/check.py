import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available.")
    print("CUDA version:", torch.version.cuda)  # 显示 CUDA 版本
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("Name of the current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")
    if torch.version.cuda is not None:
        print("CUDA version in PyTorch:", torch.version.cuda)
    else:
        print("No CUDA version information available.")

# 添加更多调试信息
print("\nAdditional Debug Information:")
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA availability:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
