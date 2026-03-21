import torch
print("--- CUDA Setup Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")   
    # Run a quick calculation on the GPU
    x = torch.tensor([1.0, 2.0]).cuda()
    print(f"Test Tensor on GPU: {x}")
else:
    print("CUDA is not available. Check your installation.")