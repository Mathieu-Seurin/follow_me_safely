import torch

# GPU compatibility setup
use_cuda = torch.cuda.is_available()
TORCH_DEVICE = torch.device("cuda" if use_cuda else "cpu")
