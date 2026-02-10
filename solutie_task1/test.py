import torch
import torch_directml

print(f"DirectML available: {torch_directml.is_available()}")
device = torch_directml.device()
print(f"Device: {device}")

# Test tensor
x = torch.randn(10, 10).to(device)
print(f"Tensor on: {x.device}")  # privateuseone:0 = DirectML GPU