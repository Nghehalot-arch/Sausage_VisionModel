"""Check Detectron2 build and standard CUDA ROIAlign in its separate environment."""
import torch
import detectron2
from detectron2 import _C
from detectron2.layers import ROIAlign

print("Detectron2", detectron2.__version__)
print("Custom extension CUDA compiler:", _C.get_cuda_version())
assert torch.cuda.is_available()
features = torch.ones((1, 1, 8, 8), device="cuda", requires_grad=True)
boxes = torch.tensor([[0, 1, 1, 6, 6]], dtype=torch.float32, device="cuda")
pooled = ROIAlign((2, 2), 1.0, 0, aligned=True)(features, boxes)
pooled.sum().backward()
assert features.grad is not None
print("CUDA ROIAlign forward/backward passed")
