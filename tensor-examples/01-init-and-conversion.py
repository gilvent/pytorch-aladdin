# Initializing tensor and converting between types

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True
)

# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)


# Other common initialization methods, try printing to see the value

x = torch.empty(size = (3, 3))
x = torch.zeros((3, 3))
x = torch.rand((5, 3))
x = torch.ones((5, 3))
# Identity matrix
x = torch.eye(5, 5)
x = torch.arange(start=0, end=5, step=1)
# 1 dimensional tensor which values are evenly spaced for n space
x = torch.linspace(start=0, end=1, steps=10)
x = torch.empty(size = (1, 5)).normal_(mean = 0, std  = 1)
x = torch.empty(size = (1, 5)).uniform_(0, 1)
x = torch.diag(torch.ones(3))

# How to convert tensors to other types
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.long()) # int64, often used
print(tensor.short()) # int16
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double())

# Array conversion to numpy and vice-versa
import numpy as np

np_array = np.zeros(5)
tensor_array = torch.from_numpy(np_array)
np_array_2 = tensor_array.numpy()

print(np_array_2)
print(tensor_array)