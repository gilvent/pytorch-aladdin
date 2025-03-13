import torch

tensor = torch.arange(9)

# Both are used to reshape but there is underlying difference, read more
tensor_3x3 = tensor.view(9, 1)
tensor_3x3_2 = tensor.reshape(3, 3)
print(tensor_3x3_2)

# Transpose the matrix
tensor1 = tensor_3x3_2.t()
print(tensor1)

# Contiguous view
print(tensor1.contiguous().view(9))

tensor2 = torch.rand((2, 5))
tensor3 = torch.rand((2, 5))
print(torch.cat((tensor2, tensor3), dim = 0))
print(torch.cat((tensor2, tensor3), dim = 1))

# Unrolling tensors
tensor4 = tensor2.view(-1)
print(tensor4.shape)

tensor5 = torch.rand(64, 2, 5)
tensor6 = tensor5.view(64, -1) # keep the first dimension, and unroll the 2x5 tensor
print(tensor6.shape)

# Permute
# Move dimension 2 to the dimension 1 position to produce tensor(64, 5, 2) from tensor(64, 2, 5)
tensor7 = tensor5.permute(0, 2, 1) 
print(tensor7.shape)

# Adding or removing dimensions
tensor8 = torch.arange(10)
print(tensor8.shape)
print(tensor8.unsqueeze(0).shape)
print(tensor8.unsqueeze(1))

tensor9 = tensor8.unsqueeze(1).unsqueeze(0)
print(tensor9.shape)
print(tensor9.squeeze(2).shape)