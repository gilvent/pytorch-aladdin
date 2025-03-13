import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print("x = ", x)

print("first batch:\n", x[0])
print("all features in last batch:\n", x[9, :])
print("first feature of all batches:\n", x[:, 0])
print("shape of the first batch:\n", x[0].shape)
print("first - 10th feature on third batch:\n", x[2, 0:10])

# Assigning value to tensor
x[0, 0] = 0

# Fancy indexing
tensor2 = torch.arange(10)
indices = [2, 5, 8]
print(tensor2[indices])

tensor3 = torch.rand(3, 5)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(tensor3[rows, cols]) # pick up tensor[1, 4] and tensor [0, 0]

# Conditional indexing
tensor4 = torch.arange(10)
print(tensor4[(tensor4 < 2) | (tensor4 > 8)])
print(tensor4[tensor4.remainder(2) == 0])
print(torch.where(tensor4 > 5, tensor4, tensor4 * 2)) # do operations on elements based on condition
print(torch.tensor([0, 0, 1, 1, 5, 6, 6]).unique())
print(tensor4.ndimension()) # check how many dimension of tensor
print(tensor4.numel()) # number of elements