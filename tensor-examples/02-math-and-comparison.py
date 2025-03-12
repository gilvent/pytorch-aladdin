import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])
print("x =", x)
print("y =", y)

# Addition (3 ways)
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x, y)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)

# inplace operations (mutate the object instead of creating new object)
t = torch.tensor([[2, 2, 2], [3, 4, 5]])
t.add_(x)
t += x  # also do inplace operation
# ! NOTE: doing t = t + x somehow creates a copy of t, so it's not doing inplace operation


# Exponentation (element-wise)
z = x.pow(2)
z = x**2

# Simple comparison (Element-wise comparison)
z = x > 0
z = x < 0

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))

x3 = torch.mm(x1, x2)  # results in 2x3 matrix
x3 = x1.mm(x2)

# Matrix exponentation
mtx = torch.rand((5, 5))
matrix_exp = mtx.matrix_power(3)

# Element-wise multiplication
z = x * y

# dot product
z = x.dot(y)
print(z)

# Batch matrix multiplication
batch = 32
m = 2
n = 8
p = 4

mtx1 = torch.rand((32, m, n))
mtx2 = torch.rand((32, n, p))
bmm = mtx1.bmm(mtx2)
print(bmm)