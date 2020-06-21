from __future__ import print_function
import torch

x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

y = torch.rand([2,5])
print(y)
y = y.view([1,10])
print(y)
