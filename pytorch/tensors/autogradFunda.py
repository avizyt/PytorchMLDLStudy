import torch

x = torch.tensor([[1, 2, 3],[4,5,6]], dtype=torch.float, requires_grad=True)
print(x)

f = x.pow(2).sum()
print(f)
f.backward()
print(x.grad)
