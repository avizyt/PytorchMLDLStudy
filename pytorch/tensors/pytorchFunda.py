import torch

x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
print(x)

print(x[1,1])

print(x[1,1].item())

# slicing
print(x[:2,1])

# boolean indexing
print(x[x<5])

print(x.t())

print(x.view(2,4))

# combine tensors
y = torch.stack((x,x))
print(y)

a,b = x.unbind(dim=1)
print(a,b)

print(torch.squeeze(x))


print(torch.rand(2,2).max().item())

mat1 = torch.rand(5,5)

L,V = torch.eig(mat1, eigenvectors=True)

print(L)
print(V)

print(torch.inverse(mat1))
print(torch.det(mat1))

U, S, V = torch.svd(mat1)
print(U)
print(S)
print(V)

print(torch.pca_lowrank(mat1,1))

# print(torch.linalg.cholesky(mat1))

print(torch.linspace(0,1,5))