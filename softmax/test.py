import torch


a = torch.randn((3,3), requires_grad=True)
b = torch.randn((3,3), requires_grad=True)
c = torch.matmul(a,b)
c.sum().backward()
print(a.grad, b.grad)
a.grad.zero_()
b.grad.zero_()
x = torch.matmul(a, b)
x.sum().backward()
print(a.grad, b.grad)
