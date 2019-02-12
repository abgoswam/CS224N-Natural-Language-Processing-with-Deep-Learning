import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.x1 = None
        self.x2 = None

    def forward(self, x):
        self.x1 = F.relu(self.fc1(x))
        self.x2 = self.fc2(self.x1)
        return self.x2


net = Net()
print(net)

input = torch.randn(1, 16 * 5 * 5)
out = net(input)

# ########################################################################
# # Zero the gradient buffers of all parameters and backprops with random
# # gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

print(net.fc1.weight.grad)
print(net.fc2.weight.grad)
print(net.x1.grad, net.x1.requires_grad)  # grad empty, but requires_grad=True. why ?
print(net.x2.grad, net.x1.requires_grad)  # grad empty, but requires_grad=True. why ?