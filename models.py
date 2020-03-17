from torch import nn
import torch


class ProteinNet(nn.Module):
    def __init__(self):
        super(ProteinNet, self).__init__()

    def forward(self, input):
        return input


if __name__ == '__main__':
    net = ProteinNet()
    x = torch.rand((10, 5, 64))

    y = net(x)

    print(y.size())
