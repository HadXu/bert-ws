from torch import nn
import torch


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class ProteinNet(nn.Module):
    def __init__(self):
        super(ProteinNet, self).__init__()
        self.cnf = CNF()

    def forward(self, input):
        out = self.cnf(input)
        return out


class CNF(nn.Module):
    def __init__(self):
        super(CNF, self).__init__()

    def forward(self, input):
        return input


def test_net():
    net = ProteinNet()
    x = torch.rand((10, 5, 64))
    y = net(x)

    print(y.size())


if __name__ == '__main__':
    test_net()
