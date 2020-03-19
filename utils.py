import torch


def accuracy(pred, target):
    return 1.


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


