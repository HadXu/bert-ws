from load import ProteinDataSet, DataLoader
from models import ProteinNet

import argparse
import numpy as np
import random
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train():
    train_files = "data/6125_validation_64.feat"
    val_files = "data/6125_training_64.feat"

    epoches = 100
    lr = 1e-4

    tr_loader = DataLoader(ProteinDataSet(filename=train_files))
    val_loader = DataLoader(ProteinDataSet(filename=val_files))

    net = ProteinNet()
    loss_fn = CrossEntropyLoss()
    optim = Adam(net.parameters(), lr=lr)

    losses = []
    val_acc = []

    for i in range(epoches):
        print(f"{i}/{epoches}....")

        # train
        for tr_x, tr_y in tr_loader:
            pred_y = net(tr_x)
            loss = loss_fn(pred_y, tr_y)
            losses.append(loss.item())
            optim.zero_grad()

        # val
        for val_x, val_y in val_loader:
            pred_y = net(val_y)

            acc = 1
            val_acc.append(acc)

    print("train done......")


if __name__ == '__main__':
    setup_seed(4115)
    parser = argparse.ArgumentParser(description='protein')
    parser.add_argument("-e", "--epochs", help="training epochs", default=22, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=256, type=int)
    args = parser.parse_args()

    cfg = {}
    cfg['epoch'] = args.epochs
    cfg['lr'] = args.lr
    cfg['batch_size'] = args.batch_size


