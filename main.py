from load import ProteinDataSet, DataLoader
from models import ProteinNet
from utils import accuracy, save_checkpoint

import time
import argparse
import numpy as np
import random
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

parser = argparse.ArgumentParser(description='protein')
parser.add_argument("-e", "--epochs", help="training epochs", default=22, type=int)
parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
parser.add_argument("-b", "--batch_size", help="batch_size", default=256, type=int)
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print("xxxxxxxxxxx")


def validate(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)
            loss = criterion(output, target)
            acc = accuracy(output, target)

        print(f"acc {acc}")


def main():
    train_loader = DataLoader(ProteinDataSet("data/6125_training_64.feat"), batch_size=1, num_workers=16,
                              pin_memory=True)
    val_loader = DataLoader(ProteinDataSet("data/6125_validation_64.feat"), batch_size=1, num_workers=16,
                            pin_memory=True)

    model = ProteinNet().cuda(args.gpu)
    criterion = CrossEntropyLoss().cuda(args.gpu)
    optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        validate(val_loader, model, criterion)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })


if __name__ == '__main__':
    setup_seed(4115)
    main()
