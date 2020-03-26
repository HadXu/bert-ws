import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {0: 0, 1: 1, 2: 2, START_TAG: 3, STOP_TAG: 4}


class ProteinDataSet(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.feats, self.labels = self.load()

    def __getitem__(self, x):
        feat = self.feats[x]
        label = self.labels[x]

        feat = np.stack(feat)  # (L, 64)
        label = np.array(label, dtype=np.int64)  # (L, )

        # print(feat.shape)
        # print(label.shape)

        return feat, label

    def __len__(self):
        return len(self.feats)

    def load(self):
        print("load feat.....")
        feats = []
        labels = []
        with open(self.filename, 'r') as f:
            line = f.readline()
            """
            读取数据,按行操作
            """
            while line:
                lenght = int(line)

                feat = []
                label = []

                for _ in range(lenght):
                    line = f.readline()
                    feat.append([float(x) for x in line.split()])

                for _ in range(lenght):
                    line = f.readline()
                    label.append(int(line))

                line = f.readline()

                feats.append(feat)
                labels.append(label)
        assert len(feats) == len(labels)
        return feats, labels


if __name__ == '__main__':
    filename = "data/6125_validation_64.feat"
    loader = DataLoader(ProteinDataSet(filename), batch_size=1)
    for x, y in loader:
        print(x.squeeze(0).size())
        print(y.squeeze(0).size())
