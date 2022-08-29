
from transformer import Encoder, adaptive_conv1d

import torch
import torch.nn as nn
from torch.nn import functional as F

import os, json, re, random
import pandas as pd
import numpy as np


class adaptive_transformer(nn.Module):
    
    def __init__(self):
        super(adaptive_transformer, self).__init__()

        self.conv1 = adaptive_conv1d(out_channel=16, kernel_size=3, padding="same")
        # self.position = nn.Parameter(torch.zeros(1, 32, 32)) # [1, n_channels, n_dims], graph embedded features
        self.position = nn.Parameter(torch.zeros(1, 16, 47)) # raw features
        self.dropout  = nn.Dropout(0.1)
        self.encoder  = Encoder(47, 6, 6) # n_dims, n_heads, n_layers
        self.conv2 = adaptive_conv1d(out_channel=1, kernel_size=1, padding="same")
        # self.cla = nn.Linear(in_features=32, out_features=6, bias=True) # graph embedded features
        self.cla = nn.Sequential(nn.Linear(in_features=47, out_features=16, bias=True),
                                nn.Sigmoid(),
                                nn.Linear(in_features=16, out_features=2, bias=True)
                            ) # raw features
    
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.position
        x = self.dropout(x)
        x, _ = self.encoder(x)
        x = self.conv2(x)
        x = x.flatten(1)
        pred = self.cla(x)
        return pred


class adaptive_conv(nn.Module):

    def __init__(self):
        super(adaptive_conv, self).__init__()

        self.layer1 = nn.Sequential(
            adaptive_conv1d(out_channel=16, kernel_size=3, padding="valid"),
            nn.BatchNorm1d(num_features=16),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding="valid"),
            nn.BatchNorm1d(num_features=32),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU()
        )

        self.cla = nn.Linear(in_features=640, out_features=2, bias=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(1)
        x = self.cla(x)

        return x



timeline_path = "../timeline/data/snapshot/"

if __name__ == "__main__":

    save_path = []
    for f in os.listdir(timeline_path):
        save_path.append(timeline_path + f)
    
    random.shuffle(save_path)
    L = len(save_path)
    train, evalu, test = save_path[: (7*L // 10)], save_path[(7 * L // 10): (8 * L // 10)], save_path[(8 * L // 10):]

    EPOCH_N = 60
    model = adaptive_transformer()
    # model = adaptive_conv()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    save_path = "model/win_lose.pkl"

    for epoch in range(EPOCH_N):
        model.train()

        record_loss = []
        for fname in train:
            optimizer.zero_grad()
            with open(fname) as fin:
                data = json.load(fin)
            winning = data[0]
            if winning == 100:
                y = 0
            else:
                y = 1
            y = torch.Tensor([y]).long()
            for idx, snap in enumerate(data[1]):
                x = np.row_stack(snap)
                x = torch.Tensor([x]).float()

                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                record_loss.append(loss.item())

            optimizer.step()

        model.eval()
        total = 0
        correct = 0
        for fname in evalu:
            with open(fname) as fin:
                data = json.load(fin)
            wanning = data[0]
            if winning == 100:
                y = 0
            else:
                y = 1
            y = torch.Tensor([y]).long()
            for snap in data[1]:
                x = np.row_stack(snap)
                x = torch.Tensor([x]).float()

                pred = model(x)
                _, pred = torch.max(pred.data, 1)
                total += 1
                correct += (y == pred).sum().item()
        record_loss = sum(record_loss) / len(record_loss)
        print("In epoch %d, loss: %f, eval acc: %d in %d" % (epoch, record_loss, correct, total))

        temp_acc = correct / total
        if temp_acc >= best_acc:
            torch.save(model, save_path)
            print("Save a better model.")
            best_acc = temp_acc

    # testing
    model = torch.load(save_path)
    model.eval()

    total   = 0
    correct = 0
    for fname in test:
        with open(fname) as fin:
            data = json.load(fin)
        winning = data[0]
        if winning == 100:
            y = 0
        else:
            y = 1
        y = torch.Tensor([y]).long()
        for snap in data[1]:
            x = np.row_stack(snap)
            x = torch.Tensor([x]).float()

            pred = model(x)
            _, pred = torch.max(pred.data, 1)
            total += 1
            correct += (y == pred).sum().item()
    print("Test accuracy: %d in %d" % (correct, total))






