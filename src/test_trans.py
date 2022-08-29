# try this new conv layer on role identification problem

# from turtle import forward
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

        self.conv1 = adaptive_conv1d(out_channel=32, kernel_size=3, padding="same")
        self.position = nn.Parameter(torch.zeros(1, 32, 32)) # 1, n_channels, n_dims
        self.dropout  = nn.Dropout(0.1)
        self.encoder  = Encoder(32, 6, 6) # n_channel, n_heads, n_layers
        self.conv2 = adaptive_conv1d(out_channel=1, kernel_size=1, padding="same")
    
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.position
        x = self.dropout(x)
        x, weights = self.encoder(x)
        x = self.conv2(x)
        return x, weights


# read champion roles and transfer into labels
# embs = np.load("all_champ_abilities.npy")
embs = np.load("embedding/ability_emb_hope_32d.npy")
with open("new_roles.json") as fin:
    roles = json.load(fin)
all_roles = set()
root      = "champ/en/"
end_line  = 0

X = []
Y = []

for f in os.listdir(root):
    champ_name = f[:-4]
    champ_file = pd.read_csv(root + f)
    nrows = len(champ_file.index)

    x = embs[end_line: end_line + nrows]
    y = roles[champ_name][0]
    end_line += nrows

    X.append(x)
    Y.append(y)
    all_roles.add(y)

print(embs.shape, end_line)

all_roles = list(all_roles)
Y = [all_roles.index(y) for y in Y]

data = list(zip(X, Y))
print(len(data))

"""
model = adaptive_conv1d(out_channel=32, kernel_size=3, padding="same")
data  = torch.randn((1, 5, 32))
outt = model(data)
print(outt.shape)
"""
model = adaptive_transformer()# .cuda()

x = data[0][0]
x = torch.Tensor([x])# .cuda()
print(x.shape)
out, _ = model(x)
print(out.shape)


