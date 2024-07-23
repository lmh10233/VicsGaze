import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math


class Gaze_regressor(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=2, drop=0.1):
        super(Gaze_regressor, self).__init__()

        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU(inplace=True)
        # self.apply(self._init_weights)
        self.loss_op = nn.L1Loss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        # print(gaze[0])
        loss = self.loss_op(gaze, label)
        return loss
        


if __name__ == '__main__':
    model = Gaze_regressor()
    print(model)
