import torch
import torch.nn as nn
import torch.nn.functional as F




class Conv2dCustom(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1) -> None:
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_dim, out_dim, stride, padding=padding),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(out_dim),
        #     nn.MaxPool2d(2))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=padding)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_dim)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.pool(x)
        return x
        # return self.conv(x)
    