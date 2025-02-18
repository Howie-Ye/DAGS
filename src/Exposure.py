from src import (os, sys, math, random, time,
    torch, nn, torchvision, F, o3d)
from src.KF_Camera import KF_Camera

class Exposure:
    def __init__(self, kf_cam:KF_Camera):


        self._scale = nn.Parameter(torch.ones(len(kf_cam)).cuda().requires_grad_(True))
        self._shift = nn.Parameter(torch.zeros(len(kf_cam)).cuda().requires_grad_(True))
        l = [
            {'params': self._scale, 'lr': 0.001, 'name': 'scale'},
            {'params': self._shift, 'lr': 0.001, 'name': 'shift'}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.001, eps=1e-15)

    def get_scale_shift(self,idx):
        return self._scale[idx], self._shift[idx]