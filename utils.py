import math
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.utils  import save_image,make_grid
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
#####  RGB<---->HSV ######
def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]
    v: torch.Tensor = maxc
    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / (v + eps)
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)
    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]
    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)
    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac
    h = (h / 6.0) % 1.0
    h = 2 * math.pi * h
    return torch.stack([h, s, v], dim=-3)
def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))
    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]
    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0).to(image.device)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)
    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    return out
class RgbToHsv(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(RgbToHsv, self).__init__()
        self.eps = eps
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_hsv(image, self.eps)
class HsvToRgb(nn.Module):
    def __init__(self) -> None:
        super(HsvToRgb, self).__init__()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return hsv_to_rgb(image)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Logger():
    def __init__(self,loss_list):
        try:
            os.mkdir('Test_outputs')
        except FileExistsError:
            pass
        self.loss_list = loss_list
        with open('Test_outputs\loss_log.csv','w') as csv_file:
            csv_writer = csv.DictWriter(csv_file,fieldnames=loss_list)
            csv_writer.writeheader()
    def lossLog(self,loss_list):
        with open('Test_outputs\loss_log.csv','a') as csv_file:
            csv_writer = csv.DictWriter(csv_file,fieldnames=self.loss_list)
            csv_writer.writerow(loss_list)


class Plotter():
    def __init__(self):
        try:
            os.mkdir('Test_outputs')
        except FileExistsError:
            pass
        self.count=0   
    def loss_plotter(self):
        data =pd.read_csv('Test_outputs/loss_log.csv')
        plt.figure(figsize=(10,5))
        plt.title("Loss")
        for losses in list(data.columns.values.tolist()):
            plt.plot(data[losses],label=str(losses))
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()  
        plt.savefig('Test_outputs/Loss')
    def loss_live_plotter(self):
        def _animation(i):
            data=pd.read_csv('Test_outputs/loss_log.csv')
            plt.cla()
            plt.title("Loss")
            for losses in list(data.columns.values.tolist()):
                plt.plot(data[losses],label=str(losses))
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            #plt.tight_layout()
        ani=FuncAnimation(plt.gcf(),_animation)
        plt.show()
    def im_live_plotter(self):
        def _animation(i):
            image=plt.imread('Test_outputs/Reconstruction.png')
            plt.cla()
            plt.imshow(image) 
            plt.axis('off')
        ani=FuncAnimation(plt.gcf(),_animation)
        plt.show()
    def im_plot(self,image,s='0'):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=128),
            transforms.ToTensor()
            ])
        image = hsv_to_rgb(image.to(device)[:16])
        image = [transform(x_) for x_ in image]
        save_image(make_grid(image, padding=2, normalize=True),'Test_outputs/'+s+'.png')
