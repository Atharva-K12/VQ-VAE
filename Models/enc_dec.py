import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,input_channels,channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels,channel,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel,input_channels,1),
        )
    def forward(self,input):
        out=self.conv(input)
        out+=input
        out=F.relu(out)
        return out
class InvResBlock(nn.Module):
    def __init__(self,input_channels,channel):
        super(InvResBlock,self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(input_channels,channel,3,padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel,input_channels,1),
        )
    def forward(self, input):
        out=self.deconv(input)
        out+=input
        out=F.relu(out)
        return out
class Encoder(nn.Module):
    def __init__(self,nd,embed_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, nd, 4, 2, 1,bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nd, nd * 2, 4, 2,1,bias=False),
            nn.BatchNorm2d(nd * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nd * 2, nd * 4, 4, 2,1,bias=False),
            nn.BatchNorm2d(nd * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nd * 4, nd * 8, 4, 2,1,bias=False),
            nn.BatchNorm2d(nd * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd * 8, embed_dim, 4, 2,1,bias=False),
            nn.ReLU(True),
            ResBlock(embed_dim,nd*8),
            ResBlock(embed_dim,nd*8),
            ResBlock(embed_dim,nd*8),
        )
    def forward(self,x):
        return self.main(x)
class Decoder(nn.Module):
    def __init__(self,nd,embed_dim):
        super().__init__()
        self.main = nn.Sequential(
            ResBlock(embed_dim,nd*8),
            nn.ConvTranspose2d(embed_dim, nd * 8, 4, 2, 1,bias=False),
            nn.BatchNorm2d(nd * 8),
            nn.ReLU(True),
            ResBlock(nd*8,nd*4),
            nn.ConvTranspose2d( nd * 8, nd * 4, 2, 2,bias=False),
            nn.BatchNorm2d(nd * 4),
            nn.ReLU(True),
            ResBlock(nd*4,nd*2),
            nn.ConvTranspose2d( nd *4, nd * 2,4, 2, 1,bias=False),
            nn.BatchNorm2d(nd * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( nd * 2, nd, 4, 2,1,bias=False),
            nn.BatchNorm2d(nd),
            nn.ReLU(True),
            nn.ConvTranspose2d( nd, 3, 4, 2, 1,bias=False),
            nn.ReLU(True),
        )
    def forward(self, input):
        return self.main(input)