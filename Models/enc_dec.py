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
        self.main =nn.Sequential(
            nn.Conv2d(3,nd,4,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(nd,nd*2,4,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(nd*2,nd*2,3,1,1),
            nn.LeakyReLU(),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            nn.LeakyReLU(),
            nn.Conv2d(nd*2,embed_dim,1,1),
            nn.LeakyReLU(),
        )
    def forward(self,x):
        return self.main(x)
class Decoder(nn.Module):
    def __init__(self,nd,embed_dim):
        super().__init__()
        self.main =nn.Sequential(
            nn.Conv2d(embed_dim,nd*2,3,1,1),
            nn.LeakyReLU(),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            ResBlock(nd*2,nd*2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nd*2,nd,4,2,1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(nd,3,4,2,1),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)