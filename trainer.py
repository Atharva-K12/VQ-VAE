import torch
import random
import utils
import Models.VQ_VAE as VQ_VAE
import loader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt


manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
batch_size = 128
image_size = 64
n_embeddings = 512
embed_dim = 64
ne = 128
nd = 128
num_epochs = 10 
lr = 0.001
beta = 0.25

random_generation_interval = 250


dataloader=loader.train_loader_fn(batch_size)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

rgb2hsv=utils.RgbToHsv()
hsv2rgb=utils.HsvToRgb()

plot=utils.Plotter()
logger=utils.Logger(['Recon_loss','VQ_loss','Total_loss'])


model=VQ_VAE.VQVAE(n_embeddings,nd,embed_dim,beta).to(device)
MSE_criterion=nn.SmoothL1Loss()
BCE_criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

try:
    for epoch in tqdm(range(num_epochs)):
        i=0
        for data in tqdm(dataloader):
            optimizer.zero_grad()
            x = rgb2hsv(data[0]).to(device)
            if x.size(0)==batch_size:
                output = model(x)
                recon_loss= MSE_criterion(output,x)
                vq_loss=model.loss
                loss=recon_loss+vq_loss
                loss.backward()
                optimizer.step()
                log_dict={
                    'Recon_loss':recon_loss.item(),
                    'VQ_loss':vq_loss.item(),
                    'Total_loss':loss.item()}
                logger.lossLog(log_dict)
                if i%random_generation_interval==0:
                    with torch.no_grad():
                        random_num=torch.randint(n_embeddings,(n_embeddings,1)).to(device)
                        random_gen=model.decode(random_num).detach().cpu()
                    plot.im_plot(random_gen,"Random_gen")
                    model.eval()
                    valid_data=loader.train_loader_fn(batch_size)
                    validation,_=next(iter(valid_data))
                    validation=rgb2hsv(validation).to(device)
                    valid_recon=model(validation)
                    plot.im_plot(validation,"Validation")
                    plot.im_plot(valid_recon,"Reconstruction")
                i+=1
        print('epoch :[%d/%d]\tTotal_loss: %.8f\tVQ_loss: %.8f\tRecon_loss: %.8f\n'% (epoch, num_epochs,loss.item(), vq_loss.item(),recon_loss.item()))
except KeyboardInterrupt:
    print('epoch :[%d/%d]\tTotal_loss: %.8f\tVQ_loss: %.8f\tRecon_loss: %.8f\n'% (epoch, num_epochs,loss.item(), vq_loss.item(),recon_loss.item()))


plot.loss_plotter()

model.eval()
valid_data=loader.train_loader_fn(batch_size)
validation,_=next(iter(valid_data))
validation=rgb2hsv(validation).to(device)
valid_recon=model(validation)
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Validation Images")
plt.imshow(np.transpose(vutils.make_grid(hsv2rgb(validation.to(device)[:16]), padding=2, normalize=True).detach().cpu(),(1,2,0)))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Validation Reconstruction Images")
plt.imshow(np.transpose(vutils.make_grid(hsv2rgb(valid_recon.to(device)[:16]), padding=2, normalize=True).detach().cpu(),(1,2,0)))
plt.savefig('Test_outputs/Reconstruction')
