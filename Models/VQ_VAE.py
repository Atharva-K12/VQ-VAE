import torch
import torch.nn as nn
import Models.enc_dec as enc_dec
import torch.nn.functional as F

MSE_criterion=nn.MSELoss()
class VectQuantize(nn.Module):
    def __init__(self,dim,num,beta):
        super(VectQuantize,self).__init__() 
        self.embed_dim=dim
        self.num_embed=num
        self.embedding=nn.Embedding(self.num_embed,self.embed_dim)
        self.beta=beta
        self.embedding.weight.data.uniform_(-1 / self.num_embed, 1 / self.num_embed)
    def decode(self,inputs):
        min_encode=torch.zeros(inputs.shape[0],self.num_embed,device=inputs.device)
        min_encode.scatter_(1,inputs,1)
        min_quantized_vector=torch.matmul(min_encode,self.embedding.weight).view(self.num_embed,1,1,self.embed_dim)
        return min_quantized_vector.permute(0,3,1,2)
    def forward(self,inputs):
        inputs=inputs.permute(0,2,3,1).contiguous() 
        flat_inputs=inputs.reshape(-1,self.embed_dim)
        dist=(torch.sum(flat_inputs**2,dim=1,keepdim=True)+torch.sum(self.embedding.weight**2,dim=1)-2*torch.matmul(flat_inputs,self.embedding.weight.t()))
        encode_indices=torch.argmin(dist,dim=1).unsqueeze(1)
        encode = torch.zeros(encode_indices.shape[0],self.num_embed,device=inputs.device)
        encode.scatter_(1,encode_indices,1)
        quantized_vector=torch.matmul(encode,self.embedding.weight).view(inputs.shape)
        e_loss = MSE_criterion(quantized_vector.detach(),inputs)
        ze_loss = MSE_criterion(quantized_vector,inputs.detach())
        loss=ze_loss+self.beta*e_loss
        quantized_vector=inputs+(quantized_vector-inputs).detach()
        return loss,quantized_vector.permute(0,3,1,2).contiguous() 

class VQVAE(nn.Module):
    def __init__(self,n_embeddings,nd,embed_dim,beta):
        super().__init__()
        self.encoder=enc_dec.Encoder(nd,embed_dim)
        self.decoder=enc_dec.Decoder(nd,embed_dim)
        self.n_embeddings=n_embeddings
        self.quantise=VectQuantize(embed_dim,self.n_embeddings,beta)
    def decode(self,index):
        quantised_vect=self.quantise.decode(index)
        x_random=self.decoder(quantised_vect)
        return x_random
    def forward(self,inputs):
        z=self.encoder(inputs)
        self.n_embeddings=inputs.size(0)
        loss,quantised_vect=self.quantise(z)
        quantised_vect=F.relu(quantised_vect)
        x_=self.decoder(quantised_vect)
        self.loss=loss
        return x_