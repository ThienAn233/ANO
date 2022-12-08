import torch
import os
import numpy as np
import seaborn as sns
import time as t
import pandas as pd
from torch import nn
import torchaudio
import librosa
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader,Dataset

#####
class Dilated(nn.Module):
    # Dilated convolution block
    def __init__(self,inp,outp,kernel,dilation,Tr=False,BN=True):
        super(Dilated,self).__init__()
        self.inp = inp
        self.outp = outp
        self.kernel =kernel
        self.BN = BN
        self.dilation = dilation
        if BN:
            bias = None
        else:
            bias = True
        if Tr :
            self.conv1 = nn.ConvTranspose1d(inp,outp,self.kernel,dilation = self.dilation,bias=bias)
        else:
            self.conv1 = nn.Conv1d(inp,outp,self.kernel,dilation = self.dilation,bias=bias)
        self.batch = nn.BatchNorm1d(outp,momentum=1)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self,inp):
        if self.BN:
            outp = self.act( self.batch( self.conv1(inp) ) )
        else:
            outp = self.conv1(inp)
        return outp
    
#####
class MNBSTD(nn.Module):

        def __init__(self,input):
            super(MNBSTD,self).__init__()
            self.input = input
        
        def forward(self,input):
            var, mean = torch.var_mean(input,dim=1)
            x = torch.cat([input,var.view(-1,1),mean.view(-1,1)],dim=1)
            return x

#####
class UTGAN(nn.Module):
    # UTGAN architecture 
    def __init__(self,encoder_dict,decoder_dict,datadis_dict,latentdis_dict,inp_siz,latent):
        super(UTGAN,self).__init__()
        self.encoder_dict = encoder_dict
        self.decoder_dict = decoder_dict
        self.datadis_dict = datadis_dict
        self.latentdis_dict = latentdis_dict
        self.inp_siz = inp_siz
        self.latent = latent
    
    def create_model(self):
        #ENCODER
        self.encoder, self.encoder_list, self.encoder_name_list = self.get_model(self.encoder_dict,"ec")
        #DECODER
        self.decoder, self.decoder_list, self.decoder_name_list = self.get_model(self.decoder_dict,"de")
        #DATADIS
        self.datadis, self.datadis_list, self.datadis_name_list = self.get_model(self.datadis_dict,'dt')
        #LATENTDIS
        self.latentdis, self.latentdis_list, self.latentdis_name_list = self.get_model(self.latentdis_dict,'lt')
      
    def gen_forward(self,input):
        return self.decoder(self.encoder(input))
    
    def gen_backward(self,input):
        return self.encoder(self.decoder(input))
    
    def dis_forward(self,sequence,latent):
        return self.datadis(sequence), self.latentdis(latent)
    
    def get_model(self, model_dict,name):
        model = nn.Sequential()
        model_list = []
        model_name_list = []
        
        for layers_name in model_dict:
            #print(layers_name)
            model_list += [model_dict[layers_name]]
            model_name_list += [layers_name]        
        for i in range(len(model_name_list)):
            model.add_module(model_name_list[i],model_list[i])
        pytorch_total_params = sum(p.numel() for p in model.parameters()) 
        print('total '+name+' params: ',pytorch_total_params)
        return model, model_list, model_name_list
    def train(self,epochs, beta, data, genoptim, disoptim, verbose=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        for epoch in range(epochs):
            for itter, (X,_) in enumerate(data):

                #train discriminators

                self.datadis.train()
                self.latentdis.train()
                disoptim.zero_grad()

                realseq = X.to(device).reshape((X.shape[0],1,self.inp_siz))
                reallat = torch.empty((realseq.shape[0],self.latent)).normal_().to(device)
                
                fakelat = self.encoder(realseq)
                fakeseq = self.decoder(reallat)

                fakesco = self.dis_forward(fakeseq,fakelat)
                realsco = self.dis_forward(realseq,reallat)

                fakecri = ((fakesco[0])**2).mean() + ((fakesco[1])**2).mean()
                realcri = ((realsco[0] - 1.)**2).mean() + ((realsco[1] - 1.)**2).mean()
                cridis = fakecri + realcri

                cridis.backward()
                disoptim.step()  

                #train generators
                self.encoder.train()
                self.decoder.train()
                genoptim.zero_grad()

                reallat = torch.empty((realseq.shape[0],self.latent)).normal_().to(device)

                fakelat = self.encoder(realseq)
                fakeseq = self.decoder(reallat)
                fakesco = self.dis_forward(fakeseq,fakelat)

                reseq = self.decoder(reallat)
                relat = self.encoder(realseq)

                rescri = nn.MSELoss()(reallat,relat) + nn.MSELoss()(realseq,reseq)
                discri = ((fakesco[0] - 1.)**2).mean() + ((fakesco[1] - 1.)**2).mean()
                crigen = rescri + beta*discri

                crigen.backward()
                genoptim.step()
            print(f'[{epoch}][{epochs}] genloss: {crigen.item()} fakegen: {discri.item()} resgen: {rescri.item()} disloss: {cridis.item()} fakedis: {fakecri.item()} realdis: {realcri.item()}')


#####
class CustomDataset(Dataset):

    def __init__(self, dir,chunk,skip,label,transform=None,val=8/7):
        self.dir = dir
        self.chunk = chunk
        self.skip = skip
        self.label = label
        self.val = val
        self.data = torchaudio.load(dir)[0][0]
        print('file: ',dir,'sampling rate: ',torchaudio.load(dir)[1])
        self.data = [self.data[i*self.skip:i*self.skip+self.chunk] for i in range((len(self.data)-self.chunk)//self.skip)]
        self.data = torch.vstack(self.data)
        self.randomidx = torch.randperm(len(self.data))
        self.datamix = self.data[self.randomidx]
        print('length of full data: ',len(self.datamix))
        self.datatrain = self.datamix[:int(len(self.datamix)//(self.val))]
        self.datavalid = self.datamix[int(len(self.datamix)//(self.val)):]
        self.transform = transform
        print('length of data',self.__len__())

    def __len__(self):
        return len(self.datatrain)

    def __getitem__(self, idx):
        output = self.datatrain[idx]
        if self.transform :
            for i in self.transform:
                output = i(output)
        return output, torch.zeros(4, dtype=torch.float).scatter_(0, torch.tensor(self.label), value=1)