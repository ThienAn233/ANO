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

#* DATA nên được lưu trữ như thế này:
#   |__Learning_data
#   |   |__Dam_ngay_dd_mm_yy
#   |   |   |__file1.csv
#   |   |   |__file2.csv     
#   |   |   |__...
#   |   |__Dam_ngay_dd_mm_yy
#   |   |   |__file1.csv
#   |   |   |__file2.csv     
#   |   |   |__...
#   |   |__...
#   |__...

########## Bản sao của bản sao của copy of AE test
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
    def __init__(self,encoder_dict=0,decoder_dict=0,datadis_dict=0,latentdis_dict=0,inp_siz=1024,latent=128):
        super(UTGAN,self).__init__()
        self.encoder_dict = encoder_dict
        self.decoder_dict = decoder_dict
        self.datadis_dict = datadis_dict
        self.latentdis_dict = latentdis_dict
        self.inp_siz = inp_siz
        self.latent = latent
        self.los = {'genlos':[],'dislos':[],'fakegen':[],'resgen':[],'fakedis':[],'realdis':[]}
    
    def create_default_model(self):
        self.inp_siz = 1024
        self.latent = 128
        e = 32
        k = 16
        u = 2 
        v=128
        self.encoder_dict = {'dil1':Dilated(1,e,2,2,BN=False),
            'dil2':Dilated(e,e,2,4),
            'dil3':Dilated(e,k,2,8),
            'dil4':Dilated(k,k,2,16),
            'dil5':Dilated(k,u,2,32),
            'dil6':Dilated(u,u,2,64),
            'dil7':Dilated(u,u,2,128),
            'dil8':Dilated(u,u,2,256),
            'flat':nn.Flatten(),
            #'drop':nn.Dropout(0.3),
            'lin1':nn.Linear(1028,self.latent*2,bias=None),
            'bat1':nn.BatchNorm1d(self.latent*2,momentum=1),
            'act1':nn.LeakyReLU(0.2),
            'lin2':nn.Linear(self.latent*2,self.latent)}
        self.decoder_dict = {'lin1':nn.Linear(self.latent,self.latent*2),
            'act1':nn.LeakyReLU(0.2),
            'lin2':nn.Linear(self.latent*2,1028,bias=None),
            'bat1':nn.BatchNorm1d(1028,momentum=1),
            'act2':nn.LeakyReLU(0.2),
            #'drop':nn.Dropout(0.3),
            'unfl':nn.Unflatten(-1,(u,1028//u)),
            'dil1':Dilated(u,u,2,256,Tr=True,BN=False),
            'dil2':Dilated(u,u,2,128,Tr=True),
            'dil3':Dilated(u,u,2,64,Tr=True),
            'dil4':Dilated(u,k,2,32,Tr=True),
            'dil5':Dilated(k,k,2,16,Tr=True),
            'dil6':Dilated(k,e,2,8,Tr=True),
            'dil7':Dilated(e,e,2,4,Tr=True),
            'dil8':Dilated(e,1,2,2,Tr=True,BN=False)}
        self.datadis_dict = {'dil1':Dilated(1,e,2,2,BN=False),
            'act1':nn.LeakyReLU(0.2),
            'dil2':Dilated(e,e,2,4,BN=False),
            'act2':nn.LeakyReLU(0.2),
            'dil3':Dilated(e,k,2,8,BN=False),
            'act3':nn.LeakyReLU(0.2),
            'dil4':Dilated(k,k,2,16,BN=False),
            'act4':nn.LeakyReLU(0.2),
            'dil5':Dilated(k,u,2,32,BN=False),
            'act5':nn.LeakyReLU(0.2),
            'dil6':Dilated(u,u,2,64,BN=False),
            'act6':nn.LeakyReLU(0.2),
            'dil7':Dilated(u,u,2,128,BN=False),
            'act7':nn.LeakyReLU(0.2),
            'dil8':Dilated(u,u,2,256,BN=False),
            'act8':nn.LeakyReLU(0.2),
            'flat':nn.Flatten(),
            'lin1':nn.Linear(1028,self.latent*2),
            'act9':nn.LeakyReLU(0.2),
            'drop':nn.Dropout(0.3),
            'lin2':nn.Linear(self.latent*2,self.latent),
            'mnbs':MNBSTD(self.latent),
            'lin3':nn.Linear(self.latent+2,1)}
        
        self.latentdis_dict = {'lin1':nn.Linear(self.latent,v*4),
            'act1':nn.LeakyReLU(0.2),
            'lin2':nn.Linear(v*4,v*3),
            'act2':nn.LeakyReLU(0.2),
            'drop':nn.Dropout(0.3),
            'lin3':nn.Linear(v*3,v*2),
            'act3':nn.LeakyReLU(0.2),
            'drop':nn.Dropout(0.3),
            'lin4':nn.Linear(v*2,self.latent),
            'mnbs':MNBSTD(self.latent),
            'lin5':nn.Linear(self.latent+2,1)}
            
        self.create_model()
    
    def create_model(self):
        #ENCODER
        self.encoder, self.encoder_list, self.encoder_name_list,a = self.get_model(self.encoder_dict,"ec")
        #DECODER
        self.decoder, self.decoder_list, self.decoder_name_list,b = self.get_model(self.decoder_dict,"de")
        #DATADIS
        self.datadis, self.datadis_list, self.datadis_name_list,c = self.get_model(self.datadis_dict,'dt')
        #LATENTDIS
        self.latentdis, self.latentdis_list, self.latentdis_name_list,d = self.get_model(self.latentdis_dict,'lt')
        
        print('4 models created')
        print('total params: ',a+b+c+d)
      
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
        return model, model_list, model_name_list, pytorch_total_params
    def train_model(self, epochs, loop, beta, data, genoptim, disoptim,logger=None, verbose=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        relo = nn.MSELoss()
        self.to(device)
        self.train()
        for epoch in range(epochs):
            for itter, (X,_) in enumerate(data):

                #train discriminators
                for i in range(loop):
                    self.datadis.train()
                    self.latentdis.train()
                    disoptim.zero_grad()

                    realseq = X.to(device).reshape((X.shape[0],1,self.inp_siz))
                    reallat = torch.empty((realseq.shape[0],self.latent)).normal_().to(device)

                    fakelat = self.encoder(realseq)
                    fakeseq = self.decoder(reallat)

                    fakesco = self.dis_forward(fakeseq,fakelat)
                    realsco = self.dis_forward(realseq,reallat)

                    fakecri = .5*(((fakesco[0])**2).mean() + ((fakesco[1])**2).mean())
                    realcri = .5*(((realsco[0] - 1.)**2).mean() + ((realsco[1] - 1.)**2).mean())
                    cridis = fakecri + realcri

                    cridis.backward()
                    disoptim.step()  

                #train generators
                self.encoder.train()
                self.decoder.train()
                genoptim.zero_grad()

                fakelat = torch.empty((realseq.shape[0],self.latent)).normal_().to(device)

                reallat = self.encoder(realseq)
                fakeseq = self.decoder(reallat)
                fakesco = self.dis_forward(fakeseq,fakelat)

                reseq = self.decoder(reallat)
                relat = self.encoder(realseq)

                rescri = relo(reallat,relat) + relo(realseq,reseq)
                discri = .5*(((fakesco[0] - 1.)**2).mean() + ((fakesco[1] - 1.)**2).mean())
                crigen = rescri + beta*discri

                crigen.backward()
                genoptim.step()
                self.los['genlos'].append(crigen.item())
                self.los['dislos'].append(cridis.item())
                self.los['fakegen'].append(discri.item())
                self.los['resgen'].append(rescri.item())
                self.los['fakedis'].append(fakecri.item())
                self.los['realdis'].append(realcri.item())
                
                if logger:
                    logger.log({'genlos':crigen.item(),'dislos':cridis.item(),'fakegen':discri.item(),'resgen':rescri.item(),'fakedis':fakecri.item(),'realdis':realcri.item()})
            if logger:
                self.eval()
                y_values = self.gen_forward(X[0].reshape(1,1,self.inp_siz).to(device)).detach().cpu().squeeze()
                x_values = [i for i in range(len(y_values))]
                y_true = X[0]
                data_f = [[x, y] for (x, y) in zip(x_values, y_values)]
                data_true = [[x, y] for (x, y) in zip(x_values, y_true)]
                table = logger.Table(data=data_f, columns = ["x", "y"])
                table_true = logger.Table(data=data_true, columns = ["x", "y"])
                logger.log({"my_output_plot_id" : logger.plot.line(table, "x", "y",
                           title="Custom Y vs X Line Plot")})
                logger.log({"my_gtruth_plot_id" : logger.plot.line(table_true, "x", "y",
                           title="Custom Y vs X Line Plot")})
                self.train()
            print(f'[{epoch}][{epochs}] genloss: {crigen.item()} fakegen: {discri.item()} resgen: {rescri.item()} disloss: {cridis.item()} fakedis: {fakecri.item()} realdis: {realcri.item()}')
        return self.los
