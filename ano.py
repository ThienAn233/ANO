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


########## Bản sao của bản sao của copy of AE test
#####

class UTGAN_CustomDataset(Dataset):

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

#####
class UTGAN_SimpleDataset(Dataset):

    def __init__(self,data,label,transform=None):
        self.data = data
        self.transform = transform
        self.label = label
        print(len(self.data))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        output = self.data[idx]
        if self.transform :
            for i in self.transform:
                output = i(output)
        return output, torch.zeros(4, dtype=torch.float).scatter_(0, torch.tensor(self.label), value=1)

#####
class VAEseq_CustomImageDataset(Dataset):
    
    def __init__(self, img_dir,
                 slice,
                 k,
                 chunk,
                 transform=None,
                 target_transform=None):
        self.img_dir = img_dir
        self.slice = slice
        self.chunk = chunk
        self.transform = transform
        self.subfile = sorted([x[0] for x in os.walk(img_dir)])[k:k+1]
        print(self.subfile)
        self.lenlist = [int(len([name for name in os.listdir(file)])//slice) for file in self.subfile]
        self.lendict = {file : int(len( os.listdir(file))//slice) for file in self.subfile}
        self.subdict = {file : sorted(os.listdir(file))[:self.lendict[file]] for file in self.subfile}
    
    def __len__(self):
        return sum(self.lenlist)
    
    def get_idex(self,idx):
        idx = idx +1
        temp = [0]+self.lenlist
        temp = [temp[i]+sum(temp[:i]) for i in range(1,len(temp))]
        for file, id in enumerate(temp):
            if idx <= id:
                return self.subfile[file], self.subdict[self.subfile[file]][idx-id-1]

    def __getitem__(self, idx):  
        file, id = self.get_idex(idx)  
        path = os.path.join(file,id)
        series = torch.tensor(pd.read_csv(path,header=None).values[:,5]).float()
        if self.transform:
            series = self.transform(series)
        return torch.vstack(series.chunk(self.chunk))
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
            'drop':nn.Dropout(0.3),
            'lin1':nn.Linear(1028,self.latent*2,bias=None),
            'bat1':nn.BatchNorm1d(self.latent*2,momentum=1),
            'act1':nn.LeakyReLU(0.2),
            'lin2':nn.Linear(self.latent*2,self.latent)}
        self.decoder_dict = {'lin1':nn.Linear(self.latent,self.latent*2),
            'act1':nn.LeakyReLU(0.2),
            'lin2':nn.Linear(self.latent*2,1028,bias=None),
            'bat1':nn.BatchNorm1d(1028,momentum=1),
            'act2':nn.LeakyReLU(0.2),
            'drop':nn.Dropout(0.3),
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
    def train_model(self,epochs, beta, data, genoptim, disoptim, verbose=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        self.to(device)
        self.train()
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
                self.los['genlos'].append(crigen.item())
                self.los['dislos'].append(cridis.item())
                self.los['fakegen'].append(discri.item())
                self.los['resgen'].append(rescri.item())
                self.los['genlos'].append(fakecri.item())
                self.los['dislos'].append(realcri.item())
            print(f'[{epoch}][{epochs}] genloss: {crigen.item()} fakegen: {discri.item()} resgen: {rescri.item()} disloss: {cridis.item()} fakedis: {fakecri.item()} realdis: {realcri.item()}')
        return self.los
########## Ano_GAN.ipynb
#####
class VAEseq(nn.Module):

    def __init__(self,encode_dict=0,decode_dict=0,encode_LSTM=0,decode_LSTM=0,inp_len=6,inp_siz=512,latent=64):
        super(VAEseq,self).__init__()
        self.inp_len = inp_len
        self.latent = latent
        self.inp_siz = inp_siz
        self.encode_dict = encode_dict
        self.decode_dict = encode_dict
        self.encode_LSTM_dict = encode_LSTM
        self.decode_LSTM_dict = decode_LSTM
        self.los = {'res':[],'kl':[],'sum':[]}
    
    def create_default_model(self):
        self.inp_len = 6
        self.inp_siz = 512
        self.encode_dict = {
            'conv1':nn.Conv1d(1,self.inp_len,3),
            'act1':nn.LeakyReLU(0.2),
            'conv2':nn.Conv1d(self.inp_len,self.inp_len//2,3),
            'act2':nn.LeakyReLU(0.2),
            'flat':nn.Flatten(),
            'unfl':nn.Unflatten(-1,(1,(self.inp_len//2)*(self.inp_siz-4))),
            'lin1':nn.Linear((self.inp_len//2)*(self.inp_siz-4),self.latent),
            'act3':nn.LeakyReLU(0.2),
        }
        self.encode_LSTM_dict = {'enLTSM':nn.LSTM(self.latent,self.latent*2,1,batch_first=True)}
        self.decode_LSTM_dict = {'deLSTM':nn.LSTM(self.latent,self.latent,1,batch_first=True)}
        self.decode_dict = {
            'lin1':nn.Linear(self.latent,(self.inp_len//2)*(self.inp_siz-4)),
            'act1':nn.LeakyReLU(0.2),
            'flat':nn.Flatten(),
            'unfl':nn.Unflatten(1,((self.inp_len//2),(self.inp_siz-4))),
            'conv1':nn.ConvTranspose1d((self.inp_len//2),self.inp_len,3),
            'act2':nn.LeakyReLU(0.2),
            'conv2':nn.ConvTranspose1d(self.inp_len,1,3),
        }
        self.create_model()
    
    def create_model(self):
        self.encode, self.encode_list, self.encode_name_list,a = self.get_model(self.encode_dict,'encode')
        self.decode, self.decode_list, self.decode_name_list,b = self.get_model(self.decode_dict,'decode')
        self.encode_LSTM , self.encode_LSTM_list, self.encode_LSTM_name_list,c = self.get_model(self.encode_LSTM_dict,'encode_LSTM')
        self.decode_LSTM , self.decode_LSTM_list, self.decode_LSTM_name_list,d = self.get_model(self.decode_LSTM_dict,'decode_LSTM')
        print('2 models created')
        print('total params: ',a+b+c+d)
        
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
    
    def encoder(self,input):
        n, c, l = input.shape 
        temp = []
        for i in range(c):
            temp.append(self.encode(input[:,i,:].reshape(n,1,l)))
        x = torch.cat(temp,dim=1)
        x, hidden = self.encode_LSTM(x)
        return x[:,:,:self.latent], x[:,:,self.latent:]

    def reparameterize(self,mean,log_var):
        norm = torch.empty(mean.shape).normal_().to(mean.device)
        return mean + norm * torch.exp(log_var / 2)
    
    def decoder(self,input):
        x, hidden = self.decode_LSTM(input)
        n, c, l = x.shape 
        temp = []
        for i in range(c):
            temp.append(self.decode(x[:,i,:].reshape(n,1,l)))
        return torch.cat(temp,dim=1)

    def forward(self,input):
        mean_log_var = self.encoder(input)
        latent = self.reparameterize(*mean_log_var)
        output = self.decoder(latent)
        return mean_log_var, latent, output
    
    def KL_divergence_loss(self,z_mean,z_log_var):
        return torch.mean (-0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1) )
    
    def train_model(self,epochs,beta,data,optim,):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        self.to(device)
        self.train()
        loss = nn.MSELoss()
        self.KL_divergence_loss() 
        for epoch in range(epochs):
            for itter,X in enumerate(data):
                X = X.to(device)
                optim.zero_grad()
                mean_log_var, latent, output = self(X)
                res = loss(X,output)
                kl = self.KL_divergence_loss(*mean_log_var)
                cri = res + beta*kl
                cri.backward()
                optim.step()  
                self.los['res'].append(res.item())
                self.los['kl'].append(kl.item())
                self.los['sum'].append(cri.item())
            print(f"[{epoch}]:[{epochs}]  sum: {self.los['sum'][-1]}  res: {self.los['res'][-1]}  kl: {self.los['kl'][-1]}")  
        return self.los
