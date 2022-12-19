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

#####
class UTGAN_CustomDataset(Dataset):

    def __init__(self, dir,chunk,skip,label,transform=None):
        self.dir = dir
        self.chunk = chunk
        self.skip = skip
        self.label = label
        self.transform = transform
        self.subfile = sorted([x[0] for x in os.walk(self.dir)][1:])
        self.lenlist = [len([name for name in os.listdir(file)]) for file in self.subfile]
        self.lendict = {file : len( os.listdir(file)) for file in self.subfile}
        self.subdict = {file : sorted(os.listdir(file))[:self.lendict[file]] for file in self.subfile}
        self.data = []
        for folder, files in self.subdict.items():
            for file in self.subdict[folder]: 
                data = torchaudio.load(os.path.join(folder,file))[0][0]
                print('file: ',file,'len',len(data),'sampling rate: ',torchaudio.load(os.path.join(folder,file))[1])
                data = [data[i*self.skip:i*self.skip+self.chunk] for i in range((len(data)-self.chunk)//self.skip)]
                data = torch.vstack(data)
                self.data.append(data)
        self.data = torch.vstack(self.data)
        print('length of data',self.__len__())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output = self.data[idx]
        if self.transform :
            for i in self.transform:
                output = i(output)
        return output, self.label

#####
class VAEseq_CustomImageDataset(Dataset):
    
    def __init__(self, img_dir,
                 slice,
                 k,
                 chunk,
                 transform=None):
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
        if self.transform :
            for i in self.transform:
                series = i(series)
        return torch.vstack(series.chunk(self.chunk))