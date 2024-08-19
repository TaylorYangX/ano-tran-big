import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import csv




class SWaTegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Define the start and end rows
        #start_row = 200000
        #end_row = 300000
        #nrows = 2000

        # Read the specified rows
        #df = pd.read_csv('your_file.csv', skiprows=start_row, nrows=nrows, header=None)

        #data = pd.read_csv(data_path + '/train.csv',sep=';',encoding="utf-8",engine='python',quoting=csv.QUOTE_NONE,error_bad_lines=False)
        #data = pd.read_csv(data_path + '/train.csv',engine='python',on_bad_lines='skip')
        
        #print("select ")
        #data = pd.read_csv(data_path + '/train.csv',engine='python',on_bad_lines='skip',nrows=nrows)
        data = pd.read_csv(data_path + '/train.csv',engine='python',on_bad_lines='skip')
        

             
        data = data.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        for i in list(data):
            data[i] = data[i].apply(lambda x: str(x).replace(",","."))
        data = data.astype(float)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        #test_data = pd.read_csv(data_path + '/test.csv',sep=';',encoding="utf-8",quoting=csv.QUOTE_NONE)
        #test_data = pd.read_csv(data_path + '/test.csv',sep=';',engine='python',on_bad_lines='skip')

        #test_data = pd.read_csv(data_path + '/test.csv',sep=';',engine='python',on_bad_lines='skip',nrows=nrows)
        test_data = pd.read_csv(data_path + '/test.csv',sep=';',engine='python',on_bad_lines='skip')
             
                 
        labels = [ float(label!= 'Normal' ) for label  in test_data["Normal/Attack"].values]
        test_data = test_data.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        for i in list(test_data):
            test_data[i]=test_data[i].apply(lambda x: str(x).replace("," , "."))
        test_data = test_data.astype(float)

        ##测试集不能fit
        ##self.scaler.fit(test_data)
        test_data = self.scaler.transform(test_data)

        self.test = test_data
        self.train = data
        self.val = self.test
        self.test_labels = labels

        print("test:", self.test.shape)
        print("train:", self.train.shape)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=100, step=100):
    
    dataset = SWaTegLoader(data_path, win_size, 1 )

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader