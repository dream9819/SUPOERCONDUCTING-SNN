# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:00 2021

@author: Lin Ranxi & Dai Benzhe
"""
from dataload import NoiseLoader
from feature_extraactor import Feature_Extraction
from Layer import Layer 
from nn import NeuralNetwoek
import numpy as np

DataPath='/home/dbz/Code/spsnn/RWCP-WAV'

type2int ={'cymbals': 0, 'metal15': 1, 'bells5': 2, 'bottle1': 3, 'phone4': 4, 'ring': 5, 'whistle1': 6, 'buzzer': 7, 'kara': 8, 'horn': 9}
int2type =  {0: 'cymbals', 1: 'metal15', 2: 'bells5', 3: 'bottle1', 4: 'phone4', 5: 'ring', 6: 'whistle1', 7: 'buzzer', 8: 'kara', 9: 'horn'}

dataloader = NoiseLoader(DataPath,int2type= int2type ,type2int = type2int)
fea_ext = Feature_Extraction( )
y,x_pre = dataloader.dataloader()
x = fea_ext.featureextractor(x_pre)
for x_i in x:
   x_i = x_i.reshape(x_i.shape[1],x_i.shape[0])

D = open('./spsnn/parament/momentum_bias2.txt', 'r')
bias2=np.loadtxt(D, delimiter=',')
D = open('./spsnn/parament/momentum_bias1.txt', 'r')   
bias1=np.loadtxt(D, delimiter=',')
D = open('./spsnn/parament/momentum_bias0.txt', 'r') 
bias0=np.loadtxt(D, delimiter=',')
D = open('./spsnn/parament/momentum_weights1.txt', 'r')
weight2 =np.loadtxt(D, delimiter=',')
D = open('./spsnn/parament/momentum_weight0.txt', 'r') 
weight1 =np.loadtxt(D, delimiter=',')
nn = NeuralNetwoek()
a = 1000
b = 100
nn.add_layer(Layer(a,a,'lif_relu',weights=np.eye(a),bias = bias0 ))
nn.add_layer( Layer( a,b, 'lif_relu',weights=weight1,bias = bias1 ))
nn.add_layer( Layer( b ,10, 'softmax',weights=weight2,bias =bias2))
nn.test(x,y)
