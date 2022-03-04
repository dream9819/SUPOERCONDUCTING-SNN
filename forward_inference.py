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
import os
import sys

DataPath='/home/dbz/Code/spsnn/RWCP-WAV'

dataloader = NoiseLoader(DataPath)
fea_ext = Feature_Extraction( )
y,x_pre = dataloader.dataloader()
x = fea_ext.featureextractor(x_pre)
for x in x:
   x = x.reshape(x.shape[1],x.shape[0])

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
