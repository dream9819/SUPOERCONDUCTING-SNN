from dataload import BalanceDataLoad_
from feature_extraactor import Feature_Extraction
from Layer import Layer 
from nn import NeuralNetwoek
import numpy as np

DataPath='/home/dbz/Code/spsnn/RWCP-WAV'

type2int ={'cymbals': 0, 'metal15': 1, 'bells5': 2, 'bottle1': 3, 'phone4': 4, 'ring': 5, 'whistle1': 6, 'buzzer': 7, 'kara': 8, 'horn': 9}
int2type =  {0: 'cymbals', 1: 'metal15', 2: 'bells5', 3: 'bottle1', 4: 'phone4', 5: 'ring', 6: 'whistle1', 7: 'buzzer', 8: 'kara', 9: 'horn'}

dataloader = BalanceDataLoad_(DataPath)
fea_ext = Feature_Extraction( )
y_train, y_test,x_pre_train,x_pre_test = dataloader.dataloader()
x_test = fea_ext.featureextractor(x_pre_test)
x_train = fea_ext.featureextractor(x_pre_train)
for x in x_test:
   x = x.reshape(x.shape[1],x.shape[0])
for x in x_train:
    x=x.reshape(x.shape[1],x.shape[0])

nn = NeuralNetwoek()
a = 1000
b = 75
nn.add_layer(Layer(a,a,'lif_relu',weights=np.eye(a)))
nn.add_layer( Layer( a,b, 'lif_relu'))
nn.add_layer( Layer( b ,10, 'softmax'))
learning_rate = 0.008
max_epochs = 360
nn.train(x_train,x_test,y_train,y_test,learning_rate,max_epochs)