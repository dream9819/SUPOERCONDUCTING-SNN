import numpy as np
D = open('./spsnn/parament/momentum_bias2.txt', 'r')
bias2=np.loadtxt(D, delimiter=',')
print(bias2)