import numpy as np
import sys
import time

class NeuralNetwoek:
    def __init__(self) :
        self._layers = [ ]
   
    def add_layer(self ,  layer):
        self._layers.append(layer)

    def feed_forward(self , x ):
        for layer in self._layers:
            x = layer.activate(x)
        return x
    #   定义经典adam优化器函数
    def Adam(self, Vector, m, v,t):
        beta1 = 0.9
        t += 1
        beta2 = 0.999
        epislon = 1e-8
        m = beta1 * m + (1 - beta1) * Vector
        v = beta2 * v + (1 - beta2) * np.multiply(Vector, Vector)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        delta = m_hat/(v_hat**0.5 + epislon)
        return delta, m, v
    #   定义动量法函数
    def Momentum(self, Vector, momentum):
        beta = 0.9
        momentum = beta * momentum + (1 - beta) * Vector
        return momentum, momentum

    def backpropagation( self ,  x ,  y , learning_rate, T, l00, l01, l02, epoch):
        output = self.feed_forward(x)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = np.array(y) - output[0]
                g = layer.apply_activation_derivative(output[0])
                layer.delta = np.dot(g, layer.error)
            else :
                next_layer = self._layers[i+1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                g = layer.apply_activation_derivative(layer.last_activation_frequency)
                layer.delta = layer.error * g

        for i in range(len(self._layers)):
            layer = self._layers[i]
            if i == 0:
                #layer.adam_delta, layer.m, layer.v = self.Adam(layer.delta, layer.m, layer.v,epoch) #   使用adam优化器更新
                layer.momentum_delta, layer.momentum_bias = self.Momentum(layer.delta, layer.momentum_bias)#    动量法

                layer.bias += layer.momentum_delta * learning_rate
                if epoch == 359:
                    with open('momentum_bias0.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                        np.savetxt(D, layer.bias, delimiter=',')
            else:
                o_j = np.atleast_2d(x if i == 0 else self._layers[i - 1].last_activation_number)
                o_i = np.atleast_2d(x if i == 0 else self._layers[i].last_activation_number)

                #   动量法更新
                layer.momentum_delta_bias, layer.momentum_bias = self.Momentum(np.multiply(layer.delta, o_i),
                                                                               layer.momentum_bias)
                layer.momentum_delta_weight, layer.momentum_weight = self.Momentum(layer.delta * o_j.T,
                                                                                  layer.momentum_weight)

                #   Adam优化器更新
                #layer.adam_delta_bias, layer.m_b, layer.v_b = self.Adam(np.multiply(layer.delta, o_i), layer.m_b,layer.v_b, epoch)
                #layer.adam_delta_weight, layer.m_w, layer.v_w = self.Adam(layer.delta * o_j.T, layer.m_w, layer.v_w,epoch)

                #   权重与偏置更新
                layer.weights += layer.momentum_delta_weight * learning_rate * 1 # 权重更新
                layer.bias += (layer.momentum_delta_bias * learning_rate).reshape(layer.delta.shape[0])

                for onebias in layer.bias:
                    if onebias > 3.9:
                        onebias = 3.9
                if epoch == 359:
                    if i == 2:
                        with open('momentum_bias2.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                            np.savetxt(D, layer.bias, delimiter=',')
                    elif i == 1:
                        with open('momentum_bias1.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                            np.savetxt(D, layer.bias, delimiter=',')
                    if i == 2:
                        with open('momentum_weights1.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                            np.savetxt(D, layer.weights, delimiter=',')
                    elif i == 1:
                        with open('momentum_weight0.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                            np.savetxt(D, layer.weights, delimiter=',')
        """
        for i in range(len(self._layers)):
            layer = self._layers[i]
            if i == 2:
                with open('Bias2.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                    np.savetxt(D, layer.bias, delimiter=',')
            if i == 1:
                with open('Bias1.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                    np.savetxt(D, layer.bias, delimiter=',')
            if i == 0:
                with open('Bias0.txt', 'w') as D:  # 打开test.txt   如果文件不存在，创建该文件。
                    np.savetxt(D, layer.bias, delimiter=',')
        """
        """
        if T == 1.5:
            l0 = self._layers[0].last_activation_number
            l1 = self._layers[1].last_activation_number
            l2 = self._layers[2].last_activation_number
            l00 = np.append(l00, l0)
            l01 = np.append(l01, l1)
            l02 = np.append(l02, l2)
        """
        if T == 1.5:

            for i in range(len(self._layers)):
                layer = self._layers[i]
                if i == 0:
                    layer.previous_factor = 1
                # TODO;fix this

                elif i == 2:
                    pass
                    #previous_layer = self._layers[i - 1]
                    layer.previous_factor, layer.current_factor = self.previousfactor(l02, l01)
                    layer.weights *= layer.previous_factor
                    layer.bias /= layer.current_factor

                else:
                    #previous_layer = self._layers[i - 1]
                    layer.previous_factor, layer.current_factor = self.previousfactor(l01,
                                                                l00)
                    layer.weights *= layer.previous_factor
                    layer.bias /= layer.current_factor

    # 执行权重以及偏置正则化
    def previousfactor(self, current_vector, previous_vector, max_p = 99):
        current_factor = np.percentile(current_vector, max_p)
        if current_factor <= 0: #此处改为<=0(2022年2月5日21:59:26)
            previous_factor = 1
        else:
            previous_factor = np.percentile(previous_vector,max_p)
            previous_factor /= current_factor
            if  previous_factor == 0:
                previous_factor = 1
        return previous_factor, current_factor

    def train( self , x_train, x_test, y_train, y_test, learning_rate, max_epochs):
        mses = [ ]
        TRA = []
        TEA = []
        MSE = []
        l00, l01, l02 = np.array([]), np.array([]), np.array([])
        for i in range(max_epochs):

            if i <= 0.1 * max_epochs:
                learning_rate += 1 / (0.1 * max_epochs) * 0.042
            elif 0.6 * max_epochs >= i > 0.1 * max_epochs:
                learning_rate -= 1 / (0.5 * max_epochs) * 0.045

            start = time.time()
            if  i%1 == 0:
                    mse = 0
                    for k in range(len( x_train )):
                        mse  += np.square( y_train[k] - self.feed_forward(x_train[k].reshape(1,x_train[k].shape[0]))[0])
                    #len( x_train )
                    mse = mse / 2
                    mses.append(mse)
                    print('Epoch:#%s,MSW:'%( i),file=f,flush =True )
                    print('Epoch:#%s,MSW:'%( i),file=sys.stdout)
                    print(mse,file=f,flush =True )
                    print(mse,file=sys.stdout)
                    print("MSE: %.5f" % np.sum(mse),file=f,flush =True)
                    print("MSE: %.5f" % np.sum(mse), file=sys.stdout)
                    MSE.append(np.sum(mse))
                    test_accurary =  self.accuracy(x_test,y_test)
                    train_accurary = self.accuracy(x_train,y_train)
                    tra = train_accurary
                    tea = test_accurary
                    TRA.append(train_accurary)
                    TEA.append(test_accurary)
                    with open('TRA.txt', 'w') as E:
                        E.write(str(TRA))
                    with open('TEA.txt', 'w') as F:
                        F.write(str(TEA))
                    with open('MSE.txt', 'w') as G:
                        G.write(str(MSE))
                    print()
                    print('TestAccuracy: %.2f'%test_accurary,file=f,flush =True )
                    print('TestAccuracy: %.2f'%test_accurary,file=sys.stdout)
                    print('TrainAccuracy: %.2f'%train_accurary,file=f,flush =True )
                    print('TrainAccuracy: %.2f'%train_accurary,file=sys.stdout)

            """
            if i > 99:
                T = 0
            elif i == 40:
                learning_rate = 0.002
            elif i == 55:
                learning_rate /= 2
            elif i == 65:
                learning_rate /= 5
            elif i ==70:
                learning_rate /= 5
            else:
                T = 1
            """
            """
            if 90 > tea > 85:  # 删除了训练集精度对学习率的影响
                learning_rate = 0.04  # 若此处仍为0.03，容易掉入局部最优解(2022年2月5日12:48:18)
            elif 95 > tea > 90:
                if learning_rate / 2 > 0.025:  # 0.001增加为0.0015(2022年2月4日19:43:38)
                    # 1.1.0.4 若lr锁定在0.0015，测试集精度会卡在97.55与97.35之间，训练集最高为99.59并不再进行改变，因此加到0.0018(2022年2月5日01:33:53)
                    # 1.1.0.5 若lr锁定在0.0018，测试集精度达到99.18
                    # 当tea<95时，若学习率为0.002，则导致训练停滞，当tea小于95时，lr为0.004依旧停滞，调整为0.01(2022年2月6日10:13:38)；0.01还是不足，增加至0.02(2022年2月6日11:15:37)
                    learning_rate /= 2
                else:
                    learning_rate = 0.015
            elif 97 > tea >= 95 :
                learning_rate = 0.012
            elif 98 > tea > 97:
                if learning_rate / 2 > 0.006:  # lr为0.004太低，难以进一步更新，下一步扩增至0.01；0.01不足，0.02
                    learning_rate /= 2
                else:
                    learning_rate = 0.008
            elif tea > 98:
                learning_rate = 0.004
            else:
                learning_rate = 0.05
            """
            print("---------------")
            print(learning_rate)
            if i >= 1:  # 更改于2022年2月5日21:53:34
                T = 0
            else:
                T = 1

            for j in range(len(x_train)):
                if j == len(x_train) - 1 and T == 1:
                    T = 1.5
                    l0 = self._layers[0].last_activation_number
                    l1 = self._layers[1].last_activation_number
                    l2 = self._layers[2].last_activation_frequency
                    l00 = np.append(l00, l0)
                    l01 = np.append(l01, l1)
                    l02 = np.append(l02, l2)
                self.backpropagation(x_train[j].reshape(1,x_train[j].shape[0]),y_train[j],learning_rate,T,l00, l01, l02, i)
                if T == 1:
                    l0 = self._layers[0].last_activation_number
                    l1 = self._layers[1].last_activation_number
                    l2 = self._layers[2].last_activation_frequency
                    l00 = np.append(l00, l0)
                    l01 = np.append(l01, l1)
                    l02 = np.append(l02, l2)
                else:
                    l00, l01, l02 = [], [], []
            end = time.time()
            print("该轮花费%.04f秒\n" % (end - start))
        return mses

    def accuracy(self, x_test_accur , y_test_accur ):
        accuracy = 0
        for k in range(len( x_test_accur )):
            y_pre = self.feed_forward(x_test_accur[k].reshape(1,x_test_accur[k].shape[0]))[0]
            accuracy += self.is_true( y_pre ,y_test_accur[k])
        accuracy = accuracy/len(x_test_accur)
        return accuracy*100


    def is_true(self,y_pre,y_true):
        y_pre_lab = self.prob2lab(y_pre)
        y_true = np.array( y_true )
        if  np.array_equal(y_pre_lab,y_true) :
            return 1
        else :
            return 0

    def prob2lab(self,y_pre):
        zero_y = np.zeros_like(y_pre)
        max_position = int(np.argmax( y_pre ))
        zero_y[ max_position ] = 1
        return  zero_y
    
    def predict(self, x_test):
        return self.feed_forward(x_test.reshape(1,x_test.shape[0]))[0]

    def test( self , x, y):
            start = time.time()
            test_accurary =  self.accuracy(x,y)
            print('TestAccuracy: %.2f'%test_accurary,file=sys.stdout)
            end = time.time()
            print("该轮花费%.04f秒\n" % (end - start))