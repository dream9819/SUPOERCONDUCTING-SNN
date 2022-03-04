import numpy as np

class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None, delta=None,
                 lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.weights = (weights if weights is not None
                        else np.random.uniform(0, 1, (n_input, n_neurons)) / 10)
        self.bias = bias if bias is not None else np.ones(n_neurons) * 1
        self.activation = activation
        self.last_activation_number = None
        self.last_activation_frequency = None
        self.error = None
        self.delta = None
        self.theta_t = 3  # 到了后期偏置总是超过3.9，增大阈值与初始偏置保证初始动态偏置为2
        self.previous_factor = 1
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.m = 0
        self.v = 0
        self.m_b = 0
        self.v_b = 0
        self.m_w = 0
        self.v_w = 0
        self.momentum_weight = 0
        self.momentum_bias = 0

        self.k = 1000 * 4 / 3750
        self.previous_factor = 1
        self.current_factor = 1

    def activate(self , x):
        r = np.dot( x[0],  self.weights ) + self.bias
        self.last_activation_number , self.last_activation_frequency = self._apply_activation( r)
        #self.max_p_number = np.percentile(self.last_activation_number, max_p = 99)
        return self.last_activation_number, self.last_activation_frequency#,self.max_p_number

    def _apply_activation( self ,r ):
        if self.activation is None:
            return r
        elif self.activation == 'lif_relu':
            k = 1000 * 4/3750
            f = [ ]
            for i in r:
                if i >= self.theta_t:
                    f.append( i*k )
                else:
                    f.append(0)
            f = np.array(f)
            y = ( self.theta_t*np.ones(self.bias.shape[0])) - self.bias
            N = np.divide( r - self.bias, y )
            N = N // 1
            return N , f
        elif self.activation == 'sigmoid':
            k = 4/3750
            f = [ ]
            for i in r:
                if i > self.theta_t:
                    f.append( i*k )
                else:
                    f.append(0)
            f = np.array(f)
            y = ( self.theta_t*np.ones(self.bias.shape[0])) - self.bias
            N = np.divide( r , y )
            N = N // 1
            return 1.0/(1.0 + np.exp(-N)),np.NaN
        elif self.activation == 'softmax':
            k = 1000 * 4/3750
            f = [ ]
            for i in r:
                if i > self.theta_t:
                    f.append( i*k )
                else:
                    f.append(0)
            f = np.array(f)
            y = ( self.theta_t*np.ones(self.bias.shape[0])) - self.bias
            N = np.divide(r, y)
            N = N // 1
            #N -= np.max(N)
            #exps = np.exp(N)

            f -= np.max(f)
            exps1 = np.exp(f)
            S = exps1/np.sum(exps1) # 更改于2022年2月5日21:53:34
            #K = np.divide(r, y) // 1
            return S, N

    def apply_activation_derivative ( self ,  r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'lif_relu':
            grad = np.array(r , copy = True)
            grad [ r <= 0 ] = 0
            grad [ r > 0 ] = 1/self.theta_t
            return grad
        elif self.activation == 'sigmoid':
            return r*( 1.0 - r )
        elif self.activation == 'softmax':
            return np.diag(r)-np.outer(r,r)
        return np.ones_like(r)
