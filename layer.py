import numpy as np

class Layer(object):
    """ can_update 控制该层网络是否可以需要更新参数 """
    def __init__(self, name, can_update=False):
        self.name = name
        self.can_update = can_update
        self.save_data = None
    
    """ 前向传播 """
    def forward(self, x):
        pass

    """ 逆向传播 """
    def backward(self, grad):
        pass

    """ 更新参数 """
    def update(self, config):
        pass

    """ 保存公共表达式，节省计算开销 """
    def save_for_backward(self, x):
        self.save_data = x

class ReLu(Layer):
    def __init__(self, name):
        super(ReLu, self).__init__(name)

    def forward(self, x):
        self.save_for_backward(x)
        x[x <= 0] = 0
        return x

    def backward(self, grad):
        # print(type(grad))
        grad[self.save_data <= 0] = 0
        return grad

class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        self.save_for_backward(1. / (1. + np.exp(-input)))
        return self.save_data

    def backward(self, grad):
        return self.save_data * (1. - self.save_data) * grad

class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def Sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self, x):
        self.save_for_backward(2 * self.Sigmoid(2 * x) - 1)
        return self.save_data

    def backward(self, grad):
        return (1 - self.save_data ** 2) * grad

class Linear(Layer):
    def __init__(self, name, input_dim, output_dim, init_weight_args):
        super(Linear,self).__init__(name, can_update=True)
        self.input_dim = input_dim 
        self.output_dim = output_dim
        """ 随机初始化 """
        # self.weight = init_weight_args * np.random.randn(input_dim, output_dim)
        """ 全零初始化 """
        # self.weight = np.zeros((input_dim, output_dim))
        """ 基于固定方差的参数初始化 均匀分布 """
        # self.weight = 0.01 * np.random.uniform(low=-0.5,high=0.5,size=(input_dim, output_dim))
        """ Xaiver初始化 """
        # M_{1} = 256, M_{2} = 10 -> M_{l} 第l层的权重矩阵
        # r = np.sqrt(6 / (256 + 10))
        # self.weight = np.random.uniform(low=-r, high=r, size=(input_dim, output_dim))
        """ He初始化 """
        # r = np.sqrt(6 / 256)
        # self.weight = np.random.uniform(low=-r, high=r, size=(input_dim, output_dim))
        """ 正交初始化 """
        # 用均值为0方差为1的高斯分布初始化
        T = np.random.normal(loc=0.0, scale=1.0, size=(input_dim, output_dim))
        # 奇异值分解得到两个正交矩阵 | full_matrices很关键 True-> 输出方阵 False -> 非方阵
        U, S, vh = np.linalg.svd(T, full_matrices=False)
        # print(U.shape,S.shape,vh.shape)
        init_matrix = U if U.shape == (input_dim, output_dim) else vh
        self.weight = init_matrix

        self.grad_w = np.zeros((input_dim, output_dim))
        self.delta_w = np.zeros((input_dim, output_dim))

        self.bias = np.zeros(output_dim)
        self.grad_b = np.zeros(output_dim)
        self.delta_b = np.zeros(output_dim)

    def forward(self, x):
        self.save_for_backward(x)
        return x.dot(self.weight) + self.bias

    def backward(self, grad):
        self.grad_w = - self.save_data.T.dot(grad)
        self.grad_b = - grad.sum(axis=0)
        # print((grad @ self.weight.T).shape, self.forward(self.save_data).shape, self.weight.shape, grad.shape)
        # assert (grad @ self.weight.T).shape == self.forward(self.save_data).shape
        return  grad.dot(self.weight.T)

    def update(self, parameter):
        learning_rate = parameter['learning_rate']
        # momentum = parameter['momentum']
        # weight_decay = parameter['weight_decay']


        """ 平凡的更新 """
        self.delta_w = - learning_rate * self.grad_w
        self.weight += self.delta_w

        self.delta_b = - learning_rate * self.grad_b
        self.bias += self.delta_b

        # print(self.name,self.grad_w,self.grad_b)

        """ 加入 momentum, weight_decay """
        # self.delta_w = momentum * self.delta_w + (self.grad_w + weight_decay * self.weight)
        # self.weight = self.weight - learning_rate * self.delta_w

        # self.delta_b = momentum * self.delta_b + (self.grad_b + weight_decay * self.bias)
         #self.bias = self.bias - learning_rate * self.delta_b

    """ 正则化使用 """
    def get_weight(self):
        return self.weight
    
    def get_grad_w(self):
        return self.grad_w
    