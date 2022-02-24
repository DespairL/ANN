import numpy as np

class L1_regular(object):
    def __init__(self, name):
        self.name = name
        self.save_data = None

    def save_for_backward(self, x):
        self.save_data = x
    
    """ loss = loss + lambda * \sum |w| """
    def forward(self, x ,loss, weight, lamb=0.05):
        self.save_for_backward(weight)
        # print(self.save_data.shape)
        return loss + lamb * np.sum(np.abs(self.save_data),axis=1) / x.shape[0]

    """ lambda * sign(w) """
    def backward(self, x, y, lamb=0.05):
        return lamb * np.sign(self.save_data) / x.shape[0]

class L2_regular(object):
    def __init__(self, name):
        self.name = name
        self.save_data = None

    def save_for_backward(self, x):
        self.save_data = x
    
    """ loss = loss + lambda * \sum |w| ** 2 """
    def forward(self, x ,loss, weight, lamb=0.05):
        self.save_for_backward(weight)
        # print(self.save_data.shape)
        return loss + lamb * np.sum(np.dot(self.save_data, self.save_data.T),axis=1) / x.shape[0]

    """ 2 * lambda * (w) """
    def backward(self, x, y, lamb=0.05):
        return 2 * lamb * self.save_data / x.shape[0]

""" 实现有点问题，期末没时间改了 """
class dropout(object):
    def __init__(self,name):
        self.name = name 
        self.save_data = None

    def save_for_backward(self, x):
        self.save_data = x
    
    def forward(self, weight, p=0.5):
        temp = np.random.binomial(1, p, size=weight.shape[1:]) / (1 - p)
        temp = temp.reshape(-1)
        dropout_ret = weight * temp
        return dropout_ret

class earlystop(object):
    def __init__(self,name,iter_stop):
        self.name = name
        self.record_acc = 0
        self.count = iter_stop
        self.iter_stop = iter_stop

    def __call__(self,now_acc):
        if now_acc > self.record_acc:
            self.record_acc = now_acc
            self.count = self.iter_stop
            return True
        self.count -= 1
        if self.count == 0:
            return False
        return True
