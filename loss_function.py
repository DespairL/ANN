import numpy as np

class Lossfunction(object):
    def __init__(self,name):
        self.name = name
    
    """ loss function expression loss(input=x, label=y) """
    def forward(self, x, y):
        pass

    """ Calculate  d loss / d x  =  (dloss/dx_1, dloss/dx_2,..., dloss/dx_n)"""
    def backward(self, x, y):
        pass

class EuclideanLoss(Lossfunction):
    def __init__(self, name):
        super(EuclideanLoss, self).__init__(name)
    
    def forward(self, x, y):
        return ((y - x) ** 2).mean(axis=0).sum() / 2
    
    def backward(self, x, y):
        return y - x

""" 该交叉熵函数实现有问题 """
class SoftmaxCrossEntropyLoss(Lossfunction):
    def __init__(self, name):
        super(SoftmaxCrossEntropyLoss, self).__init__(name)
        self.save_data = None
    
    def stable_deal(self,x):
        return x - np.max(x, axis=1, keepdims=True)

    def save(self,x):
        self.save_data = x

    def clean(self):
        self.save_data = None

    def softmax(self,x):
        x = self.stable_deal(x)
        softmax_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return softmax_x

    def log_softmax(self,x):
        x_ = self.stable_deal(x)
        e_x = np.exp(x)
        log_softmax_x = x_ - np.log(np.sum(e_x, axis=1, keepdims=True)) 
        self.save(log_softmax_x)
        return log_softmax_x

    def forward(self, x, y): 
        return -np.sum(self.log_softmax(x) * y) / x.shape[0]

    def backward(self, x, y):
        # print(( (1 - self.softmax(x)) * y  ) / x.shape[0])
        return -(np.exp(self.save_data) - y) / x.shape[0]

""" 该交叉熵函数能正常发挥 """
class SoftmaxCrossEntropyLoss2(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target): 
        shin = input - 1e-10
        loss = -np.sum((shin - np.log(np.sum(np.exp(shin), axis=1, keepdims=True))) * target) / input.shape[0]
        return loss

    def backward(self, input, target):
        shin = input - 1e-10
        prob = np.exp(shin - np.log(np.sum(np.exp(shin), axis=1, keepdims=True))) 
        return -(prob - target) / input.shape[0]

""" 没用 """
class MyCrossEntropyLoss(Lossfunction):
    def __init__(self, name):
        super(MyCrossEntropyLoss, self).__init__(name)

    def forward(self, x, y, epsilon=1e-12):
        x = np.clip(x, epsilon, 1-epsilon)
        N = x.shape[0]
        ce = - np.sum(y * np.log(x + 1e-4)) / N
        return ce

    def backward(self, x, y, epsilon=1e-12):
        x = np.clip(x, epsilon, 1-epsilon)
        N = x.shape[0]
        grad_ce = - (y / (x + 1e-4)) / N
        return grad_ce

class MSEloss(Lossfunction):
    def __init__(self, name):
        super(MSEloss, self).__init__(name)
    
    def forward(self, x, y):
        return ((x - y) ** 2).sum() / x.shape[0]

    def backward(self, x, y):
        return 2 * (y - x)  / x.shape[0]

class MAEloss(Lossfunction):
    def __init__(self, name):
        super(MAEloss, self).__init__(name)
    
    def forward(self, x, y):
        return np.abs(y - x).sum() / x.shape[0]
    
    def backward(self, x, y):
        return - np.sign(x - y) / x.shape[0]

class HuberLoss(Lossfunction):
    def __init__(self, name):
        super(HuberLoss, self).__init__(name)

    def forward(self, x, y, delta=1):
        dec = np.sum(np.abs(y - x))
        if dec < delta:
            return ((x - y) ** 2).sum() / x.shape[0]
        else:
            return delta * (np.abs(y - x).sum() / x.shape[0] - 0.5 * delta)

    def backward(self, x, y, delta=1):
        dec = np.sum(np.abs(y - x))
        if dec < delta:
            return 2 * (y - x)  / x.shape[0]
        else:
            return delta * (y - x) / x.shape[0]

if __name__ == '__main__':

    from torch.nn import CrossEntropyLoss
    import torch

    """ test croos entropy loss """
    predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
    targets = np.array([[0,0,0,1],
                   [0,0,0,1]])

    tensor_pre = torch.tensor(predictions.astype(float),requires_grad=True)
    tensor_tar = torch.tensor(targets.astype(float),requires_grad=True)
    # print(tensor_pre,tensor_tar)

    ans = 0.71355817782  #Correct answer
    loss = MyCrossEntropyLoss('test!')
    loss2 = SoftmaxCrossEntropyLoss2('qwfqw')
    loss_ = CrossEntropyLoss()

    right = loss_(tensor_pre, tensor_tar)
    right.backward()
    # print(tensor_pre.grad)

    x = loss.forward(predictions, targets)
    xx = loss2.forward(predictions, targets)
    grad__ = loss2.backward(predictions, targets)
    # print(grad__)
    assert np.isclose(x,ans)
    assert np.isclose(grad__.all(), tensor_pre.grad.all())
