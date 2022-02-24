import numpy as np

class lr_schedule(object):
    def __init__(self,name):
        self.name = name
    
    def __call__(self, parameter):
        pass

class Solid_lr(lr_schedule):
    def __init__(self, name):
        super(Solid_lr, self).__init__(name)

    def __call__(self, parameter, epoch):
        """ 前两个对比实验 lr = 1e-3, 后两个对比实验 lr = 1e-1 """
        parameter['learning_rate'] = 1e-1

class Const_decay_lr(lr_schedule):
    def __init__(self,name):
        super(Const_decay_lr, self).__init__(name)
    
    """ 训练到达一定的阶段，按照经验选择一个学习率，一般呈衰减趋势 """
    def __call__(self, parameter, epoch):
        status_dict = {
            40 : 0.001,
            60 : 0.0009,
            80 : 0.0005,
            90 : 0.0001,
        }
        if epoch in status_dict :
            parameter['learning_rate'] = status_dict[epoch]

class Inverse_time_decay_lr(lr_schedule):
    def __init__(self,name):
        super(Inverse_time_decay_lr, self).__init__(name)

    """ lr = lr / (1 + beta * t) """
    def __call__(self, parameter, epoch, beta=0.1):
        parameter['learning_rate'] = parameter['learning_rate'] / (1 + beta * epoch)

class Exponential_decay_lr(lr_schedule):
    def __init__(self,name):
        super(Exponential_decay_lr, self).__init__(name)

    """ lr = lr * (beta ** t) """
    def __call__(self, parameter, epoch, beta=0.96):
        parameter['learning_rate'] = parameter['learning_rate'] * (beta ** epoch)


class Natural_Exp_decay_lr(lr_schedule):
    def __init__(self,name):
        super(Natural_Exp_decay_lr, self).__init__(name)

    """ lr = lr * exp(-beta * t) """
    def __call__(self, parameter, epoch, beta=0.04):
        parameter['learning_rate'] = parameter['learning_rate'] * np.exp(-beta * epoch)

class cos_decay_lr(lr_schedule):
    def __init__(self,name):
        super(cos_decay_lr, self).__init__(name)

    """ lr = 1/2 * lr * (1 + cos(t pi / T))"""
    def __call__(self, parameter, epoch, T=100):
        parameter['learning_rate'] = parameter['learning_rate'] / 2 * (1 + np.cos(epoch / T * np.pi))

class Gradual_Warm_up_lr(lr_schedule):
    def __init__(self, name):
        super(Gradual_Warm_up_lr, self).__init__(name)

    """ lr = t / T lr_0 """
    def __call__(self, parameter, epoch, T=100, lr_=1e-1):
        parameter['learning_rate'] = epoch / T * lr_

class Loop_lr(lr_schedule):
    def __init__(self, name):
        super(Loop_lr, self).__init__(name)
    
    def __call__(self, parameter, epoch, step_gamma=0.98, init_lr=1e-1):
        parameter['learning_rate'] = step_gamma * parameter['learning_rate']
        if epoch % 50 == 0:
            parameter['learning_rate'] = init_lr

""" 较复杂的自适应方法未实现 """

def sgdr_lr(parameter):
    pass

def AdaGrad(parameter):
    pass

def RMSprop(parameter):
    pass

def Adam(parameter):
    pass

if __name__ == '__main__':
    # print(np.exp(-4))
    pass