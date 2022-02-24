from network import NetWork
from loss_function import EuclideanLoss,MSEloss, HuberLoss
from loss_function import MAEloss,SoftmaxCrossEntropyLoss2
from layer import Linear,ReLu,Tanh
from load_data import load_data
from train import train
from test import test
from utils import easy_normlize
from lr_schedule import Solid_lr,Const_decay_lr,Inverse_time_decay_lr
from lr_schedule import Exponential_decay_lr,Natural_Exp_decay_lr,cos_decay_lr
from lr_schedule import Gradual_Warm_up_lr,Loop_lr
from reg import L1_regular, L2_regular, earlystop
import numpy as np
import os

def Multiperception():
    model = NetWork()
    model.append(Linear('LinearLayer1', 784, 256, 0.0001))
    model.append(Tanh('Activate1'))
    model.append(Linear('LinearLayer2', 256, 10, 0.0001))
    # model.append(ReLu('Activate2'))
    Parameter = {
        # 'momentum': 0.6,
        # 'weight_decay':1e-4,
        'learning_rate' : 0.15,
        'step_gamma' : 1,
        'batch_size' : 128,
        'max_epoch' : 100,
        'test_epoch' : 10
    }
    return model, Parameter

def save_train_test(args, save_train_loss, save_train_accuracy, 
            save_test_loss, save_test_accuracy, lr_schedule='solid_lr', reg='no reg'):
    if not os.path.exists('./save/') :
        os.makedirs('./save/')
    if reg != 'no reg':
        np.save('./save/' + args + '_'+ lr_schedule + reg + '_trainloss', save_train_loss)
        np.save('./save/' + args + '_'+ lr_schedule + reg + '_trainacc', save_train_accuracy)
        np.save('./save/' + args + '_'+ lr_schedule + reg + '_testloss', save_test_loss)
        np.save('./save/' + args + '_'+ lr_schedule + reg + '_testacc', save_test_accuracy)
    else :
        np.save('./save/' + args + '_'+ lr_schedule + '_trainloss', save_train_loss)
        np.save('./save/' + args + '_'+ lr_schedule + '_trainacc', save_train_accuracy)
        np.save('./save/' + args + '_'+ lr_schedule + '_testloss', save_test_loss)
        np.save('./save/' + args + '_'+ lr_schedule + '_testacc', save_test_accuracy)
    print(' The result of ' + args + ' has been save. Using' + lr_schedule + ' and ' + reg)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()

    """ 归一化 train_x test_x  """
    train_x = easy_normlize(train_x)
    test_x = easy_normlize(test_x)
    train_y = train_y.astype(int)
    test_y = test_y.astype(int)

    loss_function_total = {
        'EuclideanLoss':EuclideanLoss('first_loss'),
        # 'SoftmaxCrossEntropyLoss':SoftmaxCrossEntropyLoss('second_loss'),
        'MSEloss':MSEloss('third_loss'),
        # 'MyCrossEntropyLoss':MyCrossEntropyLoss('dada'),
        'SoftmaxCrossEntropyLoss2':SoftmaxCrossEntropyLoss2('sis'),
        'MAEloss':MAEloss('a'),
        'HuberLoss':HuberLoss('ad')
    }
    learning_rate_schedule_total = {
        'Solid_lr':Solid_lr('solid_lr'),
        'Const_decay_lr':Const_decay_lr('Const_decay_lr'),
        'Inverse_time_decay_lr':Inverse_time_decay_lr('Inverse_time_decay_lr'),
        'Exponential_decay_lr':Exponential_decay_lr('Exponential_decay_lr'),
        'Natural_Exp_decay_lr':Natural_Exp_decay_lr('Natural_Exp_decay_lr'),
        'Cos_decay_lr':cos_decay_lr('Cos_decay_lr'),
        'Gradual_Warm_up_lr':Gradual_Warm_up_lr('Gradual_Warm_up_lr'),
        'Loop_lr':Loop_lr('Loop_lr'),
    }
    regular_total = {
        'L1_regular':L1_regular('L1_regular'),
        'L2_regular':L2_regular('L2_regular'),
        'earlystop':earlystop('earlystop', 5),
        'None':None
    }
    
    """ earlystop 时开启 """
    early = regular_total['earlystop']

    reg = regular_total['L2_regular']
    loss = loss_function_total['SoftmaxCrossEntropyLoss2']
    model, parameter = Multiperception()
    lr_schedule = learning_rate_schedule_total['Const_decay_lr']
    save_train_loss = []
    save_train_accuracy = []
    save_test_loss = []
    save_test_accuracy = []

    for epoch in range(parameter['max_epoch']):
        ret_save_loss, ret_save_accuracy = train(model, loss, parameter, train_x, train_y, parameter['batch_size'], reg)
        save_train_loss += ret_save_loss
        save_train_accuracy += ret_save_accuracy
        ret_test_loss, ret_test_accuracy = test(model, loss, test_x, test_y, parameter['batch_size'])
        
        """ earlystop 时开启 """
        goon_train = early(ret_test_accuracy)
        if not goon_train:
             break

        save_test_loss.append(ret_test_loss) 
        save_test_accuracy.append(ret_test_accuracy)
        # print(save_test_loss,save_test_accuracy)
        
        
        lr_schedule(parameter, epoch)
        
    
    assert save_test_loss != []
    assert save_test_accuracy != []
    
    # save_train_test('baseline',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)
    # save_train_test('EuclideanLoss',save_train_loss,save_train_accuracy,save_test_loss,save_test_accurany)
    # save_train_test('MSEloss',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)
    # save_train_test('HuberLoss',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)
    # save_train_test('MAEloss',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)

    # save_train_test('baselinezeroinit',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)
    # save_train_test('baselineuniforminit',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)
    # save_train_test('baselinexavierinit_tanh',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)
    # save_train_test('baselineheinit',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)
    # save_train_test('baselineorthogonalinit',save_train_loss,save_train_accuracy,save_test_loss,save_test_accuracy)

    # save_train_test('baseline',save_train_loss,save_train_accuracy,
    #              save_test_loss,save_test_accuracy,lr_schedule=lr_schedule.name,
    #            reg=reg.name if reg != None else 'no reg')

    save_train_test('find_best', save_train_loss, save_train_accuracy, save_test_loss, save_test_accuracy)
