import numpy as np
import matplotlib.pyplot as plt
import os

def plot_acc(acc, path, label=None):
    acc = acc * 100.0
    total_epoch = len(acc) 
    epoch_list = list(range(0,total_epoch))
    plt.cla()
    plt.xlabel('epoches of batch')
    plt.ylabel('Accuracy:%')
    if label != None:
        plt.plot(epoch_list, acc, label=label)
    else :
        plt.plot(epoch_list, acc)
    # plt.savefig(path)
    # plt.show()

def plot_loss(loss, path, label=None):
    total_epoch = len(loss)
    epoch_list = list(range(0,total_epoch))
    plt.cla()
    plt.xlabel('epoches of batch')
    plt.ylabel('loss')
    if label != None:
        plt.plot(epoch_list, loss, label=label)
    else :
        plt.plot(epoch_list, loss)
    # plt.savefig(path)
    # plt.show()

def plot_baseline_result(ob, acc=True):
    plt.cla()
    path = './pics/baseline/'
    if not os.path.exists(path) :
        os.makedirs(path)
    obj = np.load('./save/baseline_solid_lr_' + ob + '.npy')
    if acc:
        acc = acc * 100.0
        total_epoch = len(acc) 
        epoch_list = list(range(0,total_epoch))
        plt.plot(epoch_list, acc)
    else :
        total_epoch = len(loss)
        epoch_list = list(range(0,total_epoch))
        plt.plot(epoch_list, loss)


def plot_args(args, label=None, lr='solid_lr'):
    path = './pics/' + args + '/'
    if not os.path.exists(path) :
        os.makedirs(path)
    if lr == 'solid_lr':
        train_acc = np.load('./save/' + args + '_solid_lr_trainacc.npy')
        train_loss = np.load('./save/' + args + '_solid_lr_trainloss.npy')
        test_acc = np.load('./save/' + args + '_solid_lr_testacc.npy')
        test_loss = np.load('./save/' + args + '_solid_lr_testloss.npy')
    else:
        train_acc = np.load('./save/' + args + '_'+ lr +'_trainacc.npy')
        train_loss = np.load('./save/' + args + '_'+ lr +'_trainloss.npy')
        test_acc = np.load('./save/' + args + '_'+ lr +'_testacc.npy')
        test_loss = np.load('./save/' + args + '_'+ lr +'_testloss.npy')
    plot_acc(train_acc, path + 'trainacc', label)
    base_trainacc = np.load('./save/baseline_solid_lr_trainacc.npy')
    plt.plot(list(range(0,len(base_trainacc))), base_trainacc * 100.0, label='baseline')
    plt.legend()
    # plt.show()
    plt.savefig(path + 'trainacc')
    plot_loss(train_loss, path + 'trainloss', label)
    base_trainloss = np.load('./save/baseline_solid_lr_trainloss.npy')
    plt.plot(list(range(0,len(base_trainloss))), base_trainloss, label='baseline')
    plt.legend()
    # plt.show()
    plt.savefig(path + 'trainloss')
    plot_acc(test_acc, path + 'testacc', label)
    base_testacc = np.load('./save/baseline_solid_lr_testacc.npy')
    plt.plot(list(range(0,len(base_testacc))), base_testacc * 100.0, label='baseline')
    plt.legend()
    # plt.show()
    plt.savefig(path + 'testacc')
    plot_loss(test_loss, path + 'testloss', label)
    base_testloss = np.load('./save/baseline_solid_lr_testloss.npy')
    plt.plot(list(range(0,len(base_testloss))), base_testloss, label='baseline')
    plt.legend()
    # plt.show()
    plt.savefig(path + 'testloss')



def plot_unit(path, unit_args, args_list, args_type, lr,lr_flag=False):
    plt.cla()
    plt.xlabel('epoches of batch')
    plt.ylabel(unit_args)
    for args in args_list:
        if lr != 'solid_lr' :
            lr = args
            args = 'baseline'
        temp_data = np.load('./save/' + args + '_' + lr + '_' + unit_args + '.npy')
        total_epoch = len(temp_data)
        epoch_list = list(range(0,total_epoch))
        if 'baseline' in args and args != 'baseline':
            args = args.replace('baseline', '')
        plt.plot(epoch_list, temp_data, label=lr)
        if lr_flag:
            lr = 'ad' # 任意不为'solid_lr'的值
    plt.legend()
    # plt.show()
    plt.savefig(path + unit_args)

def plot_diff(args_list, args_type, lr='solid_lr',lr_flag=False):
    plot_list = ['trainloss', 'trainacc', 'testloss', 'testacc']
    path = './pics/' + args_type + '/'
    if not os.path.exists(path) :
        os.makedirs(path)
    for unit_args in plot_list:
        plot_unit(path, unit_args, args_list, args_type, lr,lr_flag)

def plot_comp_reg():
    name = {
        'L1_regular':'baseline_Solid_lrL1_regular_',
        'L2_regular':'baseline_Solid_lrL2_regular_',
        'earlystop':'earlystop_solid_lr_'
    }
    plot_list = ['trainloss', 'trainacc', 'testloss', 'testacc']
    path = './pics/' + 'reg_diff' +'/'
    c = 1
    if not os.path.exists(path) :
        os.makedirs(path)
    for unit_args in plot_list:
        plt.cla()
        plt.xlabel('epoches of batch')
        plt.ylabel(unit_args)
        for reg in name.keys():
            if c == 1:
                baseline_data = np.load('./save/baseline_solid_lr_' + unit_args + '.npy')
                total_epoch = len(baseline_data)
                epoch_list = list(range(0,total_epoch))
                plt.plot(epoch_list, baseline_data, label='baseline')
                c += 1
            comp_data = np.load('./save/' + name[reg] + unit_args + '.npy')
            plt.plot(list(range(0,len(comp_data))), comp_data, label=reg)
        plt.legend()
        # plt.show()
        c = 1
        plt.savefig(path + unit_args)

def plot_args_d(args, label, lr='solid_lr'):
    path = './pics/' + args + label +'/'
    if not os.path.exists(path) :
        os.makedirs(path)
    if lr == 'solid_lr':
        trainacc = np.load('./save/' + args + '_solid_lr_trainacc.npy')
        trainloss = np.load('./save/' + args + '_solid_lr_trainloss.npy')
        testacc = np.load('./save/' + args + '_solid_lr_testacc.npy')
        testloss = np.load('./save/' + args + '_solid_lr_testloss.npy')
    else:
        trainacc = np.load('./save/' + args + '_'+ lr +'_trainacc.npy')
        trainloss = np.load('./save/' + args + '_'+ lr +'_trainloss.npy')
        testacc = np.load('./save/' + args + '_'+ lr +'_testacc.npy')
        testloss = np.load('./save/' + args + '_'+ lr +'_testloss.npy')
    base_trainacc = np.load('./save/baseline_solid_lr_trainacc.npy')
    base_trainloss = np.load('./save/baseline_solid_lr_trainloss.npy')
    base_testacc = np.load('./save/baseline_solid_lr_testacc.npy')
    base_testloss = np.load('./save/baseline_solid_lr_testloss.npy')

    plot_list = ['trainloss', 'trainacc', 'testloss', 'testacc']
    plot_dict = {
        'trainloss':(trainacc,base_trainacc),
        'trainacc':(trainloss,base_trainloss),
        'testloss':(testloss,base_testloss),
        'testacc':(testacc,base_testacc)
    }
    for unit_args in plot_list:
        sp = plot_dict[unit_args][0]
        base = plot_dict[unit_args][1]
        plt.cla()
        plt.xlabel('epoches of batch')
        plt.ylabel(unit_args)
        plt.plot(list(range(0,len(base))), base, label='baseline')
        plt.plot(list(range(0,len(sp))), sp, label=lr)
        plt.legend()
        # plt.show()
        plt.savefig(path + unit_args)

def print_final_acc(path):
    arr = np.load(path)
    return arr[-1]

def plot_maybe_best():
    plot_list = ['trainloss', 'trainacc', 'testloss', 'testacc']
    path = './pics/find_best/'
    if not os.path.exists(path) :
        os.makedirs(path)
    for unit_args in plot_list:
        plt.cla()
        plt.xlabel('epoches(train:batch_epoch,test:total_epoch)')
        plt.ylabel(unit_args)
        sp = np.load('./save/find_best_solid_lr_' + unit_args + '.npy')
        plt.plot(list(range(0,len(sp))), sp)
        # plt.legend()
        # plt.show()
        plt.savefig(path + unit_args)

if __name__ == '__main__':
    experiment_loss = ['MSEloss','MAEloss','HuberLoss','baseline']
    experiment_init_method = ['baseline', 'baselinezeroinit', 'baselineuniforminit', 'baselinexavierinit', 
                              'baselinexavierinit_tanh', 'baselineheinit', 'baselineorthogonalinit']
    experiment_lr_method = ['baseline', 'Const_decay_lr', 'Inverse_time_decay_lr', 'Exponential_decay_lr',
                            'Natural_Exp_decay_lr', 'Cos_decay_lr', 'Gradual_Warm_up_lr', 'Loop_lr']
    # plot_baseline_result()
    # plot_args('Euclideanloss')
    # plot_args('MSEloss', label='MSEloss')
    # plot_args('MAEloss', label='MAEloss')
    # plot_args('HuberLoss', label='HuberLoss')
    # plot_args('baselinezeroinit', label='zeroinit')
    # plot_args('baselineuniforminit', label='uniforminit')
    # plot_args('baselinexavierinit', label='xavierinit')
    # plot_args('baselinexavierinit_tanh', label='xavierinit_tanh')
    # plot_args('baselineheinit', label='heinit')
    # plot_args('baselineorthogonalinit', label='orthogonalinit')
    # plot_diff(experiment_loss, 'loss_diff')
    # plot_diff(experiment_init_method, 'init_diff')
    # plot_diff(experiment_lr_method, 'lr_diff', lr_flag=True)
    # plot_args_d('baseline', '2', 'Gradual_Warm_up_lr')
    # plot_args_d('baseline', '3', 'Loop_lr')
    # plot_comp_reg('L1_regular')
    # plot_comp_reg('L2_regular')


    plot_maybe_best()


    """ reg方法总体对比 """
    # plot_comp_reg()


    
