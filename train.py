import numpy as np
from dataloader import Dataloader
from utils import onehot,get_accuracy

def train(model, loss, parameter, x, y, batch_size, reg):
    loss_list = []
    accuracy_list = []
    mean_loss = 0
    mean_accuracy = 0
    ret_loss = []
    ret_accrancy = []
    count = 0
    
    for batch_x, batch_y in Dataloader(x, y, batch_size):
        # print(batch_x,batch_y)
        true_y = onehot(batch_y, 10)
        output = model.forward(batch_x)
        # print(output)
        out_loss = loss.forward(output, true_y)
        grad = loss.backward(output, true_y)
        if reg != None:
            # print(grad.shape)
            weight = model.get_final_weight()
            out_loss = reg.forward(output, out_loss, weight)
            model.backward(grad)
            grad_w = model.get_final_grad_w()
            # print(grad_w.shape)
            grad_w += reg.backward(output, true_y)
            model.set_final_grad_w(grad_w)
        else :
            model.backward(grad)
        model.update(parameter)
        # print(output)
        accuracy = get_accuracy(output, batch_y)
        loss_list.append(out_loss)
        accuracy_list.append(accuracy)
        count += 1
        # print(accuracy)
        """ 每50轮 计算训练阶段性的loss、accrancy，返回以便于保存画图 """
        if count % 50 == 0 :
            mean_accuracy = np.mean(accuracy_list)
            mean_loss  = np.mean(loss_list)
            ret_loss.append(mean_loss)
            ret_accrancy.append(mean_accuracy)
            accuracy_list = []
            loss_list = []
            print(f"Iteration:{count:3}, Batch Loss:{mean_loss:.8f}, Batch accuracy:{mean_accuracy:.8f}")
    return ret_loss, ret_accrancy

