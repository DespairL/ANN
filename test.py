import numpy as np
from dataloader import Dataloader
from utils import onehot,get_accuracy

def test(model, loss, x, y, batch_size):
    loss_list = []
    accuracy_list = []
    for batch_x, batch_y in Dataloader(x, y, batch_size):
        true_y = onehot(batch_y, class_num=10)
        output = model.forward(batch_x)
        batch_loss = loss.forward(output, true_y)
        batch_accuracy = get_accuracy(output, batch_y)
        loss_list.append(batch_loss)
        accuracy_list.append(batch_accuracy)
    mean_loss = np.mean(loss_list)
    mean_accuracy = np.mean(accuracy_list)
    print(f"Test : Batch Loss:{mean_loss:.8f}, Batch accuracy:{mean_accuracy:.8f}")
    return mean_loss, mean_accuracy
