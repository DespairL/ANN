import numpy as np

def onehot(label, class_num):
    temp = np.eye(class_num)
    ret = temp[label]
    return ret

def decode_onehot(label,class_num):
    return np.argmax(label,axis=1)

def get_accuracy(pred, true):
    # print(np.argmax(pred, axis=1))
    # print(np.argmax(true, axis=1))
    correct_num = np.sum(np.argmax(pred, axis=1) == true)
    total_num = len(true)
    return correct_num / total_num

def normalize(x):
    min = np.min(x,axis=1,keepdims=True)
    max = np.max(x,axis=1,keepdims=True)
    x = (x - min) / (max - min)
    return x

def easy_normlize(x):
    x = (x - 128.0) / 255.0
    return x