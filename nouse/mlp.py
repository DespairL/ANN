import numpy as np

def ReLU(x):
    return np.maximum(0,x)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    s = np.max(x, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] 
    return e_x / div

class Activate:
    def __init__(self, active_function):
        self.active_function = active_function

def one_hot(label, label_numbers):
    if type(label) == np.ndarray:
        ret = np.zeros((label.shape[0], label_numbers), dtype=np.int64)
        # 默认从0开始!!
        for i in range(label.shape[0]):
            ret[i][int(label[i])] = 1
    else :
        ret = np.zeros(label_numbers, dtype=np.int64)
        ret[int(label)] = 1
    return ret

# 需要保证true_y是真实选定的lable，那么这个函数与pytorch官方给定的交叉熵函数就等价了
def cross_entropy_loss(pred_y, true_y):
    pred_y = np.log(softmax(pred_y))
    ret = 0
    index = 0
    for y in pred_y:
        ret -= y[true_y[index]]
        index += 1
    ret /= index
    return ret

class Linear_Layer:
    def __init__(self, input_length, output_length, bias=True):
        # Baseline : 参数随机初始化
        self.weight = np.random.randn(input_length, output_length)
        if bias :
            self.bias = np.random.randn(output_length)
        else :
            self.bias = np.zeros(output_length)
        # self.active_function = active_function
    
    def forward(self, x, active_function=Sigmoid):
        return x @ self.weight + self.bias

    # x为上一层的输入
    def backward(self, delta_weight, delta_bias, learning_rate=0.5):
        self.weight += learning_rate * delta_weight
        self.bias += learning_rate * delta_bias
        

class MLP:
    def __init__(self):
        self.layer1 = Linear_Layer(784, 256)
        self.layer2 = Linear_Layer(256, 256)
        self.layer3 = Linear_Layer(256, 64)
        self.layer4 = Linear_Layer(64, 10) # MNIST 0-9 10个label
    
    def forward(self, x, active_function=Sigmoid):
        x = active_function(self.layer1.forward(x))
        x = active_function(self.layer2.forward(x))
        x = active_function(self.layer3.forward(x))
        x = self.layer4.forward(x)
        x = softmax(x)
        return x
    """ TODO: 实现反向传播 """
    def backward(self, x, output, y, learning_rate=0.5, active_function=Sigmoid):
        out_layer1 = active_function(self.layer1.forward(x))
        out_layer2 = active_function(self.layer2.forward(out_layer1))
        out_layer3 = active_function(self.layer2.forward(out_layer2))
        gradient_layer4 = (y - output) * (output) * (1 - output)
        print(gradient_layer4.shape)

        print(np.dot(gradient_layer4, self.layer4.weight.T).shape)
        gradient_layer3 = out_layer3 * (1 - out_layer3) * (np.dot(gradient_layer4, self.layer4.weight.T))
        
        delta_weight_layer1 = np.dot(x.T, e) / x.shape[0]
        delta_weight_layer3 = np.dot(out_layer1.T, g) / x.shape[0]
        delta_bias_layer1 = -np.mean(e, axis = 0)
        delta_bias_layer3 = -np.mean(g, axis = 0)
        self.layer1.backward(delta_weight_layer1, delta_bias_layer1, learning_rate)
        self.layer3.backward(delta_weight_layer3, delta_bias_layer3, learning_rate)


    def train(self, train_x, train_y):
        encode_y = one_hot(train_y, label_numbers=10)
        output = self.forward(train_x)
        loss = cross_entropy_loss(output, train_y.astype(int))
        self.backward(train_x, output, encode_y)
        # pass

    def test(self, test_x, test_y):
        self.total_loss = 0
        self.correct = 0
        self.correct_rate = 0
        output = self.forward(test_x)
        self.test_loss = cross_entropy_loss(output, test_y.astype(int))
        pred = np.argmax(output, axis=1)
        total_num = len(test_y)
        for i in range(total_num) :
            if test_y[i] == pred[i] : 
                self.correct += 1
        self.correct_rate = self.correct / total_num
        print('Test set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        self.test_loss, self.correct, total_num,
        100. * self.correct_rate))

class MLP_mini:
    def __init__(self):
        self.layer1 = Linear_Layer(784, 256)
        self.layer2 = Linear_Layer(256, 10) # MNIST 0-9 10个label
    
    def forward(self, x, active_function=Sigmoid):
        x = active_function(self.layer1.forward(x))
        x = softmax(self.layer2.forward(x))
        return x
    
    """ TODO: 实现反向传播 """
    def backward(self, x, output, y, learning_rate=1, active_function=Sigmoid):
        out_layer1 = active_function(self.layer1.forward(x))
        out_layer2 = active_function(self.layer2.forward(out_layer1))

        gradient_layer2 = (y - out_layer2) * (out_layer2) * (1 - out_layer2)
        # print(gradient_layer2.shape)

        # print(np.dot(gradient_layer2, self.layer2.weight.T).shape)
        gradient_layer1 = out_layer1 * (1 - out_layer1) * (np.dot(gradient_layer2, self.layer2.weight.T))
        
        delta_weight_layer1 = np.dot(x.T, gradient_layer1) / x.shape[0]
        delta_weight_layer2 = np.dot(out_layer1.T, gradient_layer2) / x.shape[0]
        delta_bias_layer1 = -np.mean(gradient_layer1, axis = 0)
        delta_bias_layer2 = -np.mean(gradient_layer2, axis = 0)
        self.layer1.backward(delta_weight_layer1, delta_bias_layer1, learning_rate)
        self.layer2.backward(delta_weight_layer2, delta_bias_layer2, learning_rate)


    def train(self, train_x, train_y):
        encode_y = one_hot(train_y, label_numbers=10)
        output = self.forward(train_x)
        loss = cross_entropy_loss(output, train_y.astype(int))
        self.backward(train_x, output, encode_y)
        # pass

    def test(self, test_x, test_y):
        self.total_loss = 0
        self.correct = 0
        self.correct_rate = 0
        output = self.forward(test_x)
        test_y = test_y.astype(int)
        self.test_loss = cross_entropy_loss(output, test_y)
        pred = np.argmax(output, axis=1)
        total_num = len(test_y)
        self.correct = np.sum(pred == test_y)
        self.correct_rate = self.correct / total_num
        print("Test set: loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
             self.test_loss, self.correct, total_num, 100.0 * self.correct_rate))

def test_one_hot(input):

    import torch
    import torch.nn.functional as F 

    input = input.astype(np.int64)
    test1 = F.one_hot(torch.from_numpy(input), num_classes=10)
    test2 = one_hot(input, label_numbers=10)

    print(test1)
    print(test2)

    assert test1.all() == test2.all()

    print("one-hot encoder ok !")

def test_cross_entropy_loss():

    import torch
    import torch.nn.functional as F 

    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    # print(input, target, sep='\n')
    ground_truth = F.cross_entropy(input, target)
    print(f"pytorch entropy loss : {ground_truth.item()}")
    my_loss = cross_entropy_loss(input.detach().numpy(), target.detach().numpy())
    print(f"my entorpy loss : {my_loss}")
    
    assert np.abs(my_loss - ground_truth.item()) < 2 * 1e-4

    print("cross_entropy_loss ok !")


if __name__ == "__main__" :
    # x1 = np.array([[1,2,3,6],[2,4,5,6],[1,2,3,6]])
    # print(softmax(x1))
    # print(Sigmoid(1))
    
    from load_data import load_data

    model = MLP_mini()
    train_x, train_y, test_x, test_y = load_data()

    """ test one-hot """ 
    # print(train_y[0]) # 5.0 we need to one-hot encoder to encode 
    # print(one_hot(train_y[0], label_numbers=10)) 
    # print(one_hot(train_y, label_numbers=10)) 
    # test_one_hot(train_y)

    """ test cross entropy loss """ 
    # test_cross_entropy_loss()

    """ begin train -- have a try """
    train_x = train_x.reshape(-1, 784)
    test_x = test_x.reshape(-1, 784)
    # print(train_x.shape, test_x.shape)
    # out = model.forward(train_x) # (60000,10)
    # print(out.shape)
    for epoch in range(500):
        model.train(train_x, train_y)
        if epoch % 2 == 0 : 
            model.test(test_x, test_y)
            



    